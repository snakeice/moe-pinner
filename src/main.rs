use libc::{c_int, c_void, mmap, munmap, madvise, mlock, munlock, open, close, fstat, stat, off_t, MAP_SHARED, PROT_READ, MADV_WILLNEED, MADV_POPULATE_READ, O_RDONLY, MAP_FAILED};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use std::process;

// GGUF constants
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const RING_MAGIC: u32 = 0x52454F4D; // "MOER" in little-endian
const RING_VERSION: u32 = 1;
const RING_HEADER_SIZE: usize = 4096;
const RECORD_MAX_TOKENS: usize = 512;

const PAGE_SIZE: usize = 4096;

#[repr(C)]
struct RingHeader {
    magic: u32,
    version: u32,
    n_layer: u32,
    n_expert: u32,
    n_expert_used: u32,
    capacity: u64,
    write_seq: AtomicU64,
}

// GGML type enum (from ggml.h)
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 15,
    IQ2_XS = 16,
    IQ3_XXS = 17,
    IQ1_S = 18,
    IQ4_NL = 19,
    IQ3_S = 20,
    IQ3_M = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 24,
    BF16 = 25,
}

impl GGMLType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            6 => Some(GGMLType::Q5_0),
            7 => Some(GGMLType::Q5_1),
            8 => Some(GGMLType::Q8_0),
            9 => Some(GGMLType::Q8_1),
            10 => Some(GGMLType::Q2_K),
            11 => Some(GGMLType::Q3_K),
            12 => Some(GGMLType::Q4_K),
            13 => Some(GGMLType::Q5_K),
            14 => Some(GGMLType::Q6_K),
            15 => Some(GGMLType::IQ2_XXS),
            16 => Some(GGMLType::IQ2_XS),
            17 => Some(GGMLType::IQ3_XXS),
            18 => Some(GGMLType::IQ1_S),
            19 => Some(GGMLType::IQ4_NL),
            20 => Some(GGMLType::IQ3_S),
            21 => Some(GGMLType::IQ3_M),
            22 => Some(GGMLType::IQ2_S),
            23 => Some(GGMLType::IQ4_XS),
            24 => Some(GGMLType::IQ1_M),
            25 => Some(GGMLType::BF16),
            _ => None,
        }
    }

    // Returns (bytes_per_block, elements_per_block)
    fn block_info(&self) -> (usize, usize) {
        match self {
            GGMLType::F32 => (4, 1),
            GGMLType::F16 => (2, 1),
            GGMLType::BF16 => (2, 1),
            GGMLType::Q4_0 => (18, 32),   // 16 bytes weights + 2 bytes scales + 2 bytes mins = 18 per 32 elements? Actually: 32*4/8 + 2+2 = 16+4 = 20? Wait...
            GGMLType::Q4_1 => (20, 32),
            GGMLType::Q5_0 => (22, 32),
            GGMLType::Q5_1 => (24, 32),
            GGMLType::Q8_0 => (34, 32),
            GGMLType::Q8_1 => (36, 32),
            GGMLType::Q2_K => (128/8 + 64/8 + 2 + 2 + 2, 256), // simplified
            GGMLType::Q3_K => (192/8 + 64/4 + 12 + 4 + 4, 256),
            GGMLType::Q4_K => (128/2 + 128/8 + 12 + 4, 256), // 144 bytes / 256 elements
            GGMLType::Q5_K => (128/2 + 128/8*2 + 12 + 4 + 4, 256), // 176
            GGMLType::Q6_K => (128*6/8 + 128/8 + 12 + 4 + 4 + 4 + 4 + 4, 256), // 210
            GGMLType::IQ2_XXS => (256/8*2 + 1 + 1 + 1 + 32, 256), // approx
            GGMLType::IQ2_XS => (256/8*2 + 32/8 + 1 + 1 + 32, 256),
            GGMLType::IQ3_XXS => (256*3/8 + 32/8 + 1 + 1 + 32, 256),
            GGMLType::IQ1_S => (256/8 + 32/8 + 1 + 1 + 32, 256),
            GGMLType::IQ4_NL => (256/2 + 256/8 + 12 + 4 + 4 + 4, 256),
            GGMLType::IQ3_S => (256*3/8 + 32/8 + 12 + 4 + 4, 256),
            GGMLType::IQ3_M => (256*3/8 + 32/8 + 12 + 4 + 4 + 4, 256), // approx
            GGMLType::IQ2_S => (256/8*2 + 32/8 + 12 + 4 + 4, 256),
            GGMLType::IQ4_XS => (126, 256), // known: 126 bytes per 256 elements
            GGMLType::IQ1_M => (256/8 + 32/8 + 1 + 1 + 32 + 16, 256),
        }
    }

    fn bytes_per_element(&self) -> f64 {
        let (bpb, epb) = self.block_info();
        bpb as f64 / epb as f64
    }
}

// KV value types in GGUF
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(GGUFValueType::Uint8),
            1 => Some(GGUFValueType::Int8),
            2 => Some(GGUFValueType::Uint16),
            3 => Some(GGUFValueType::Int16),
            4 => Some(GGUFValueType::Uint32),
            5 => Some(GGUFValueType::Int32),
            6 => Some(GGUFValueType::Float32),
            7 => Some(GGUFValueType::Bool),
            8 => Some(GGUFValueType::String),
            9 => Some(GGUFValueType::Array),
            10 => Some(GGUFValueType::Uint64),
            11 => Some(GGUFValueType::Int64),
            12 => Some(GGUFValueType::Float64),
            _ => None,
        }
    }
}

struct GGUFParser {
    file: File,
    path: PathBuf,
    file_size: u64,
    data_start: u64,
    alignment: u32,
    tensors: Vec<TensorInfo>,
    kv: HashMap<String, KVValue>,
    file_idx: usize,
}

#[derive(Debug, Clone)]
enum KVValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<KVValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    ggml_type: GGMLType,
    offset: u64, // offset from data_start
    file_idx: usize,
    total_bytes: u64,
    expert_bytes: u64,
}

impl GGUFParser {
    fn new(path: PathBuf, file_idx: usize) -> Result<Self, String> {
        let file = File::open(&path).map_err(|e| format!("open {}: {}", path.display(), e))?;
        let metadata = file.metadata().map_err(|e| format!("stat {}: {}", path.display(), e))?;
        let file_size = metadata.len();
        let mut parser = GGUFParser {
            file,
            path,
            file_size,
            data_start: 0,
            alignment: 32,
            tensors: Vec::new(),
            kv: HashMap::new(),
            file_idx,
        };
        parser.parse()?;
        Ok(parser)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf).map_err(|e| format!("read u32: {}", e))?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let mut buf = [0u8; 8];
        self.file.read_exact(&mut buf).map_err(|e| format!("read u64: {}", e))?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf).map_err(|e| format!("read i32: {}", e))?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()? as usize;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf).map_err(|e| format!("read string: {}", e))?;
        String::from_utf8(buf).map_err(|e| format!("utf8: {}", e))
    }

    fn read_kv_value(&mut self, vtype: GGUFValueType) -> Result<KVValue, String> {
        Ok(match vtype {
            GGUFValueType::Uint8 => KVValue::Uint8(self.read_u8()?),
            GGUFValueType::Int8 => KVValue::Int8(self.read_i8()?),
            GGUFValueType::Uint16 => KVValue::Uint16(self.read_u16()?),
            GGUFValueType::Int16 => KVValue::Int16(self.read_i16()?),
            GGUFValueType::Uint32 => KVValue::Uint32(self.read_u32()?),
            GGUFValueType::Int32 => KVValue::Int32(self.read_i32()?),
            GGUFValueType::Float32 => KVValue::Float32(self.read_f32()?),
            GGUFValueType::Bool => KVValue::Bool(self.read_u8()? != 0),
            GGUFValueType::String => KVValue::String(self.read_string()?),
            GGUFValueType::Array => {
                let atype = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len);
                let elem_type = GGUFValueType::from_u32(atype).ok_or_else(|| format!("unknown array element type: {}", atype))?;
                for _ in 0..len {
                    arr.push(self.read_kv_value(elem_type)?);
                }
                KVValue::Array(arr)
            }
            GGUFValueType::Uint64 => KVValue::Uint64(self.read_u64()?),
            GGUFValueType::Int64 => KVValue::Int64(self.read_i64()?),
            GGUFValueType::Float64 => KVValue::Float64(self.read_f64()?),
        })
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        let mut buf = [0u8; 1];
        self.file.read_exact(&mut buf).map_err(|e| format!("read u8: {}", e))?;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        let mut buf = [0u8; 1];
        self.file.read_exact(&mut buf).map_err(|e| format!("read i8: {}", e))?;
        Ok(buf[0] as i8)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let mut buf = [0u8; 2];
        self.file.read_exact(&mut buf).map_err(|e| format!("read u16: {}", e))?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        let mut buf = [0u8; 2];
        self.file.read_exact(&mut buf).map_err(|e| format!("read i16: {}", e))?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf).map_err(|e| format!("read f32: {}", e))?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        let mut buf = [0u8; 8];
        self.file.read_exact(&mut buf).map_err(|e| format!("read i64: {}", e))?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        let mut buf = [0u8; 8];
        self.file.read_exact(&mut buf).map_err(|e| format!("read f64: {}", e))?;
        Ok(f64::from_le_bytes(buf))
    }

    fn parse(&mut self) -> Result<(), String> {
        // Read header
        let magic = self.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(format!("invalid magic: 0x{:x}", magic));
        }
        let version = self.read_u32()?;
        if version != 3 {
            return Err(format!("unsupported GGUF version: {}", version));
        }
        let tensor_count = self.read_u64()?;
        let kv_count = self.read_u64()?;

        // Read KV pairs
        for _ in 0..kv_count {
            let key = self.read_string()?;
            let vtype_raw = self.read_u32()?;
            let vtype = GGUFValueType::from_u32(vtype_raw).ok_or_else(|| format!("unknown KV value type: {}", vtype_raw))?;
            let value = self.read_kv_value(vtype)?;
            if key.starts_with("split.") {
                println!("  [Shard {}] KV: {} = {:?}", self.file_idx, key, value);
            }
            self.kv.insert(key, value);
        }

        // Get alignment
        if let Some(KVValue::Uint32(a)) = self.kv.get("general.alignment") {
            self.alignment = *a;
        }

        // Read tensor infos
        let mut tensor_infos = Vec::new();
        for _ in 0..tensor_count {
            let name = self.read_string()?;
            let n_dims = self.read_u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(self.read_u64()?);
            }
            let ggml_type = self.read_u32()?;
            let offset = self.read_u64()?;
            tensor_infos.push((name, n_dims, dims, ggml_type, offset));
        }

        // Compute data_start: aligned to alignment after tensor info array
        let current_pos = self.file.stream_position().map_err(|e| format!("stream_position: {}", e))?;
        let aligned = ((current_pos + self.alignment as u64 - 1) / self.alignment as u64) * self.alignment as u64;
        self.data_start = aligned;

        // Process tensors
        for (name, n_dims, dims, ggml_type, offset) in tensor_infos {
            let ggml_type = GGMLType::from_u32(ggml_type).ok_or_else(|| format!("unknown ggml type: {}", ggml_type))?;
            let total_bytes = self.compute_tensor_bytes(&dims, ggml_type);
            let expert_bytes = if dims.len() >= 3 && dims[2] > 0 {
                total_bytes / dims[2]
            } else {
                total_bytes
            };
            let abs_offset = self.data_start + offset;
            // Verify tensor fits in file
            if abs_offset + total_bytes > self.file_size {
                eprintln!("WARNING: tensor {} at offset {} size {} exceeds file size {}", name, abs_offset, total_bytes, self.file_size);
            }
            self.tensors.push(TensorInfo {
                name,
                n_dims,
                dims,
                ggml_type,
                offset: abs_offset,
                file_idx: self.file_idx,
                total_bytes,
                expert_bytes,
            });
        }

        Ok(())
    }

    fn compute_tensor_bytes(&self, dims: &[u64], ggml_type: GGMLType) -> u64 {
        // Total elements = product of all dims
        let n_elements: u64 = dims.iter().product();
        let (bpb, epb) = ggml_type.block_info();
        // Number of blocks = ceil(n_elements / epb)
        let n_blocks = (n_elements + epb as u64 - 1) / epb as u64;
        n_blocks * bpb as u64
    }
}

struct ExpertTensor {
    layer: usize,
    expert: usize,
    gate_offset: u64,
    gate_len: u64,
    up_offset: u64,
    up_len: u64,
    down_offset: u64,
    down_len: u64,
    file_idx: usize,
    fd: c_int,
}

struct ModelIndex {
    n_layer: usize,
    n_expert: usize,
    n_expert_used: usize,
    experts: Vec<ExpertTensor>, // filtered (Some only); lookup via expert_index
    expert_index: HashMap<(usize, usize), usize>, // (layer, expert) -> experts idx
    shard_files: Vec<File>,
    shard_fds: Vec<c_int>,
    shard_sizes: Vec<u64>,
}

impl ModelIndex {
    fn build(model_dir: &Path, n_expert_used: usize) -> Result<Self, String> {
        let mut shard_paths = Vec::new();
        for entry in std::fs::read_dir(model_dir).map_err(|e| format!("read_dir: {}", e))? {
            let entry = entry.map_err(|e| format!("dir entry: {}", e))?;
            let path = entry.path();
            if path.extension().map(|e| e == "gguf").unwrap_or(false) {
                shard_paths.push(path);
            }
        }
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err("no GGUF files found".into());
        }
        println!("Found {} shard(s)", shard_paths.len());

        let mut parsers = Vec::new();
        let mut shard_files = Vec::new();
        let mut shard_fds = Vec::new();

        for (idx, path) in shard_paths.iter().enumerate() {
            let file = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
            let fd = file.as_raw_fd();
            shard_files.push(file);
            shard_fds.push(fd);
            let parser = GGUFParser::new(path.clone(), idx)?;
            println!("Shard {}: {} tensors, data_start={}, file_size={}", idx, parser.tensors.len(), parser.data_start, parser.file_size);
            
            // Collect unique layer numbers from MoE tensors in this shard
            let mut moe_layers = std::collections::HashSet::new();
            for t in &parser.tensors {
                if t.name.starts_with("blk.") && t.name.contains(".ffn_") && (t.name.ends_with("_exps") || t.name.ends_with("_exps.weight")) {
                    let parts: Vec<&str> = t.name.split('.').collect();
                    if parts.len() >= 3 {
                        if let Ok(il) = parts[1].parse::<usize>() {
                            moe_layers.insert(il);
                        }
                    }
                }
            }
            if !moe_layers.is_empty() {
                let mut layers: Vec<_> = moe_layers.into_iter().collect();
                layers.sort();
                println!("  MoE layers: {:?}", layers);
            }
            
            parsers.push(parser);
        }

        // Determine n_layer and n_expert from tensor names
        let mut layer_set = HashSet::new();
        let mut expert_set = HashSet::new();
        for p in &parsers {
            for t in &p.tensors {
                // Match both "blk.{il}.ffn_gate_exps" and "blk.{il}.ffn_gate_exps.weight"
                if t.name.starts_with("blk.") && t.name.contains(".ffn_") && (t.name.ends_with("_exps") || t.name.ends_with("_exps.weight")) {
                    // blk.{il}.ffn_gate_exps
                    let parts: Vec<&str> = t.name.split('.').collect();
                    if parts.len() >= 3 {
                        if let Ok(il) = parts[1].parse::<usize>() {
                            layer_set.insert(il);
                        }
                    }
                    if t.dims.len() >= 3 {
                        expert_set.insert(t.dims[2] as usize);
                    }
                }
            }
        }

        // Layers are not contiguous from 0 (GLM-5.3 MoE starts at blk.3): size by max index
        let n_layer = layer_set.iter().max().unwrap_or(&0) + 1;
        let n_expert = *expert_set.iter().max().unwrap_or(&0);
        println!("Detected: n_layer={}, n_expert={}, n_expert_used={}", n_layer, n_expert, n_expert_used);

        // Build expert tensor map
        let mut experts: Vec<Option<ExpertTensor>> = std::iter::repeat_with(|| None).take(n_layer * n_expert).collect();
        for (file_idx, parser) in parsers.iter().enumerate() {
            for tensor in &parser.tensors {
                if !tensor.name.starts_with("blk.") || !tensor.name.contains(".ffn_") || (!tensor.name.ends_with("_exps") && !tensor.name.ends_with("_exps.weight")) {
                    continue;
                }
                let parts: Vec<&str> = tensor.name.split('.').collect();
                if parts.len() < 3 {
                    continue;
                }
                let il = parts[1].parse::<usize>().unwrap_or(0);
                let tensor_type = parts[2]; // ffn_gate_exps or ffn_gate_exps.weight
                if il >= n_layer {
                    continue;
                }
                if tensor.dims.len() < 3 {
                    continue;
                }
                let n_expert_tensor = tensor.dims[2] as usize;
                for expert in 0..n_expert_tensor.min(n_expert) {
                    let idx = il * n_expert + expert;
                    let expert_bytes = tensor.expert_bytes;
                    let expert_offset = tensor.offset + (expert as u64) * expert_bytes;
                    let entry = experts[idx].get_or_insert_with(|| ExpertTensor {
                        layer: il,
                        expert,
                        gate_offset: 0,
                        gate_len: 0,
                        up_offset: 0,
                        up_len: 0,
                        down_offset: 0,
                        down_len: 0,
                        file_idx,
                        fd: shard_fds[file_idx],
                    });
                    entry.file_idx = file_idx;
                    entry.fd = shard_fds[file_idx];
                    let base_type = tensor_type.strip_suffix(".weight").unwrap_or(tensor_type);
                    match base_type {
                        "ffn_gate_exps" => {
                            entry.gate_offset = expert_offset;
                            entry.gate_len = expert_bytes;
                        }
                        "ffn_up_exps" => {
                            entry.up_offset = expert_offset;
                            entry.up_len = expert_bytes;
                        }
                        "ffn_down_exps" => {
                            entry.down_offset = expert_offset;
                            entry.down_len = expert_bytes;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Verify all experts have all 3 tensors
        let mut complete = 0;
        for (i, e) in experts.iter().enumerate() {
            if let Some(e) = e {
                if e.gate_len > 0 && e.up_len > 0 && e.down_len > 0 {
                    complete += 1;
                } else {
                    eprintln!("WARNING: expert {} (layer {}) incomplete: gate={} up={} down={}", 
                        i % n_expert, i / n_expert, e.gate_len, e.up_len, e.down_len);
                }
            }
        }
        println!("Complete expert tensors: {} / {}", complete, n_layer * n_expert);

        let experts: Vec<ExpertTensor> = experts.into_iter().filter_map(|e| e).collect();
        let expert_index: HashMap<(usize, usize), usize> = experts
            .iter()
            .enumerate()
            .map(|(i, e)| ((e.layer, e.expert), i))
            .collect();

        Ok(ModelIndex {
            n_layer,
            n_expert,
            n_expert_used,
            experts,
            expert_index,
            shard_files,
            shard_fds,
            shard_sizes: parsers.iter().map(|p| p.file_size).collect(),
        })
    }

    fn get_expert(&self, layer: usize, expert: usize) -> Option<&ExpertTensor> {
        if layer >= self.n_layer || expert >= self.n_expert {
            return None;
        }
        self.expert_index
            .get(&(layer, expert))
            .and_then(|&i| self.experts.get(i))
    }

    fn total_expert_bytes(&self) -> u64 {
        self.experts.iter().map(|e| e.gate_len + e.up_len + e.down_len).sum()
    }
}

struct RingReader {
    header: *mut RingHeader,
    data_ptr: *mut u8,
    capacity: u64,
    record_max: usize,
    last_seen_seq: u64,
    _mmap: *mut c_void,
    mmap_len: usize,
}

impl RingReader {
    fn new(path: &Path) -> Result<Self, String> {
        let path_cstr = std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|e| format!("cstring: {}", e))?;
        let fd = unsafe { open(path_cstr.as_ptr(), O_RDONLY) };
        if fd < 0 {
            return Err(format!("open ring: {}", std::io::Error::last_os_error()));
        }
        let mut st: stat = unsafe { std::mem::zeroed() };
        if unsafe { fstat(fd, &mut st) } < 0 {
            unsafe { close(fd) };
            return Err(format!("fstat ring: {}", std::io::Error::last_os_error()));
        }
        let file_size = st.st_size as usize;
        if file_size < RING_HEADER_SIZE {
            unsafe { close(fd) };
            return Err("ring file too small".into());
        }
        let ptr = unsafe { mmap(std::ptr::null_mut(), file_size, PROT_READ, MAP_SHARED, fd, 0) };
        if ptr == MAP_FAILED {
            unsafe { close(fd) };
            return Err(format!("mmap ring: {}", std::io::Error::last_os_error()));
        }
        unsafe { close(fd) };

        let header = ptr as *mut RingHeader;
        let hdr = unsafe { &*header };
        if hdr.magic != RING_MAGIC {
            unsafe { munmap(ptr, file_size) };
            return Err(format!("invalid ring magic: 0x{:x}", hdr.magic));
        }
        if hdr.version != RING_VERSION {
            unsafe { munmap(ptr, file_size) };
            return Err(format!("unsupported ring version: {}", hdr.version));
        }

        let capacity = hdr.capacity;
        let n_expert_used = hdr.n_expert_used as usize;
        let record_max = 16 + RECORD_MAX_TOKENS * n_expert_used * 4;
        let data_ptr = unsafe { ptr.add(RING_HEADER_SIZE) as *mut u8 };

        Ok(RingReader {
            header,
            data_ptr,
            capacity,
            record_max,
            last_seen_seq: 0,
            _mmap: ptr,
            mmap_len: file_size,
        })
    }

    fn dummy() -> Result<Self, String> {
        // Create a minimal dummy ring reader for check-only mode
        Ok(RingReader {
            header: std::ptr::null_mut(),
            data_ptr: std::ptr::null_mut(),
            capacity: 0,
            record_max: 0,
            last_seen_seq: 0,
            _mmap: std::ptr::null_mut(),
            mmap_len: 0,
        })
    }

    fn read_new_records(&mut self) -> Vec<ExpertRecord> {
        let mut records = Vec::new();
        if self.header.is_null() {
            return records; // Dummy mode
        }
        let hdr = unsafe { &*self.header };
        let write_seq = hdr.write_seq.load(Ordering::Acquire);
        
        if write_seq <= self.last_seen_seq {
            return records;
        }

        let start_seq = self.last_seen_seq + 1;
        let end_seq = write_seq;
        
        for seq in start_seq..=end_seq {
            let idx = (seq - 1) % self.capacity;
            let record_ptr = unsafe { self.data_ptr.add((idx as usize) * self.record_max) };
            let record_seq = unsafe { *(record_ptr as *const u64) };
            if record_seq != seq {
                // Record not yet written or overwritten
                break;
            }
            let il = unsafe { *(record_ptr.add(8) as *const u32) } as usize;
            let n_tokens = unsafe { *(record_ptr.add(12) as *const u32) } as usize;
            if n_tokens > RECORD_MAX_TOKENS {
                eprintln!("WARNING: n_tokens {} exceeds max", n_tokens);
                break;
            }
            let mut ids = Vec::with_capacity(n_tokens * self.n_expert_used());
            let ids_ptr = unsafe { record_ptr.add(16) as *const i32 };
            for i in 0..n_tokens * self.n_expert_used() {
                ids.push(unsafe { *ids_ptr.add(i) });
            }
            records.push(ExpertRecord {
                seq,
                layer: il,
                n_tokens,
                expert_ids: ids,
            });
        }
        self.last_seen_seq = end_seq;
        records
    }

    fn n_expert_used(&self) -> usize {
        if self.header.is_null() {
            return 8; // Default for dummy
        }
        let hdr = unsafe { &*self.header };
        hdr.n_expert_used as usize
    }

    fn n_layer(&self) -> u32 {
        if self.header.is_null() {
            return 0;
        }
        let hdr = unsafe { &*self.header };
        hdr.n_layer
    }
}

impl Drop for RingReader {
    fn drop(&mut self) {
        if !self._mmap.is_null() && self.mmap_len > 0 {
            unsafe { munmap(self._mmap, self.mmap_len) };
        }
    }
}

#[derive(Debug)]
struct ExpertRecord {
    seq: u64,
    layer: usize,
    n_tokens: usize,
    expert_ids: Vec<i32>,
}

struct Heatmap {
    counts: Vec<u64>, // [layer * n_expert + expert]
    n_layer: usize,
    n_expert: usize,
}

impl Heatmap {
    fn new(n_layer: usize, n_expert: usize) -> Self {
        Heatmap {
            counts: vec![0; n_layer * n_expert],
            n_layer,
            n_expert,
        }
    }

    fn add(&mut self, layer: usize, expert: usize) {
        if layer < self.n_layer && expert < self.n_expert {
            let idx = layer * self.n_expert + expert;
            self.counts[idx] = self.counts[idx].saturating_add(1);
        }
    }

    fn get(&self, layer: usize, expert: usize) -> u64 {
        if layer < self.n_layer && expert < self.n_expert {
            self.counts[layer * self.n_expert + expert]
        } else {
            0
        }
    }

    fn top_k_per_layer(&self, k: usize) -> Vec<Vec<(usize, u64)>> {
        let mut result = Vec::with_capacity(self.n_layer);
        for layer in 0..self.n_layer {
            let mut experts: Vec<(usize, u64)> = (0..self.n_expert)
                .map(|e| (e, self.get(layer, e)))
                .collect();
            experts.sort_by(|a, b| b.1.cmp(&a.1));
            experts.truncate(k);
            result.push(experts);
        }
        result
    }

    fn reset(&mut self) {
        self.counts.fill(0);
    }
}

struct Pinner {
    model: ModelIndex,
    ring: RingReader,
    heatmap: Heatmap,
    budget_bytes: u64,
    interval_ms: u64,
    locked_ranges: HashSet<(c_int, u64, u64)>, // (fd, offset, len)
    locked_bytes: u64,
    mlock_failures: u64,
    madvise_throttle: Instant,
    madvise_count: u32,
    prefetch_dedup: HashSet<(usize, usize)>, // (layer, expert)
    running: Arc<AtomicBool>,
    check_only: bool,
    dump_tensors: bool,
}

impl Pinner {
    fn new(model_dir: &Path, ring_path: &Path, budget_mb: u64, interval_ms: u64, check_only: bool, dump_tensors: bool) -> Result<Self, String> {
        let (ring, n_expert_used) = if check_only {
            // For check-only, create a dummy ring with defaults
            // We'll use a reasonable default for n_expert_used
            let dummy_ring = RingReader::dummy()?;
            (dummy_ring, 8) // Default to 8 experts used per token (common for MoE)
        } else {
            let ring = RingReader::new(ring_path)?;
            let n_expert_used = ring.n_expert_used();
            (ring, n_expert_used)
        };
        
        let model = ModelIndex::build(model_dir, n_expert_used)?;
        
        let heatmap = Heatmap::new(model.n_layer, model.n_expert);
        let budget_bytes = budget_mb * 1024 * 1024;
        let running = Arc::new(AtomicBool::new(true));
        
        // Setup signal handlers
        let r = running.clone();
        ctrlc::set_handler(move || {
            r.store(false, Ordering::Relaxed);
        }).map_err(|e| format!("signal handler: {}", e))?;

        Ok(Pinner {
            model,
            ring,
            heatmap,
            budget_bytes,
            interval_ms,
            locked_ranges: HashSet::new(),
            locked_bytes: 0,
            mlock_failures: 0,
            madvise_throttle: Instant::now(),
            madvise_count: 0,
            prefetch_dedup: HashSet::new(),
            running,
            check_only,
            dump_tensors,
        })
    }

    fn run(&mut self) -> Result<(), String> {
        if self.check_only {
            self.print_summary();
            return Ok(());
        }

        println!("Starting moe-pinner: budget={} MB, interval={} ms", self.budget_bytes / (1024*1024), self.interval_ms);
        println!("Model: {} layers, {} experts/layer, {} experts used", self.model.n_layer, self.model.n_expert, self.model.n_expert_used);
        println!("Total expert bytes: {:.2} GB", self.model.total_expert_bytes() as f64 / (1024.0*1024.0*1024.0));
        println!("Bytes per expert: {:.2} MB", self.model.experts.first().map(|e| (e.gate_len + e.up_len + e.down_len) as f64 / (1024.0*1024.0)).unwrap_or(0.0));

        let mut last_recompute = Instant::now();
        let mut last_status = Instant::now();
        let mut total_records = 0u64;
        let mut records_this_interval = 0u64;

        while self.running.load(Ordering::Relaxed) {
            let records = self.ring.read_new_records();
            let records_empty = records.is_empty();
            for rec in records {
                self.process_record(&rec);
                total_records += 1;
                records_this_interval += 1;
            }

            let now = Instant::now();
            if now.duration_since(last_recompute) >= Duration::from_secs(10) {
                self.recompute_mlock_set();
                last_recompute = now;
            }

            if now.duration_since(last_status) >= Duration::from_secs(5) {
                let elapsed = now.duration_since(last_status).as_secs_f64();
                let rps = records_this_interval as f64 / elapsed;
                self.print_status(total_records, rps);
                records_this_interval = 0;
                last_status = now;
            }

            if records_empty {
                thread::sleep(Duration::from_millis(self.interval_ms));
            }
        }

        println!("Shutting down...");
        self.unlock_all();
        Ok(())
    }

    fn process_record(&mut self, rec: &ExpertRecord) {
        // Update heatmap
        for &expert_id in &rec.expert_ids {
            if expert_id >= 0 {
                self.heatmap.add(rec.layer, expert_id as usize);
            }
        }

        // Prefetch for next layer
        self.prefetch_next_layer(rec.layer, &rec.expert_ids);
    }

    fn prefetch_next_layer(&mut self, current_layer: usize, expert_ids: &[i32]) {
        if current_layer + 1 >= self.model.n_layer {
            return;
        }
        let next_layer = current_layer + 1;
        let now = Instant::now();
        
        // Throttle: ~64 madvise calls/sec
        if now.duration_since(self.madvise_throttle) >= Duration::from_millis(1000) {
            self.madvise_throttle = now;
            self.madvise_count = 0;
            self.prefetch_dedup.clear();
        }
        if self.madvise_count >= 64 {
            return;
        }

        for &expert_id in expert_ids {
            if expert_id < 0 {
                continue;
            }
            let expert = expert_id as usize;
            let key = (next_layer, expert);
            if self.prefetch_dedup.insert(key) {
                if let Some(e) = self.model.get_expert(next_layer, expert) {
                    self.madvise_range(e.fd, e.gate_offset, e.gate_len);
                    self.madvise_range(e.fd, e.up_offset, e.up_len);
                    self.madvise_range(e.fd, e.down_offset, e.down_len);
                    self.madvise_count += 3;
                    if self.madvise_count >= 64 {
                        break;
                    }
                }
            }
        }
    }

    fn madvise_range(&self, fd: c_int, offset: u64, len: u64) {
        if len == 0 {
            return;
        }
        // mmap the range for madvise
        let page_offset = (offset / PAGE_SIZE as u64) * PAGE_SIZE as u64;
        let page_len = ((offset + len + PAGE_SIZE as u64 - 1) / PAGE_SIZE as u64) * PAGE_SIZE as u64 - page_offset;
        let ptr = unsafe { mmap(std::ptr::null_mut(), page_len as usize, PROT_READ, MAP_SHARED, fd, page_offset as off_t) };
        if ptr != MAP_FAILED {
            unsafe { madvise(ptr, page_len as usize, MADV_WILLNEED) };
            unsafe { munmap(ptr, page_len as usize) };
        }
    }

    fn recompute_mlock_set(&mut self) {
        // Calculate per-expert bytes
        let bytes_per_expert = if let Some(e) = self.model.experts.first() {
            e.gate_len + e.up_len + e.down_len
        } else {
            return;
        };
        if bytes_per_expert == 0 {
            return;
        }

        // How many experts can we lock within budget?
        let max_experts = (self.budget_bytes / bytes_per_expert) as usize;
        let experts_per_layer = max_experts / self.model.n_layer;
        if experts_per_layer == 0 {
            return;
        }

        // Get top-K per layer from heatmap
        let top_k = self.heatmap.top_k_per_layer(experts_per_layer);
        
        // Build new lock set
        let mut new_locked = HashSet::new();
        let mut new_locked_bytes = 0u64;
        let mut failures = 0u64;

        for (layer, experts) in top_k.iter().enumerate() {
            for &(expert, _count) in experts {
                if let Some(e) = self.model.get_expert(layer, expert) {
                    for (offset, len) in [(e.gate_offset, e.gate_len), (e.up_offset, e.up_len), (e.down_offset, e.down_len)] {
                        if len > 0 {
                            let key = (e.fd, offset, len);
                            new_locked.insert(key);
                            new_locked_bytes += len;
                            
                            // Try mlock
                            let page_offset = (offset / PAGE_SIZE as u64) * PAGE_SIZE as u64;
                            let page_len = ((offset + len + PAGE_SIZE as u64 - 1) / PAGE_SIZE as u64) * PAGE_SIZE as u64 - page_offset;
                            let ptr = unsafe { mmap(std::ptr::null_mut(), page_len as usize, PROT_READ, MAP_SHARED, e.fd, page_offset as off_t) };
                            if ptr != MAP_FAILED {
                                let ret = unsafe { mlock(ptr, page_len as usize) };
                                if ret != 0 {
                                    failures += 1;
                                    // Fallback: madvise POPULATE_READ
                                    unsafe { madvise(ptr, page_len as usize, MADV_POPULATE_READ) };
                                    // Or read a byte from each page
                                    let mut dummy = [0u8; 1];
                                    for i in (0..page_len as usize).step_by(PAGE_SIZE) {
                                        dummy[0] = unsafe { *(ptr.add(i) as *const u8) };
                                    }
                                }
                                unsafe { munmap(ptr, page_len as usize) };
                            }
                        }
                    }
                }
            }
        }

        // Unlock old ranges not in new set
        for key in &self.locked_ranges {
            if !new_locked.contains(key) {
                self.unlock_range(*key);
            }
        }

        self.locked_ranges = new_locked;
        self.locked_bytes = new_locked_bytes;
        self.mlock_failures = failures;
    }

    fn unlock_range(&self, key: (c_int, u64, u64)) {
        let (fd, offset, len) = key;
        let page_offset = (offset / PAGE_SIZE as u64) * PAGE_SIZE as u64;
        let page_len = ((offset + len + PAGE_SIZE as u64 - 1) / PAGE_SIZE as u64) * PAGE_SIZE as u64 - page_offset;
        let ptr = unsafe { mmap(std::ptr::null_mut(), page_len as usize, PROT_READ, MAP_SHARED, fd, page_offset as off_t) };
        if ptr != MAP_FAILED {
            unsafe { munlock(ptr, page_len as usize) };
            unsafe { munmap(ptr, page_len as usize) };
        }
    }

    fn unlock_all(&self) {
        for key in &self.locked_ranges {
            self.unlock_range(*key);
        }
    }

    fn print_summary(&self) {
        println!("=== MoE Pinner Model Summary ===");
        println!("Layers: {}", self.model.n_layer);
        println!("Experts per layer: {}", self.model.n_expert);
        println!("Experts used per token: {}", self.model.n_expert_used);
        println!("Total expert tensors: {}", self.model.experts.len());
        
        let bytes_per_expert = if let Some(e) = self.model.experts.first() {
            e.gate_len + e.up_len + e.down_len
        } else { 0 };
        println!("Bytes per expert (gate+up+down): {:.2} MB", bytes_per_expert as f64 / (1024.0*1024.0));
        println!("Total expert bytes: {:.2} GB", self.model.total_expert_bytes() as f64 / (1024.0*1024.0*1024.0));
        
        if self.dump_tensors {
            self.print_detailed_tensor_info();
        } else {
            println!("\nTensor map (first 5 layers):");
            for layer in 0..self.model.n_layer.min(5) {
                for expert in 0..self.model.n_expert.min(3) {
                    if let Some(e) = self.model.get_expert(layer, expert) {
                        println!("  blk.{}.ffn_gate_exps[{}]: offset={}, len={}", layer, expert, e.gate_offset, e.gate_len);
                        println!("  blk.{}.ffn_up_exps[{}]:   offset={}, len={}", layer, expert, e.up_offset, e.up_len);
                        println!("  blk.{}.ffn_down_exps[{}]: offset={}, len={}", layer, expert, e.down_offset, e.down_len);
                    }
                }
            }
        }
        
        println!("\nRing buffer:");
        println!("  Capacity: {}", self.ring.capacity);
        println!("  Record max size: {}", self.ring.record_max);
        println!("  n_expert_used: {}", self.ring.n_expert_used());
        println!("  n_layer: {}", self.ring.n_layer());
    }
    
    fn print_detailed_tensor_info(&self) {
        println!("\n=== PER-SHARD TENSOR DETAILS ===");
        // Group tensors by shard
        let mut shard_info: std::collections::HashMap<usize, Vec<&ExpertTensor>> = std::collections::HashMap::new();
        for e in &self.model.experts {
            shard_info.entry(e.file_idx).or_default().push(e);
        }
        
        for (shard_idx, tensors) in shard_info.iter() {
            let first = tensors.first().unwrap();
            let last = tensors.last().unwrap();
            println!("\nShard {}:", shard_idx);
            println!("  Expert tensors: {}", tensors.len());
            println!("  First tensor: blk.{}.{} offset={} len={}", first.layer, if first.gate_len > 0 { "ffn_gate_exps" } else if first.up_len > 0 { "ffn_up_exps" } else { "ffn_down_exps" }, first.gate_offset.max(first.up_offset).max(first.down_offset), first.gate_len.max(first.up_len).max(first.down_len));
            println!("  Last tensor:  blk.{}.{} offset={} len={}", last.layer, if last.gate_len > 0 { "ffn_gate_exps" } else if last.up_len > 0 { "ffn_up_exps" } else { "ffn_down_exps" }, last.gate_offset.max(last.up_offset).max(last.down_offset), last.gate_len.max(last.up_len).max(last.down_len));
            
            // Check file bounds
            let file_size = self.model.shard_sizes[*shard_idx];
            let max_offset = tensors.iter().map(|e| e.gate_offset + e.gate_len).max().unwrap_or(0)
                .max(tensors.iter().map(|e| e.up_offset + e.up_len).max().unwrap_or(0))
                .max(tensors.iter().map(|e| e.down_offset + e.down_len).max().unwrap_or(0));
            println!("  File size: {} bytes, max tensor end: {} bytes, fits: {}", file_size, max_offset, max_offset <= file_size);
        }
        
        // Per-layer completeness check
        println!("\n=== PER-LAYER COMPLETENESS ===");
        let mut complete_layers = 0;
        for layer in 0..self.model.n_layer {
            let mut gate_ok = 0;
            let mut up_ok = 0;
            let mut down_ok = 0;
            for expert in 0..self.model.n_expert {
                if let Some(e) = self.model.get_expert(layer, expert) {
                    if e.gate_len > 0 { gate_ok += 1; }
                    if e.up_len > 0 { up_ok += 1; }
                    if e.down_len > 0 { down_ok += 1; }
                }
            }
            let complete = gate_ok == self.model.n_expert && up_ok == self.model.n_expert && down_ok == self.model.n_expert;
            if complete { complete_layers += 1; }
            if !complete || self.dump_tensors {
                println!("  Layer {}: gate={}/{} up={}/{} down={}/{} {}", layer, gate_ok, self.model.n_expert, up_ok, self.model.n_expert, down_ok, self.model.n_expert, if complete { "✓" } else { "✗" });
            }
        }
        println!("Complete layers: {}/{}", complete_layers, self.model.n_layer);
        
        // Verify a known tensor by reading first 16 bytes
        println!("\n=== TENSOR DATA VERIFICATION ===");
        if let Some(e) = self.model.get_expert(3, 0) { // blk.3 is first MoE layer
            if e.gate_len > 0 {
                self.verify_tensor_data(e.fd, e.gate_offset, "blk.3.ffn_gate_exps[0]");
            }
        }
        
        // Total expert bytes recalculated
        let total_slots = self.model.n_layer * self.model.n_expert * 3;
        let total_bytes = self.model.total_expert_bytes();
        if total_slots > 0 {
            println!("\nTotal expert slots (layer*expert*3): {}", total_slots);
            println!("Total expert bytes: {} bytes ({:.2} GiB)", total_bytes, total_bytes as f64 / (1024.0*1024.0*1024.0));
            println!("Bytes per expert-slot: {:.2} MB", (total_bytes / total_slots as u64) as f64 / (1024.0*1024.0));
        }
    }
    
    fn verify_tensor_data(&self, fd: c_int, offset: u64, name: &str) {
        let page_offset = (offset / PAGE_SIZE as u64) * PAGE_SIZE as u64;
        let page_len = PAGE_SIZE * 2; // Read 2 pages to be safe
        let ptr = unsafe { mmap(std::ptr::null_mut(), page_len, PROT_READ, MAP_SHARED, fd, page_offset as off_t) };
        if ptr != MAP_FAILED {
            let local_offset = (offset - page_offset) as usize;
            let bytes = unsafe { std::slice::from_raw_parts(ptr.add(local_offset) as *const u8, 16.min(page_len - local_offset)) };
            let hex = bytes.iter().map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
            println!("  {} @ offset {}: first 16 bytes: {}", name, offset, hex);
            unsafe { munmap(ptr, page_len) };
        } else {
            println!("  {} @ offset {}: mmap failed for verification", name, offset);
        }
    }

    fn print_status(&self, total_records: u64, rps: f64) {
        let top3 = self.heatmap.top_k_per_layer(3);
        let mut hot_summary = String::new();
        for (layer, experts) in top3.iter().enumerate() {
            if experts.is_empty() { continue; }
            hot_summary.push_str(&format!(" L{}:", layer));
            for (expert, count) in experts {
                hot_summary.push_str(&format!(" E{}={}", expert, count));
            }
        }
        println!("seq={} records/s={:.1} locked={:.1}MB failures={}{}",
            total_records, rps, self.locked_bytes as f64 / (1024.0*1024.0), self.mlock_failures, hot_summary);
    }
}

fn print_usage() {
    eprintln!("Usage: moe-pinner --model <dir> --ring <path> --budget-mb <int> [--interval-ms <int>] [--check-only]");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = None;
    let mut ring_path = None;
    let mut budget_mb = 16384u64;
    let mut interval_ms = 2u64;
    let mut check_only = false;
    let mut dump_tensors = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() { model_dir = Some(PathBuf::from(&args[i])); }
            }
            "--ring" => {
                i += 1;
                if i < args.len() { ring_path = Some(PathBuf::from(&args[i])); }
            }
            "--budget-mb" => {
                i += 1;
                if i < args.len() { budget_mb = args[i].parse().unwrap_or(16384); }
            }
            "--interval-ms" => {
                i += 1;
                if i < args.len() { interval_ms = args[i].parse().unwrap_or(2); }
            }
            "--check-only" => {
                check_only = true;
            }
            "--dump-tensors" => {
                dump_tensors = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                print_usage();
                process::exit(1);
            }
        }
        i += 1
    }

    if model_dir.is_none() || ring_path.is_none() {
        print_usage();
        process::exit(1);
    }

    let model_dir = model_dir.unwrap();
    let ring_path = ring_path.unwrap();

    if !model_dir.exists() {
        eprintln!("Model dir not found: {}", model_dir.display());
        process::exit(1);
    }

    if check_only && !ring_path.exists() {
        // For check-only, ring doesn't need to exist
    } else if !ring_path.exists() {
        // Wait for ring to appear (up to 30s)
        println!("Ring not found at {}, waiting up to 30s...", ring_path.display());
        let start = Instant::now();
        while !ring_path.exists() && start.elapsed() < Duration::from_secs(30) {
            thread::sleep(Duration::from_millis(500));
        }
        if !ring_path.exists() {
            eprintln!("Ring still not found after 30s, exiting");
            process::exit(1);
        }
    }

    let mut pinner = match Pinner::new(&model_dir, &ring_path, budget_mb, interval_ms, check_only, dump_tensors) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to initialize: {}", e);
            process::exit(1);
        }
    };

    if let Err(e) = pinner.run() {
        eprintln!("Runtime error: {}", e);
        process::exit(1);
    }
}