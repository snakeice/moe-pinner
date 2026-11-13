# moe-pinner

Sidecar daemon que mantém experts quentes de um MoE residentes na RAM
(heatmap → mlock → prefetch), trabalhando em conjunto com um fork do
llama.cpp que exporta as escolhas do router.

## Uso

```bash
# 1. Engine (llama.cpp, branch moe-router-export)
llama-server -m <shard1> -ngl 99 --override-tensor "(exps|shexp)=CPU" \
  -c 8192 -ub 256 -b 512 -fa on -t 16 \
  --moe-router-export /dev/shm/moe_ring.bin --port 8080

# 2. Daemon
ulimit -l unlimited   # senão o mlock degrada para madvise
target/release/moe-pinner \
  --model <dir-dos-shards-gguf> \
  --ring /dev/shm/moe_ring.bin \
  --budget-mb 16384

# Só validar o índice de experts (não precisa do engine rodando)
moe-pinner --model <dir> --ring /tmp/x.bin --check-only --dump-tensors
```

## Protocolo do ring (compartilhado com o patch C++)

- Header (4096 bytes, LE): magic `0x52454F4D`, version 1, n_layer u32,
  n_expert u32, n_expert_used u32, pad u32, capacity u64, write_seq u64
  (atômico, 1-based).
- Record `i` em `4096 + ((seq-1) mod capacity) * record_max`, com
  `record_max = 16 + 512 * n_expert_used * 4`:
  seq u64, il u32, n_tokens u32, e `n_tokens * n_expert_used` i32 ids
  (token-major). Reader espera `seq == i+1` no slot.

## Estado

- Parser GGUF validado contra GLM-5.3-Flash UD-IQ4_XS: 43/43 camadas
  MoE (blk.3–45) × 288 experts × gate/up/down, 140,47 GiB de experts.
- O patch C++ correspondente vive no fork llama.cpp, branch
  `moe-router-export` (commit 9c75a07, base 5676dfb).
