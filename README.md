## I build Qwen2.5-3B from scratch in this repository.

---
## 文件架构
```
DecoderOnlyLLM/
├── qwen/
│   ├── config.py
│   ├── embedding.py
│   ├── norm.py
│   ├── mlp.py
│   ├── rope.py
│   ├── attention.py
│   ├── block.py
│   └── model.py
│
├── tests/
│   ├── test_norm.py
│   ├── test_mlp.py
│   ├── test_rope.py
│   ├── test_attention.py
│   ├── test_block.py
│   └── test_model.py
│
├── load_weights.py
└── generate.py
```
---
## 预计实现顺序
```
Config / Embedding
        │
        ├── RMSNorm
        ├── MLP
        └── Basic Attention
                 │
                RoPE
                 │
           GQA Attention
                 │
             Decoder Block
                 │
              Full Model
                 │
        Checkpoint Compatibility
                 │
           Generation / KV Cache
```
---
## 实现计划及完成进度

| 步骤 | 名称 | 状态 |
| --- | --- | --- |
| 1 | Config 与 Token Embedding | 进行中（Config 已完成） |
| 2 | RMSNorm | 未完成 |
| 3 | SwiGLU MLP | 未完成 |
| 4 | 基础 Causal Attention | 未完成 |
| 5 | RoPE | 未完成 |
| 6 | GQA 与完整 Attention | 未完成 |
| 7 | Decoder Block | 未完成 |
| 8 | 完整 Qwen2.5 Model | 未完成 |
| 9 | 官方权重加载与对比 | 未完成 |
| 10 | 生成与 KV Cache | 未完成 |


