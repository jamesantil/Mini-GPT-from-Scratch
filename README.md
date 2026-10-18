# Mini-GPT: Generative AI from Scratch

A character-level language model built from the ground up in **PyTorch**, as part of the WiDS (Winter in Data Science) programme at IIT Bombay.

The project starts with a simple **Bigram** baseline and progressively builds it into a **GPT-style decoder-only Transformer** with masked multi-head self-attention, then compares the two.

## Architecture

| Component | Detail |
|---|---|
| Model | Decoder-only Transformer |
| Layers | 6 Transformer blocks |
| Attention | 6 heads, masked (causal) self-attention |
| Embedding size | 192 |
| Context length | 128 characters |
| Optimizer | AdamW, learning rate 3e-4 |
| Training | 4,000 iterations, batch size 64 |

Each block combines multi-head attention and a feed-forward network, with learned positional embeddings so the model knows character order.

## What's in this repo

| File | Purpose |
|---|---|
| `gpt_wids_chatgpt (1).py` | Full GPT implementation and training loop |
| `input.txt` | Training text corpus |
| `WiDS_Report_24B0303_2026.pdf` | Full report: method, debugging, results, Bigram vs Transformer comparison |

The earlier Bigram baseline lives in [WiDS_Code](https://github.com/jamesantil/WiDS_Code).

## Run it

```bash
pip install torch
python "gpt_wids_chatgpt (1).py"
```

It trains on `input.txt`, prints the validation loss, and generates sample text.

## Results

The Transformer achieves much lower cross-entropy loss than the Bigram baseline and generates text with far better syntax and long-range context, since it attends to up to 128 previous characters instead of only one. See the report for the full comparison.
