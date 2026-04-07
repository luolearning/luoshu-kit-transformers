# Luoshu Kit for Transformers (PoC)

## Luoshu Addressing Framework

This repository implements a Luoshu-based addressing system for neural representations,
including anchor–path encoding and direct decoding of internal locations.

This is a minimal proof-of-concept showing that Luoshu-based direct addressing works on GPT-2.

## Key Idea

Instead of searching for circuits or using attribution patching, we:

- Map the residual stream into a 27×27 Luoshu grid
- Use path-based addressing (e.g. (4,9,2)) to select coordinates
- Directly intervene on those coordinates
- Observe changes in output behavior

## Example

We demonstrate:

- Shifting token rankings (e.g. Berlin vs Frankfurt)
- Modulating semantic strength (e.g. "cold")

## Run

```bash
pip install -r requirements.txt
python3 test_luoshu_gpt_full.py
```

This project was developed by Jianmin Luo through a human–AI collaborative process. ChatGPT contributed to the clarification and formalization of the recursive Luoshu principle, and to its translation into an executable addressing framework underlying the system.
