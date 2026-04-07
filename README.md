# Luoshu Kit for Transformers (PoC)

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

This project was developed by Jianmin Luo through a human–AI collaborative process. ChatGPT contributed to the clarification and programmatic formulation of the recursive Luoshu principle that underlies the system.
