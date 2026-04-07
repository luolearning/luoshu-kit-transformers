# Luoshu Kit V0.2 Transformers (PoC)

## LuoshuKit is a plug-in. Inject it into any model.

## A mechanistic interpretability layer — every value now has a computed address instead of being searched.

LuoshuKit implements a structured addressing layer for neural representations,
where internal values are assigned addresses that can be directly decoded rather than located through search.

---


## Luoshu Addressing Framework

This repository implements a Luoshu-based addressing system for neural representations, including anchor–path encoding and direct decoding of internal locations.

This is a minimal proof-of-concept showing that Luoshu-based direct addressing works on GPT-2.

---

## Key Idea

Instead of searching for circuits (e.g. attribution patching), we:

- Map the residual stream into a 27×27 Luoshu grid  
- Use path-based addressing (e.g. (4,9,2)) to select coordinates  
- Directly intervene on those coordinates  
- Observe changes in output behavior  

---

## Example

We demonstrate:

- Shifting token rankings (e.g. Berlin vs Frankfurt)  
- Modulating semantic strength (e.g. "cold")  

---

## Run

pip install -r requirements.txt  
python3 test_luoshu_gpt_full.py  

---

## Notes

This project was developed by Jianmin Luo.  
AI tools were used to assist in clarifying and formalizing the recursive Luoshu principle and its implementation.
