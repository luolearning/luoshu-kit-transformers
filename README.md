# Luoshu Kit V0.2 Transformers (PoC)

## LuoshuKit is a plug-in. Inject it into any model.

## A mechanistic interpretability layer — every value now has a computed address instead of being searched.

LuoshuKit implements a structured addressing layer for neural representations,
introducing a coordinate system over internal values.

---

## Luoshu Addressing Framework

This repository is a Transformer-based implementation of LuoshuKit, instantiating the Luoshu addressing system with anchor–path encoding for resolving internal coordinates.

It illustrates how the Luoshu structure can be applied to Transformer architectures (GPT-2).

---

## Key Idea

Instead of search-based circuit tracing (e.g. attribution patching), LuoshuKit treats internal representations as an addressable space:

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

```bash
pip install -r requirements.txt
python3 test_luoshu_gpt_full.py
```
---

## Notes

This project was developed by Jianmin Luo.  
AI tools were used to assist in clarifying and formalizing the recursive Luoshu principle and its implementation.
