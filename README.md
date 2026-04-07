# Luoshu Kit V0.3 Transformers (PoC)

## LuoshuKit is a plug-in. Inject it into any model.

## A mechanistic interpretability layer — every value now has a computed address instead of being searched.

LuoshuKit implements a structured addressing layer for neural representations, introducing a coordinate system over internal values that can be directly accessed without search.

---


## Luoshu Addressing Framework

This repository represents the full Luoshu addressing system on Transformer architectures (GPT-2), using anchor–path encoding to resolve internal coordinates.

It illustrates direct addressing over Transformer representations.


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

This will run a simple intervention demo on GPT-2 and print changes in token rankings and semantic strength.
Try modifying the token pairs (e.g. Berlin → Paris) to explore how addressing affects outputs.

---

## Notes

This project was developed by Jianmin Luo.  
AI tools were used to assist in clarifying and formalizing the recursive Luoshu principle and its implementation.
