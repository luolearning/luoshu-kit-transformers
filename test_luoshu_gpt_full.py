from transformers import GPT2Tokenizer, GPT2LMHeadModel
from luoshu_principle import LuoshuPrinciple
import torch
import json


# =========================
# 1. Setup
# =========================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

lp = LuoshuPrinciple()

# 用来保存所有实验结果
results = []


# =========================
# 2. Helpers
# =========================
def get_topk_tokens(logits, k=5):
    last = logits[0, -1]
    values, indices = torch.topk(last, k)
    tokens = [tokenizer.decode([idx.item()]) for idx in indices]
    scores = [float(v.item()) for v in values]
    return list(zip(tokens, scores))


def path_to_dim(path):
    """
    path -> (i, j) -> flat dim in 27x27 grid
    Uses first 729 dims of GPT-2 hidden state.
    """
    i, j = lp.decode(path)
    dim = i * 27 + j
    return dim


def patch_dims_from_path(path, radius=0):
    """
    Build a set of dims around the Luoshu coordinate.
    radius=0 -> single point
    radius=1 -> local patch (up to 3x3 in Luoshu grid)
    """
    i, j = lp.decode(path)
    dims = []

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ii = i + di
            jj = j + dj
            if 0 <= ii < 27 and 0 <= jj < 27:
                dims.append(ii * 27 + jj)

    return sorted(set(dims))


def run_base(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return out.logits


def run_luoshu_intervention(prompt, path, layer_idx=5, radius=0, strength=-10.0):
    """
    Apply Luoshu-based intervention at a chosen transformer block.

    path: tuple like (4, 9, 2)
    layer_idx: which GPT-2 block to hook
    radius: 0 for single address, 1 for local patch
    strength: additive perturbation
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    dims = patch_dims_from_path(path, radius=radius)

    def hook(module, inputs, output):
        # GPT-2 block output may be tuple(hidden, present, ...)
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        hidden = hidden.clone()

        # Only modify first 729 dims that correspond to 27x27 Luoshu frame
        for dim in dims:
            if dim < hidden.shape[-1]:
                hidden[:, -1, dim] += strength

        if rest is None:
            return hidden
        return (hidden,) + rest

    handle = model.transformer.h[layer_idx].register_forward_hook(hook)

    try:
        with torch.no_grad():
            out = model(**inputs)
    finally:
        handle.remove()

    return out.logits


def compare(prompt, path, layer_idx=5, radius=0, strength=-10.0, k=5):
    base_logits = run_base(prompt)
    int_logits = run_luoshu_intervention(
        prompt=prompt,
        path=path,
        layer_idx=layer_idx,
        radius=radius,
        strength=strength,
    )

    decoded_coord = lp.decode(path)
    dims = patch_dims_from_path(path, radius=radius)
    before_topk = get_topk_tokens(base_logits, k=k)
    after_topk = get_topk_tokens(int_logits, k=k)

    print("=" * 60)
    print("Prompt:", prompt)
    print("Path:", path)
    print("Layer:", layer_idx)
    print("Radius:", radius)
    print("Strength:", strength)
    print("Decoded coord:", decoded_coord)
    print("Dims:", dims)
    print("-" * 60)
    print("Before top-k:")
    for tok, score in before_topk:
        print(f"  {repr(tok):>12}  {score:.4f}")
    print("-" * 60)
    print("After top-k:")
    for tok, score in after_topk:
        print(f"  {repr(tok):>12}  {score:.4f}")
    print("=" * 60)
    print()

    # 保存结果
    results.append({
        "prompt": prompt,
        "path": list(path),
        "layer": layer_idx,
        "radius": radius,
        "strength": strength,
        "decoded_coord": list(decoded_coord),
        "dims": dims,
        "before": [
            {"token": tok, "score": score} for tok, score in before_topk
        ],
        "after": [
            {"token": tok, "score": score} for tok, score in after_topk
        ],
    })


# =========================
# 3. Main experiments
# =========================
if __name__ == "__main__":
    prompts = [
        "The capital of France is",
        "The capital of France is Paris. The capital of Germany is",
        "The opposite of hot is",
    ]

    paths = [
        (4, 9, 2),
        (3, 5, 7),
        (8, 1, 6),
        (1, 1, 1),
    ]

    for prompt in prompts:
        for path in paths:
            compare(
                prompt=prompt,
                path=path,
                layer_idx=5,
                radius=1,
                strength=-10.0,
                k=5,
            )

    # 保存成 JSON 文件
    with open("luoshu_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved results to luoshu_results.json")
