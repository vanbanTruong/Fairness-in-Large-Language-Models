import numpy as np
import torch


def install_head_masks(model, masks, N_HEADS, HEAD_DIM, fire_counter):

    handles = []
    for l, block in enumerate(model.transformer.h):
        def post_hook(module, inputs, output, l=l):
            fire_counter[0] += 1
            D = output.shape[-1] // 3
            q, k, v = output[..., :D], output[..., D:2 * D], output[..., 2 * D:]
            B, T, _ = v.shape
            v_h = v.view(B, T, N_HEADS, HEAD_DIM)
            v_h = v_h * masks[l].view(1, 1, N_HEADS, 1)
            v = v_h.reshape(B, T, D)
            return torch.cat([q, k, v], dim=-1)
        handles.append(block.attn.c_attn.register_forward_hook(post_hook))
    return handles


def encode_word(model, tokenizer, DEVICE, word):
    text = f"This is a {word}."
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    out = model(ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1]


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))[0]


def seat_effect_size(model, tokenizer, DEVICE, X, Y, A, B, verbose=True):
    embA = [encode_word(model, tokenizer, DEVICE, a) for a in A]
    embB = [encode_word(model, tokenizer, DEVICE, b) for b in B]

    def s(w):
        ew = encode_word(model, tokenizer, DEVICE, w)
        ma = torch.stack([_cos(ew, ea) for ea in embA]).mean()
        mb = torch.stack([_cos(ew, eb) for eb in embB]).mean()
        return ma - mb

    sX = torch.stack([s(x) for x in X])
    sY = torch.stack([s(y) for y in Y])
    pooled = torch.cat([sX, sY])
    std = pooled.std(unbiased=False).detach().clamp(min=1e-8)
    d = (sX.mean() - sY.mean()) / std
    if verbose:
        print(f"    SEAT d = {d.item():.4f}   std = {std.item():.4f}")
    return d


def compute_gbe_matrix(model, tokenizer, DEVICE, X, Y, A, B, loss_scale=1.0,
                       verbose=True):

    N_LAYERS = model.config.n_layer
    N_HEADS = model.config.n_head
    HEAD_DIM = model.config.n_embd // N_HEADS
    param_dtype = next(model.parameters()).dtype

    orig_requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    fire_counter = [0]
    try:
        with torch.enable_grad():
            masks = torch.ones(N_LAYERS, N_HEADS, device=DEVICE,
                               dtype=param_dtype, requires_grad=True)
            handles = install_head_masks(model, masks, N_HEADS, HEAD_DIM, fire_counter)
            try:
                d = seat_effect_size(model, tokenizer, DEVICE, X, Y, A, B,
                                     verbose=verbose)
                loss = torch.abs(d) * loss_scale
                loss.backward()
            finally:
                for h in handles:
                    h.remove()

        if fire_counter[0] == 0:
            raise RuntimeError(
                "Head-mask hooks never fired. 'model.transformer.h[*].attn.c_attn' "
                "was not on the executed forward path for this model/version."
            )
        if masks.grad is None:
            raise RuntimeError(
                "masks.grad is None after backward(): the mask was not connected to "
                "the loss graph. The Value-masking intervention did not affect the "
                "forward pass (likely a fused-attention path or a no_grad context)."
            )
        if torch.isnan(masks.grad).any() or torch.isinf(masks.grad).any():
            raise RuntimeError(
                "Mask gradient contains NaN/Inf: the forward pass was finite but the "
                "backward blew up. This is weight-dependent (real GPT-2 activations, "
                "not the architecture). To localize the exact op, re-run wrapped in "
                "`with torch.autograd.set_detect_anomaly(True):` -- it will name the "
                "function whose backward produced the NaN. Most likely fixes, in order: "
                "(1) load with attn_implementation='eager'; (2) run the model in float64; "
                "(3) clamp the offending activation once you know which it is."
            )
        gbe = masks.grad.detach().cpu().numpy()
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(orig_requires_grad[n])

    return gbe

def compute_gbe(gbe_matrix: np.ndarray) -> float:
    return float((gbe_matrix > 0).mean())


def compute_gbe_mass(gbe_matrix: np.ndarray) -> float:
    g = np.asarray(gbe_matrix, dtype=float)
    denom = np.abs(g).sum()
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(g[g > 0].sum() / denom)


def random_partition(group1, group2, rng):
    pool = list(group1) + list(group2)
    idx = rng.permutation(len(pool))
    n1 = len(group1)
    return [pool[i] for i in idx[:n1]], [pool[i] for i in idx[n1:]]


def gbe_permutation_null(model, tokenizer, DEVICE, X, Y, A, B,
                         n_perm=50, seed=42, permute="targets",
                         loss_scale=1.0, progress=True):
    rng = np.random.default_rng(seed)
    props, masses = [], []
    for i in range(n_perm):
        if permute == "targets":
            Xp, Yp = random_partition(X, Y, rng); Ap, Bp = list(A), list(B)
        elif permute == "attributes":
            Ap, Bp = random_partition(A, B, rng); Xp, Yp = list(X), list(Y)
        elif permute == "both":
            Xp, Yp = random_partition(X, Y, rng)
            Ap, Bp = random_partition(A, B, rng)
        else:
            raise ValueError(f"unknown permute mode {permute!r}")

        m = compute_gbe_matrix(model, tokenizer, DEVICE, Xp, Yp, Ap, Bp,
                               loss_scale=loss_scale, verbose=False)
        props.append(compute_gbe(m))
        masses.append(compute_gbe_mass(m))
        if progress and (i + 1) % 10 == 0:
            print(f"      null permutation {i+1}/{n_perm}", end="\r")
    if progress:
        print(" " * 40, end="\r")
    return np.array(props, dtype=float), np.array(masses, dtype=float)


def null_summary(observed: float, null_vals: np.ndarray, chance: float = 0.5) -> dict:
    nv = np.asarray(null_vals, dtype=float)
    nv = nv[np.isfinite(nv)]
    if nv.size == 0 or not np.isfinite(observed):
        return {"observed": observed, "null_mean": float("nan"),
                "null_ci_low": float("nan"), "null_ci_high": float("nan"),
                "p_value": float("nan"), "significant": False}
    n_extreme = int(np.sum(np.abs(nv - chance) >= abs(observed - chance)))
    p = (1 + n_extreme) / (1 + nv.size)
    return {
        "observed":     round(float(observed), 4),
        "null_mean":    round(float(nv.mean()), 4),
        "null_ci_low":  round(float(np.percentile(nv, 2.5)), 4),
        "null_ci_high": round(float(np.percentile(nv, 97.5)), 4),
        "p_value":      round(float(p), 4),
        "significant":  bool(p < 0.05),
    }