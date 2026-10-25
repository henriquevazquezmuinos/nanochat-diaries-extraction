"""
Attention flow computed from a dense replay of the full token sequence.

The sparse `conversation.attn.bin` dump keeps only the top-K keys per (layer, head, query) row.
That is fine for inspecting a single row, but composing those sparse rows across layers loses
mass multiplicatively (a path dies as soon as it is outside the top-K at one layer). To get a
faithful output->input flow we therefore replay the sequence with dense attention capture
(nanochat/gpt.py `attn_capture`) and do the composition in memory. Only the resulting T x T
matrix is written, so the on-disk artifact stays small:

    flow.npz
      flow            : float16 [T, T]  flow[q, j] = share of position q's routed representation
                                         that comes from position j after all layers (lower triangular)
      row_mass        : float32 [T]     sum of each flow row (sanity check, ~1.0 for dense replay)
      tokens          : int32   [T]     token ids that were replayed
      text            : unicode [T]     decoded tokens (empty if no tokenizer was given)
      assistant_ranges: int32   [R, 2]  inclusive (start, end) ranges of assistant tokens
      n_layer         : int32
      n_head          : int32
      include_identity: bool            whether the residual/identity term was added per layer
      head_reduction  : unicode         "mean" (heads averaged) or "none" (per-head flow kept)
      format_version  : int32
      flow_per_layer  : float16 [L, T, T]   (optional, only with per_layer=True)
      per_layer_row_mass: float32 [L, T]    (optional, only with per_layer=True)

When head_reduction="none", flow is [H, T, T] and row_mass / per_layer_row_mass keep the head axis.

Caveat: attention rollout is a heuristic. nanochat also has smear, x0 residuals, a mid-layer
backout subtraction, value embeddings and MLPs, none of which are represented by attention alone.
Use this as an attention-flow view, not as an exact attribution of the logits.
"""
import os

import numpy as np
import torch

from nanochat.attn_dump import replay_attention

FLOW_FILE = "flow.npz"
FORMAT_VERSION = 1


def _eye_like(A, T):
    return torch.eye(T, dtype=A.dtype, device=A.device)


@torch.inference_mode()
def rollout_flow(capture, include_identity=True, head_reduction="mean", per_layer=False):
    """
    Compose dense attention across layers into a token-to-token flow matrix.

    capture: list (one per layer) of (1, H, T, T) fp32 attention, as returned by replay_attention.
    include_identity: add an identity term per layer, i.e. A <- normalize(A + I), to approximate
        the residual stream (standard attention rollout).
    head_reduction: "mean" -> average heads, returning (T, T); "none"/None -> keep heads, (H, T, T).
    per_layer: also return the flow after each layer as a list of L matrices.

    Returns R (or (R, layers)) in fp32. Each row sums to ~1; R[q, j] is the share of position q's
    representation flowing from position j.
    """
    if head_reduction is None:
        head_reduction = "none"
    if head_reduction not in ("mean", "none"):
        raise ValueError(f"head_reduction must be 'mean' or 'none', got {head_reduction!r}")

    first = capture[0]
    T, H = first.shape[-1], first.shape[1]
    if head_reduction == "mean":
        R = torch.eye(T, dtype=torch.float32, device=first.device)
    else:
        R = torch.eye(T, dtype=torch.float32, device=first.device).expand(H, T, T).clone()

    layers = []
    for att in capture:
        A = att[0].float()
        if head_reduction == "mean":
            A = A.mean(dim=0)
        if include_identity:
            A = A + _eye_like(A, T)
            A = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
        R = A @ R
        if per_layer:
            layers.append(R.clone())

    return (R, layers) if per_layer else R


def _flow_arrays(flow, row_mass, tokens, assistant_ranges, tokenizer, include_identity,
                 head_reduction, n_layer, n_head, dtype, per_layer_flow=None, per_layer_mass=None):
    ranges = np.asarray(list(assistant_ranges), dtype=np.int32).reshape(-1, 2)
    text = np.asarray([tokenizer.decode([t]) for t in tokens]) if tokenizer is not None else np.asarray([], dtype="<U1")
    arrays = {
        "format_version": np.int32(FORMAT_VERSION),
        "flow": flow.astype(dtype),
        "row_mass": row_mass.astype(np.float32),
        "tokens": np.asarray(tokens, dtype=np.int32),
        "text": text,
        "assistant_ranges": ranges,
        "n_layer": np.int32(n_layer),
        "n_head": np.int32(n_head),
        "include_identity": np.bool_(include_identity),
        "head_reduction": np.asarray(head_reduction),
        "dtype": np.asarray(np.dtype(dtype).name),
    }
    if per_layer_flow is not None:
        arrays["flow_per_layer"] = per_layer_flow.astype(dtype)
        arrays["per_layer_row_mass"] = per_layer_mass.astype(np.float32)
    return arrays


def save_flow_from_capture(capture, tokens, assistant_ranges, out_dir, tokenizer=None,
                           include_identity=True, head_reduction="mean", per_layer=False,
                           dtype=np.float16):
    """
    Compute the dense-replay flow and write flow.npz into out_dir.
    assistant_ranges: list of inclusive (start, end) index ranges of assistant tokens inside tokens.
    Returns the path written.
    """
    os.makedirs(out_dir, exist_ok=True)
    if per_layer:
        flow, layers = rollout_flow(capture, include_identity, head_reduction, per_layer=True)
    else:
        flow = rollout_flow(capture, include_identity, head_reduction, per_layer=False)
        layers = None
    flow_np = flow.cpu().numpy()
    row_mass = flow.sum(-1).cpu().numpy()
    per_layer_flow = None
    per_layer_mass = None
    if layers is not None:
        per_layer_flow = np.stack([layer.cpu().numpy() for layer in layers])
        per_layer_mass = np.stack([layer.sum(-1).cpu().numpy() for layer in layers])
    arrays = _flow_arrays(
        flow_np, row_mass, tokens, assistant_ranges, tokenizer, include_identity,
        head_reduction, len(capture), capture[0].shape[1], dtype, per_layer_flow, per_layer_mass,
    )
    path = os.path.join(out_dir, FLOW_FILE)
    np.savez(path, **arrays)
    return path


def save_flow(model, tokens, assistant_ranges, out_dir, tokenizer=None, include_identity=True,
              head_reduction="mean", per_layer=False, dtype=np.float16):
    """Replay tokens with dense attention capture, compute the flow, and write flow.npz."""
    capture, _ = replay_attention(model, tokens)
    return save_flow_from_capture(
        capture, tokens, assistant_ranges, out_dir, tokenizer=tokenizer,
        include_identity=include_identity, head_reduction=head_reduction,
        per_layer=per_layer, dtype=dtype,
    )


def load_flow(path):
    """Load a flow.npz written by save_flow / save_flow_from_capture into a dict of arrays."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
