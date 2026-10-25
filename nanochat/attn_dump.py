"""
Save internal model state (attention + logit probabilities) for a finished conversation.

Instead of instrumenting the decode loop, we *replay* the full conversation through the model once
with attention capture enabled (see `attn_capture` in nanochat/gpt.py). With RoPE and QK-norm the
attention weights for a given token sequence are the same as at decode time (up to low-precision noise).

Outputs (in out_dir):
  conversation.attn.bin  sparse top-K attention, binary format v1 (little-endian, planar):
      header (16 bytes): magic "NCAT", u16 version, T, L, H, K, u16 reserved
      keys    : u16[L*H*T*K]  key index per stored entry, 0xFFFF = padding (rows with < K keys)
      weights : u8 [L*H*T*K]  attention weight * 255, sorted descending within a row
      dropped : u8 [L*H*T]    attention mass NOT stored (outside top-K) * 255
      Row index = (layer * H + head) * T + query.
  logits.json            per predicted assistant token: sampled token + top-N logits and softmax probs
  tokens.json            the conversation token ids (and decoded strings if a tokenizer is given)
"""
import json
import os
import struct
import sys
from array import array

import torch

MAGIC = b"NCAT"
VERSION = 1
HEADER = 16
SENTINEL = 0xFFFF
ATTN_TOPK = 16


def encode_attn_bin(T, L, H, K, keys, weights, dropped):
    """Pack already-sparsified attention into the .attn.bin layout. keys: array('H'); weights, dropped: bytes-like."""
    assert T < SENTINEL, f"T={T} does not fit in u16 key indices"
    assert len(keys) == L * H * T * K and len(weights) == L * H * T * K and len(dropped) == L * H * T
    if sys.byteorder == "big":
        keys = array("H", keys)
        keys.byteswap()
    header = struct.pack("<4sHHHHHH", MAGIC, VERSION, T, L, H, K, 0)
    return header + keys.tobytes() + bytes(weights) + bytes(dropped)


def decode_attn_bin(buf):
    """Inverse of encode_attn_bin (reference decoder, used in tests). Returns (T, L, H, K, keys, weights, dropped)."""
    magic, version, T, L, H, K, _ = struct.unpack("<4sHHHHHH", buf[:HEADER])
    assert magic == MAGIC and version == VERSION, "not a v1 NCAT file"
    n = L * H * T * K
    keys = array("H")
    keys.frombytes(buf[HEADER:HEADER + 2 * n])
    if sys.byteorder == "big":
        keys.byteswap()
    weights = buf[HEADER + 2 * n:HEADER + 3 * n]
    dropped = buf[HEADER + 3 * n:HEADER + 3 * n + L * H * T]
    assert len(buf) == HEADER + 3 * n + L * H * T, "unexpected file size"
    return T, L, H, K, keys, weights, dropped


def sparsify_layer(att, K=ATTN_TOPK):
    """
    att: (H, T, T) softmax weights for one layer (causal => entries with key > query are exactly 0).
    Returns (keys u16 (H*T*K) as torch int32 cpu, weights u8 (H*T*K), dropped u8 (H*T)), flattened in row order
    (head * T + query), matching the per-layer slice of the file layout.
    """
    H, T, _ = att.shape
    k = min(K, T)
    vals, idx = att.topk(k, dim=-1)  # sorted descending
    if k < K:  # pad to K columns
        vals = torch.cat([vals, vals.new_zeros(H, T, K - k)], dim=-1)
        idx = torch.cat([idx, idx.new_zeros(H, T, K - k)], dim=-1)
    # a row at query q has only q+1 valid keys: the rest are zero-weight padding
    valid = torch.arange(K, device=att.device).view(1, 1, K) < (torch.arange(T, device=att.device).view(1, T, 1) + 1)
    idx = torch.where(valid, idx, torch.full_like(idx, SENTINEL))
    wq = (vals * 255 + 0.5).floor().clamp(0, 255).to(torch.int32)
    wq = torch.where(valid, wq, torch.zeros_like(wq))
    dropped = (255 - wq.sum(-1)).clamp(min=0)
    return idx.to(torch.int32).cpu(), wq.to(torch.uint8).cpu(), dropped.to(torch.uint8).cpu()


@torch.inference_mode()
def replay_attention(model, tokens):
    """
    Replay `tokens` (list[int]) through `model` with dense attention capture.
    Returns (capture, logits) where capture is a list (one per layer) of (1, H, T, T) fp32
    attention weights and logits is (1, T, vocab).
    """
    device = model.get_device()
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    capture = []
    logits = model(ids, attn_capture=capture)  # (1, T, vocab)
    assert len(capture) == model.config.n_layer
    return capture, logits


def attention_blob(capture, attn_topk=ATTN_TOPK, free=False):
    """
    Pack dense captured attention (list of (1, H, T, T)) into the .attn.bin v1 layout.
    free=True drops each layer's dense matrix from `capture` as soon as it is sparsified.
    """
    T = capture[0].shape[-1]
    L, H = len(capture), capture[0].shape[1]
    keys, weights, dropped = [], [], []
    for l in range(L):
        k_, w_, d_ = sparsify_layer(capture[l][0], attn_topk)
        if free:
            capture[l] = None  # free the dense (H, T, T) matrix as soon as it is sparsified
        keys.append(k_.reshape(-1))
        weights.append(w_.reshape(-1))
        dropped.append(d_.reshape(-1))
    keys = array("H", torch.cat(keys).to(torch.int32).tolist())
    weights = torch.cat(weights).numpy().tobytes()
    dropped = torch.cat(dropped).numpy().tobytes()
    return encode_attn_bin(T, L, H, attn_topk, keys, weights, dropped)


@torch.inference_mode()
def capture_internals(model, tokens, attn_topk=ATTN_TOPK, logit_positions=()):
    """
    Replay `tokens` (list[int]) through `model` with attention capture.
    Returns (attn_bin_bytes, logits_by_pos) where logits_by_pos maps each position in `logit_positions`
    to its (vocab,) fp32 logits on CPU.
    """
    capture, logits = replay_attention(model, tokens)
    blob = attention_blob(capture, attn_topk, free=True)
    logits_by_pos = {p: logits[0, p].float().cpu() for p in logit_positions}
    return blob, logits_by_pos


def assistant_prediction_positions(tokens, assistant_ranges):
    """Positions q whose logits predicted an assistant token, i.e. tokens[q+1] is in a [start, end] range (inclusive)."""
    pos = []
    for start, end in assistant_ranges:
        pos.extend(q for q in range(start - 1, end) if 0 <= q < len(tokens) - 1)
    return pos


def logits_record(q, tokens, logits, tokenizer=None, top_n=50):
    """JSON-able record of the model's next-token distribution at position q (target = tokens[q+1])."""
    probs = torch.softmax(logits, dim=-1)
    top_p, top_i = probs.topk(min(top_n, probs.numel()))
    dec = (lambda t: tokenizer.decode([t])) if tokenizer is not None else (lambda t: None)
    target = tokens[q + 1]
    return {
        "position": q,
        "target_token": target,
        "target_text": dec(target),
        "target_logit": logits[target].item(),
        "target_prob": probs[target].item(),
        "top": [
            {"token": i, "text": dec(i), "logit": logits[i].item(), "prob": p}
            for i, p in zip(top_i.tolist(), top_p.tolist())
        ],
    }


def save_internals_from_capture(capture, logits, tokens, assistant_ranges, out_dir, tokenizer=None,
                                attn_topk=ATTN_TOPK, probs_top_n=50):
    """
    Write conversation.attn.bin, logits.json, tokens.json into out_dir from an already captured replay.
    assistant_ranges: list of inclusive (start, end) index ranges of assistant tokens inside `tokens`.
    Returns the paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    positions = assistant_prediction_positions(tokens, assistant_ranges)
    blob = attention_blob(capture, attn_topk)
    logits_by_pos = {p: logits[0, p].float().cpu() for p in positions}
    attn_path = os.path.join(out_dir, "conversation.attn.bin")
    with open(attn_path, "wb") as f:
        f.write(blob)
    logits_path = os.path.join(out_dir, "logits.json")
    with open(logits_path, "w") as f:
        json.dump({
            "note": "recomputed by replaying the conversation; top-N entries per predicted assistant token. "
                    "probs are the plain softmax of the logits (temperature 1, no top-k).",
            "top_n": probs_top_n,
            "steps": [logits_record(q, tokens, logits_by_pos[q], tokenizer, probs_top_n) for q in positions],
        }, f)
    tokens_path = os.path.join(out_dir, "tokens.json")
    with open(tokens_path, "w") as f:
        json.dump({
            "tokens": tokens,
            "text": [tokenizer.decode([t]) for t in tokens] if tokenizer is not None else None,
            "assistant_ranges": [list(r) for r in assistant_ranges],
        }, f)
    return [attn_path, logits_path, tokens_path]


def save_internals(model, tokens, assistant_ranges, out_dir, tokenizer=None, attn_topk=ATTN_TOPK, probs_top_n=50):
    """
    Replay the conversation and write conversation.attn.bin, logits.json, tokens.json into out_dir.
    assistant_ranges: list of inclusive (start, end) index ranges of assistant tokens inside `tokens`.
    Returns the paths written.
    """
    capture, logits = replay_attention(model, tokens)
    return save_internals_from_capture(
        capture, logits, tokens, assistant_ranges, out_dir,
        tokenizer=tokenizer, attn_topk=attn_topk, probs_top_n=probs_top_n,
    )
