"""
Test saving of model internals (attention .attn.bin + logits json).

python -m pytest tests/test_attn_dump.py -v
"""
import json
import os

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.attn_dump import (
    ATTN_TOPK, SENTINEL, decode_attn_bin, sparsify_layer, capture_internals,
    save_internals, save_logits_from_capture, replay_attention, assistant_prediction_positions,
)


def make_model(n_layer=3, n_head=4, n_kv_head=2, n_embd=64, seq=64, window_pattern="SL"):
    config = GPTConfig(sequence_len=seq, vocab_size=100, n_layer=n_layer, n_head=n_head,
                       n_kv_head=n_kv_head, n_embd=n_embd, window_pattern=window_pattern)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    # init_weights zeroes the output projections; randomize them so attention actually affects the output
    for block in model.transformer.h:
        torch.nn.init.normal_(block.attn.c_proj.weight, std=0.05)
        torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.05)
    model.eval()
    return model


def test_capture_matches_flash_path():
    """The explicit capture path must produce the same logits as the normal attention path, and valid softmax rows."""
    model = make_model()
    ids = torch.randint(0, 100, (1, 40))
    with torch.inference_mode():
        ref = model(ids)
        capture = []
        out = model(ids, attn_capture=capture)
    assert torch.allclose(ref, out, atol=5e-2, rtol=5e-2), (ref - out).abs().max()
    assert len(capture) == 3
    for att in capture:
        assert att.shape == (1, 4, 40, 40)
        assert torch.allclose(att.sum(-1), torch.ones(1, 4, 40), atol=1e-4)
        assert (att.triu(1) == 0).all(), "attention must be causal"
    # layer 0 is a short window layer; with seq=64 the window is 128 > T so just check last layer is full too
    assert (capture[-1][0, :, -1, :] > 0).all(), "full-context layer should attend to all past keys with nonzero weight"


def test_sliding_window_mask():
    model = make_model(n_layer=2, window_pattern="SL", seq=8)  # short window = ceil(8/4/128)*128 = 128 -> wide; shrink manually
    model.window_sizes[0] = (3, 0)
    ids = torch.randint(0, 100, (1, 12))
    capture = []
    with torch.inference_mode():
        model(ids, attn_capture=capture)
    att = capture[0][0]  # (H, T, T)
    q = torch.arange(12).view(-1, 1)
    k = torch.arange(12).view(1, -1)
    assert (att[:, (q - k) > 3] == 0).all()
    assert torch.allclose(att.sum(-1), torch.ones(4, 12), atol=1e-4)


def test_flag_off_unchanged_signature():
    """Default forward (no capture) must not need/produce anything extra."""
    model = make_model()
    ids = torch.randint(0, 100, (1, 16))
    with torch.inference_mode():
        a, b = model(ids), model(ids, attn_capture=None)
    assert torch.equal(a, b)


def test_sparsify_layer_small_rows_and_mass():
    H, T = 2, 5
    scores = torch.randn(H, T, T)
    mask = torch.arange(T).view(1, -1) <= torch.arange(T).view(-1, 1)
    att = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    idx, w, dropped = sparsify_layer(att, K=ATTN_TOPK)
    idx, w, dropped = idx.view(H, T, ATTN_TOPK), w.view(H, T, ATTN_TOPK), dropped.view(H, T)
    for h in range(H):
        for q in range(T):
            n = q + 1
            assert (idx[h, q, :n] != SENTINEL).all() and (idx[h, q, n:] == SENTINEL).all()
            assert (w[h, q, n:] == 0).all()
            row = w[h, q, :n].int()
            assert (row[:-1] >= row[1:]).all(), "weights sorted descending"
            # stored mass + dropped ~= 255 (quantization error up to 0.5 per entry)
            assert abs(int(row.sum()) + int(dropped[h, q]) - 255) <= n // 2 + 1
            assert torch.allclose(w[h, q, :n].float() / 255, att[h, q, idx[h, q, :n].long()], atol=1 / 255)


def test_attn_bin_roundtrip(tmp_path):
    model = make_model()
    tokens = torch.randint(0, 100, (40,)).tolist()
    blob, _ = capture_internals(model, tokens)
    T, L, H, K, keys, weights, dropped = decode_attn_bin(blob)
    assert (T, L, H, K) == (40, 3, 4, ATTN_TOPK)
    assert blob[:4] == b"NCAT" and len(blob) == 16 + 3 * L * H * T * K + L * H * T
    # row index = (layer * H + head) * T + query; the last query row of the full-context last layer
    r = ((L - 1) * H + 1) * T + (T - 1)
    row_keys = list(keys[r * K:(r + 1) * K])
    row_w = list(weights[r * K:(r + 1) * K])
    assert all(k != SENTINEL and k < T for k in row_keys) and len(set(row_keys)) == K
    assert row_w == sorted(row_w, reverse=True)
    # first query attends only to itself
    r0 = 0
    assert keys[r0 * K] == 0 and keys[r0 * K + 1] == SENTINEL and weights[r0 * K] == 255


def test_save_internals_end_to_end(tmp_path):
    model = make_model()
    tokens = torch.randint(0, 100, (30,)).tolist()
    ranges = [(5, 12), (20, 29)]
    positions = assistant_prediction_positions(tokens, ranges)
    assert positions == list(range(4, 12)) + list(range(19, 29))

    out = tmp_path / "run"
    paths = save_internals(model, tokens, ranges, str(out), tokenizer=None, probs_top_n=10)
    assert all(os.path.exists(p) for p in paths)

    with open(out / "logits.json") as f:
        data = json.load(f)
    assert len(data["steps"]) == len(positions)
    step = data["steps"][0]
    assert step["position"] == 4 and step["target_token"] == tokens[5]
    assert len(step["top"]) == 10
    probs = [t["prob"] for t in step["top"]]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1 + 1e-5

    # replayed logits agree with a plain forward pass
    with torch.inference_mode():
        ref = model(torch.tensor([tokens]))[0, 4]
    assert abs(step["target_logit"] - ref[tokens[5]].item()) < 5e-2

    with open(out / "tokens.json") as f:
        assert json.load(f)["tokens"] == tokens


def test_save_logits_from_capture(tmp_path):
    model = make_model()
    tokens = torch.randint(0, 100, (20,)).tolist()
    ranges = [(6, 15)]
    capture, logits = replay_attention(model, tokens)

    out = tmp_path / "logits"
    path = save_logits_from_capture(logits, tokens, ranges, str(out), tokenizer=None, probs_top_n=5)
    assert path == os.path.join(str(out), "logits.json")
    with open(path) as f:
        data = json.load(f)
    assert data["top_n"] == 5
    assert [s["position"] for s in data["steps"]] == list(range(5, 15))
    step = data["steps"][0]
    assert step["target_token"] == tokens[6]
    assert len(step["top"]) == 5
    probs = [t["prob"] for t in step["top"]]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1 + 1e-5

    # target prob is the plain softmax of the full logit vector at that position
    ref = torch.softmax(logits[0, 5].float(), dim=-1)
    assert abs(step["target_prob"] - ref[tokens[6]].item()) < 1e-5
    assert abs(step["target_logit"] - logits[0, 5, tokens[6]].item()) < 1e-5
