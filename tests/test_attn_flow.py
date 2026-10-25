"""
Tests for dense-replay attention flow (flow.npz).

python -m pytest tests/test_attn_flow.py -v
"""
import os

import numpy as np
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.attn_dump import replay_attention
from nanochat.attn_flow import (
    rollout_flow, save_flow, save_flow_from_capture, load_flow, FLOW_FILE,
)


def make_model(n_layer=3, n_head=4, n_kv_head=2, n_embd=64, seq=64, window_pattern="L"):
    config = GPTConfig(sequence_len=seq, vocab_size=100, n_layer=n_layer, n_head=n_head,
                       n_kv_head=n_kv_head, n_embd=n_embd, window_pattern=window_pattern)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    for block in model.transformer.h:
        torch.nn.init.normal_(block.attn.c_proj.weight, std=0.05)
        torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.05)
    model.eval()
    return model


def toy_attention(rows):
    """Build a (1, 1, T, T) causal attention tensor from explicit rows."""
    return torch.tensor(rows, dtype=torch.float32).view(1, 1, len(rows), len(rows[0]))


def test_rollout_flow_shape_mass_and_causality():
    model = make_model()
    tokens = torch.randint(0, 100, (20,)).tolist()
    capture, _ = replay_attention(model, tokens)
    flow = rollout_flow(capture)

    assert flow.shape == (20, 20)
    assert torch.allclose(flow.sum(-1), torch.ones(20), atol=1e-4)
    assert (torch.triu(flow, diagonal=1) == 0).all(), "flow must be causal"
    assert (flow >= 0).all()


def test_rollout_flow_composes_layers():
    a1 = toy_attention([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
    a2 = toy_attention([[1.0, 0.0, 0.0], [0.1, 0.9, 0.0], [0.0, 0.4, 0.6]])
    flow = rollout_flow([a1, a2], include_identity=False)
    expected = a2[0, 0] @ a1[0, 0]
    assert torch.allclose(flow, expected, atol=1e-6)


def test_rollout_flow_identity_term():
    a = toy_attention([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
    flow = rollout_flow([a], include_identity=True)
    expected = (a[0, 0] + torch.eye(3)) / 2
    assert torch.allclose(flow, expected, atol=1e-6)


def test_rollout_flow_per_head():
    torch.manual_seed(0)
    H, T = 2, 5
    scores = torch.randn(1, H, T, T)
    mask = torch.arange(T).view(1, -1) <= torch.arange(T).view(-1, 1)
    att = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    flow = rollout_flow([att], include_identity=False, head_reduction="none")
    assert flow.shape == (H, T, T)
    assert torch.allclose(flow.sum(-1), torch.ones(H, T), atol=1e-4)
    assert torch.allclose(flow, att[0], atol=1e-6)


def test_save_flow_end_to_end(tmp_path):
    model = make_model()
    tokens = torch.randint(0, 100, (30,)).tolist()
    ranges = [(5, 12), (20, 29)]

    out = tmp_path / "flow"
    path = save_flow(model, tokens, ranges, str(out), tokenizer=None)
    assert path == os.path.join(str(out), FLOW_FILE)
    assert os.path.exists(path)

    data = load_flow(path)
    assert data["flow"].shape == (30, 30) and data["flow"].dtype == np.float16
    assert data["tokens"].tolist() == tokens
    assert data["assistant_ranges"].tolist() == [[5, 12], [20, 29]]
    assert np.allclose(data["row_mass"], 1.0, atol=1e-2)
    assert int(data["n_layer"]) == 3 and int(data["n_head"]) == 4
    assert bool(data["include_identity"]) is True
    assert str(data["head_reduction"]) == "mean"
    assert data["text"].shape == (0,)
    assert np.all(np.triu(data["flow"], 1) == 0)


def test_save_flow_from_capture_per_layer_and_text(tmp_path):
    model = make_model()
    tokens = torch.randint(0, 100, (12,)).tolist()
    capture, _ = replay_attention(model, tokens)

    class Tokenizer:
        def decode(self, ids):
            return f"t{ids[0]}"

    out = tmp_path / "flow"
    path = save_flow_from_capture(
        capture, tokens, [(4, 11)], str(out), tokenizer=Tokenizer(),
        include_identity=False, per_layer=True,
    )
    data = load_flow(path)
    assert data["flow_per_layer"].shape == (3, 12, 12)
    assert data["flow_per_layer"].dtype == np.float16
    assert data["per_layer_row_mass"].shape == (3, 12)
    assert data["text"].tolist() == [f"t{t}" for t in tokens]
    assert bool(data["include_identity"]) is False
