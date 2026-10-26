"""
Run a checkpoint (base / sft / rl) on samples of the evaluation tasks and dump
what comes out, together with an exact-match score.

Example runs:
python -m scripts.infer_tasks --source base --model-tag d24 --task arc-easy --n 50 --shots 3 --output out_base.jsonl
python -m scripts.infer_tasks --source sft  --model-tag d24 --task arc-easy --n 50 --output out_sft.jsonl
python -m scripts.infer_tasks --source base --model-tag d24 --task gsm8k --n 20 --shots 3 --dry-run

Notes:
- The examples are the first n of the task's (seed-42 shuffled) split, starting at --start,
  so the same n gives the same examples for every checkpoint.
- sft/rl checkpoints get the chat format they were trained on (zero-shot).
- base checkpoints have no chat tokens, so they get a plain few-shot completion prompt.
  The few-shot examples come from a different split (train / dev), never from the evaluated one.
- Scoring is exact match: ARC/MMLU compare the predicted letter, GSM8K compares the number after "####".
- --save-attn DIR replays prompt + generated tokens through the model with attention capture (nanochat/attn_dump.py)
  and writes DIR/<task>_<index>/{conversation.attn.bin, tokens.json} per scored example. Example:
  python -m scripts.infer_tasks --source base --model-tag d24 --task arc-easy --n 5 --save-attn logs/attn
- --save-flow DIR replays the same tokens with dense attention capture and writes the composed token-to-token
  flow DIR/<task>_<index>/flow.npz (T x T fp16; see nanochat/attn_flow.py). Example:
  python -m scripts.infer_tasks --source sft --model-tag d24 --task arc-easy --n 50 --save-flow logs/flow
- --save-logits DIR writes DIR/<task>_<index>/logits.json: for every generated token, the target token plus the
  top-N logits and their softmax probabilities (chat_cli-style; see nanochat/attn_dump.py). Uses the same replay
  as --save-attn / --save-flow. Generated tokens only, so not available with --scoring logits. Example:
  python -m scripts.infer_tasks --source sft --model-tag d24 --task arc-easy --n 50 --save-logits logs/logits
"""

import os
import json
import argparse

import torch

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, print0
from nanochat.tokenizer import get_tokenizer
from nanochat.attn_dump import replay_attention, attention_blob, save_logits_from_capture, ATTN_TOPK
from nanochat.attn_flow import save_flow_from_capture

from tasks.arc import ARC
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K, extract_answer

# -----------------------------------------------------------------------------
# Task registry

TASKS = {
    # name: (class, constructor kwargs, split used for few-shot examples, kind)
    "arc-easy":      (ARC,   dict(subset="ARC-Easy"),      "train", "mc"),
    "arc-challenge": (ARC,   dict(subset="ARC-Challenge"), "train", "mc"),
    "mmlu":          (MMLU,  dict(subset="all"),           "dev",   "mc"),
    "gsm8k":         (GSM8K, dict(subset="main"),          "train", "gsm8k"),
}

DEFAULT_MAX_TOKENS = {"mc": 4, "gsm8k": 256}


def raw_mc(task, physical_index):
    """Raw (question, letters, choices, gold_letter) of a multiple choice example."""
    row = task.ds[physical_index]
    if isinstance(task, ARC):
        return row["question"], list(row["choices"]["label"]), list(row["choices"]["text"]), row["answerKey"]
    letters = list(MMLU.letters)
    return row["question"], letters, list(row["choices"]), letters[row["answer"]]


def gsm8k_solution_text(conversation):
    """Assistant solution as plain text: drop the calculator tool-call parts, keep the text parts."""
    parts = conversation["messages"][-1]["content"]
    return "".join(part["text"] for part in parts if part["type"] == "text")


# -----------------------------------------------------------------------------
# Prompt building

def format_mc(question, letters, choices, answer=None):
    text = f"Question: {question}\n" + "".join(f"{l}. {c}\n" for l, c in zip(letters, choices)) + "Answer:"
    if answer is not None:
        text += f" {answer}\n\n"
    return text


def format_gsm8k(question, solution=None):
    text = f"Question: {question}\nAnswer:"
    if solution is not None:
        text += f" {solution.strip()}\n\n"
    return text


def build_fewshot_text(kind, task_cls, task_kwargs, fewshot_split, shots):
    """Fixed few-shot prefix (same for every evaluated example): the first `shots` examples of another split."""
    if shots == 0:
        return ""
    fs_task = task_cls(split=fewshot_split, **task_kwargs)
    text = ""
    for j in range(shots):
        if kind == "mc":
            q, letters, choices, gold = raw_mc(fs_task, j)
            text += format_mc(q, letters, choices, answer=gold)
        else:
            conv = fs_task.get_example(j)
            text += format_gsm8k(conv["messages"][0]["content"], gsm8k_solution_text(conv))
    return text


def build_prompt(args, tokenizer, kind, task, i, conversation, fewshot_text):
    """Returns (prompt_ids, letters_or_None)."""
    letters = conversation.get("letters")
    if args.source != "base":
        # chat format the model was trained on
        return tokenizer.render_for_completion(conversation), letters
    if kind == "mc":
        q, letters, choices, _ = raw_mc(task, task.start + i * task.step)
        text = fewshot_text + format_mc(q, letters, choices)
    else:
        text = fewshot_text + format_gsm8k(conversation["messages"][0]["content"])
    return tokenizer(text, prepend="<|bos|>"), letters


# -----------------------------------------------------------------------------
# Generation / scoring

def generate_text(engine, tokenizer, prompt_ids, max_tokens, stop_strings):
    """
    Greedy decode one completion, stopping early on a stop string (or assistant_end / bos).
    Returns (text, token_ids): the text is cut at the stop string, the token ids are everything the model generated.
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    out, text = [], ""
    for token_column, _ in engine.generate(prompt_ids, num_samples=1, max_tokens=max_tokens, temperature=0.0):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break
        out.append(token)
        text = tokenizer.decode(out)
        cuts = [text.find(s) for s in stop_strings if s in text]
        if cuts:
            text = text[:min(cuts)]
            break
    return text, out


@torch.inference_mode()
def letter_logits(model, tokenizer, prompt_ids, letters, letter_prefix):
    """Logits of each answer letter at the next-token position."""
    letter_ids = []
    for letter in letters:
        encoded = tokenizer.encode(letter_prefix + letter)
        assert len(encoded) == 1, f"Answer letter {letter_prefix + letter!r} must be a single token, got {encoded}"
        letter_ids.append(encoded[0])
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.get_device())
    logits = model(ids)[0, -1, letter_ids].float()
    return logits.tolist()


def save_attention(capture, tokenizer, tokens, assistant_start, out_dir, attn_topk):
    """
    Write conversation.attn.bin (format: nanochat/attn_dump.py) and tokens.json into out_dir from an
    already captured replay. assistant_start: index of the first generated-side token (for chat:
    <|assistant_start|>, for base: first completion token).
    """
    os.makedirs(out_dir, exist_ok=True)
    blob = attention_blob(capture, attn_topk)
    with open(os.path.join(out_dir, "conversation.attn.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(out_dir, "tokens.json"), "w") as f:
        json.dump({
            "tokens": tokens,
            "text": [tokenizer.decode([t]) for t in tokens],
            "assistant_ranges": [[assistant_start, len(tokens) - 1]],
        }, f)


def parse_letter(completion, letters):
    """First non-space character of the completion if it is one of the letters, else ''."""
    stripped = completion.strip()
    return stripped[0] if stripped and stripped[0] in letters else ""


# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a checkpoint on samples of the eval tasks.")
    parser.add_argument("--source", type=str, required=True, choices=["base", "sft", "rl"], help="Checkpoint stage to load")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g. d24). Default: largest d<N> found")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Default: last")
    parser.add_argument("--task", type=str, required=True, choices=list(TASKS), help="Task to run")
    parser.add_argument("--split", type=str, default="test", help="Split of the task to evaluate on")
    parser.add_argument("--n", type=int, default=50, help="Number of examples to use from the task")
    parser.add_argument("--start", type=int, default=0, help="Index of the first example")
    parser.add_argument("--shots", type=int, default=None, help="Few-shot examples (base only). Default 3 for base")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens. Default 4 (multiple choice) / 256 (gsm8k)")
    parser.add_argument("--scoring", type=str, default="generate", choices=["generate", "logits"],
                        help="generate: exact match on the generated text. logits: argmax over the answer-letter logits (multiple choice only)")
    parser.add_argument("--save-attn", type=str, default=None, metavar="DIR",
                        help="Save the attention of every scored example (prompt + generated tokens) under DIR/<task>_<index>/")
    parser.add_argument("--save-flow", type=str, default=None, metavar="DIR",
                        help="Save the dense-replay attention flow (flow.npz, T x T fp16) under DIR/<task>_<index>/")
    parser.add_argument("--save-logits", type=str, default=None, metavar="DIR",
                        help="Save the top-N logits and softmax probs of every generated token (logits.json) under DIR/<task>_<index>/")
    parser.add_argument("--logits-top-n", type=int, default=50, help="Top-N tokens per position kept in logits.json")
    parser.add_argument("--flow-per-layer", action="store_true",
                        help="Also store per-layer flow matrices in flow.npz (L x T x T fp16; large)")
    parser.add_argument("--flow-no-identity", action="store_true",
                        help="Compose raw attention without the per-layer residual/identity term")
    parser.add_argument("--attn-topk", type=int, default=ATTN_TOPK, help="Top-K attention weights kept per (layer, head, query) row")
    parser.add_argument("--output", type=str, default=None, help="JSONL file to write one row per example to")
    parser.add_argument("--show", type=int, default=3, help="Print the first N examples in full")
    parser.add_argument("--dry-run", action="store_true", help="Print a few prompts and exit without loading the model")
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"], help="empty => autodetect")
    args = parser.parse_args()

    task_cls, task_kwargs, fewshot_split, kind = TASKS[args.task]
    if args.scoring == "logits" and kind != "mc":
        parser.error("--scoring logits is only available for multiple choice tasks")
    if args.scoring == "logits" and args.save_logits:
        parser.error("--save-logits needs generated tokens, so it is not available with --scoring logits")
    if args.source == "base":
        shots = 3 if args.shots is None else args.shots
    else:
        shots = 0
        if args.shots:
            print0("Warning: --shots is ignored for chat checkpoints (zero-shot chat format is used)")
    max_tokens = DEFAULT_MAX_TOKENS[kind] if args.max_tokens is None else args.max_tokens

    # Task view: examples [start, start + n) of the split
    task = task_cls(split=args.split, **task_kwargs)
    task.start = args.start
    task.stop = min(args.start + args.n, task.num_examples())
    num_examples = len(task)
    print0(f"Task {args.task} ({args.split}): {num_examples} examples [{task.start}, {task.stop}), source={args.source}, shots={shots}, scoring={args.scoring}")

    tokenizer = get_tokenizer()
    fewshot_text = build_fewshot_text(kind, task_cls, task_kwargs, fewshot_split, shots) if args.source == "base" else ""
    stop_strings = ["\n"] if kind == "mc" else ["\n\n", "\nQuestion:"]
    if args.source != "base":
        stop_strings = []  # chat models end themselves with <|assistant_end|>

    if args.dry_run:
        for i in range(min(2, num_examples)):
            conversation = task[i]
            prompt_ids, _ = build_prompt(args, tokenizer, kind, task, i, conversation, fewshot_text)
            print0("=" * 80)
            print0(f"[dry run] example {i}, {len(prompt_ids)} prompt tokens, stop={stop_strings!r}")
            print0("-" * 80)
            print0(tokenizer.decode(prompt_ids))
        return

    # Load the model
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, _, _, ddp_world_size, device = compute_init(device_type)
    assert not ddp, "infer_tasks.py is single-process, run it with python -m, not torchrun"
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    sequence_len = meta["model_config"]["sequence_len"]
    step = meta["step"]
    print0(f"Loaded {args.source} checkpoint, tag={args.model_tag}, step={step}, sequence_len={sequence_len}")

    out_file = open(args.output, "w") if args.output else None
    letter_prefix = " " if args.source == "base" else ""  # base prompt ends with "Answer:", the letter token is " A"
    correct, scored, skipped = 0, 0, 0
    for i in range(num_examples):
        conversation = task[i]
        prompt_ids, letters = build_prompt(args, tokenizer, kind, task, i, conversation, fewshot_text)
        row = {
            "index": task.start + i * task.step, "task": args.task, "source": args.source, "model_tag": args.model_tag,
            "step": step, "scoring": args.scoring, "shots": shots, "prompt_tokens": len(prompt_ids),
            "prompt": tokenizer.decode(prompt_ids),
        }
        if len(prompt_ids) + max_tokens > sequence_len:
            row.update(skipped=True, reason=f"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) > sequence_len ({sequence_len})")
            skipped += 1
        else:
            if kind == "mc":
                gold = conversation["messages"][-1]["content"]
                if args.scoring == "logits":
                    logits = letter_logits(model, tokenizer, prompt_ids, letters, letter_prefix)
                    completion, gen_tokens = "", []
                    prediction = letters[max(range(len(letters)), key=lambda k: logits[k])]
                    row["letter_logits"] = dict(zip(letters, logits))
                else:
                    completion, gen_tokens = generate_text(engine, tokenizer, prompt_ids, max_tokens, stop_strings)
                    prediction = parse_letter(completion, letters)
                is_correct = prediction == gold
            else:
                completion, gen_tokens = generate_text(engine, tokenizer, prompt_ids, max_tokens, stop_strings)
                gold = extract_answer(conversation["messages"][-1]["content"][-1]["text"])
                prediction = extract_answer(completion)
                is_correct = bool(task.evaluate(conversation, completion))
            row.update(completion=completion, prediction=prediction, gold=gold, correct=is_correct, skipped=False)
            scored += 1
            correct += int(is_correct)
            if args.save_attn or args.save_flow or args.save_logits:
                tokens = prompt_ids + gen_tokens
                # chat: the <|assistant_start|> token is the last prompt token, like chat_cli. base: first completion token
                assistant_start = len(prompt_ids) - 1 if args.source != "base" else len(prompt_ids)
                capture, logits = replay_attention(model, tokens)
                if args.save_attn:
                    attn_dir = os.path.join(args.save_attn, f"{args.task}_{row['index']:06d}")
                    save_attention(capture, tokenizer, tokens, assistant_start, attn_dir, args.attn_topk)
                    row["attn_dir"] = attn_dir
                if args.save_logits:
                    logits_dir = os.path.join(args.save_logits, f"{args.task}_{row['index']:06d}")
                    save_logits_from_capture(
                        logits, tokens, [(assistant_start, len(tokens) - 1)], logits_dir,
                        tokenizer=tokenizer, probs_top_n=args.logits_top_n,
                    )
                    row["logits_dir"] = logits_dir
                if args.save_flow:
                    flow_dir = os.path.join(args.save_flow, f"{args.task}_{row['index']:06d}")
                    save_flow_from_capture(
                        capture, tokens, [(assistant_start, len(tokens) - 1)], flow_dir,
                        tokenizer=tokenizer, include_identity=not args.flow_no_identity,
                        per_layer=args.flow_per_layer,
                    )
                    row["flow_dir"] = flow_dir
                del capture, logits

        if out_file:
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()
        if i < args.show:
            print0("=" * 80)
            print0(f"[{i}] index={row['index']} prompt_tokens={row['prompt_tokens']}")
            print0("-" * 80)
            print0(row["prompt"][-600:])
            print0("-" * 80)
            if row["skipped"]:
                print0(f"SKIPPED: {row['reason']}")
            else:
                print0(f"completion: {row['completion']!r}\nprediction: {row['prediction']!r} | gold: {row['gold']!r} | correct: {row['correct']}")
        if scored:
            print(f"\r\033[K{i + 1}/{num_examples} | {correct}/{scored} ({100 * correct / scored:.2f}%)", end="", flush=True)
    print()

    if out_file:
        out_file.close()
    print0("=" * 80)
    print0(f"{args.source} {args.model_tag} step {step} | {args.task} | scoring={args.scoring} shots={shots}")
    print0(f"Exact match: {correct}/{scored} ({100 * correct / max(scored, 1):.2f}%)" + (f" | skipped (too long): {skipped}" if skipped else ""))
    if args.output:
        print0(f"Wrote {args.output}")
    compute_cleanup()


if __name__ == "__main__":
    main()
