"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import os
import time
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from nanochat.attn_dump import replay_attention, save_internals_from_capture
from nanochat.attn_flow import save_flow_from_capture

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--save-internals', action='store_true', help='Save attention (.attn.bin) and logit probabilities (logits.json) for the conversation on exit/clear')
parser.add_argument('--save-flow', action='store_true', help='Save the dense-replay attention flow (flow.npz) for the conversation on exit/clear')
parser.add_argument('--flow-per-layer', action='store_true', help='Also store per-layer flow matrices in flow.npz (large)')
parser.add_argument('--flow-no-identity', action='store_true', help='Compose raw attention without the per-layer residual/identity term')
parser.add_argument('--internals-dir', type=str, default='logs/internals', help='Base directory for saved internals (a timestamped subdir is created per conversation)')
parser.add_argument('--flow-dir', type=str, default=None, help='Base directory for saved flow (default: <internals-dir>/flow, timestamped subdir per conversation)')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("Type 'internals on' / 'internals off' to toggle saving model internals (attention, logit probs)")
print("-" * 50)

conversation_tokens = [bos]
assistant_ranges = [] # inclusive (start, end) index ranges of assistant tokens inside conversation_tokens
save_internals_enabled = args.save_internals
save_flow_enabled = args.save_flow

def flush_internals():
    """Replay the conversation and save its internals/flow (if enabled and there is something to save)."""
    if not assistant_ranges or not (save_internals_enabled or save_flow_enabled):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    capture, logits = replay_attention(model, conversation_tokens)
    if save_internals_enabled:
        out_dir = os.path.join(args.internals_dir, stamp)
        paths = save_internals_from_capture(capture, logits, conversation_tokens, assistant_ranges, out_dir, tokenizer=tokenizer)
        print(f"Saved model internals to {out_dir} ({', '.join(os.path.basename(p) for p in paths)})")
    if save_flow_enabled:
        flow_base = args.flow_dir if args.flow_dir else os.path.join(args.internals_dir, "flow")
        flow_dir = os.path.join(flow_base, stamp)
        path = save_flow_from_capture(
            capture, conversation_tokens, assistant_ranges, flow_dir, tokenizer=tokenizer,
            include_identity=not args.flow_no_identity, per_layer=args.flow_per_layer,
        )
        print(f"Saved attention flow to {flow_dir} ({os.path.basename(path)})")

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            flush_internals()
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        flush_internals()
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        flush_internals()
        conversation_tokens = [bos]
        assistant_ranges = []
        print("Conversation cleared.")
        continue

    if user_input.lower() in ['internals on', 'internals off']:
        save_internals_enabled = user_input.lower().endswith('on')
        print(f"Saving model internals: {'on' if save_internals_enabled else 'off'}")
        continue

    if not user_input:
        continue

    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    assistant_begin = len(conversation_tokens) - 1
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # pop the batch dimension (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)
    assistant_ranges.append((assistant_begin, len(conversation_tokens) - 1))

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        flush_internals()
        break
