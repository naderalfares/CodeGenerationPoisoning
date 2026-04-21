#!/usr/bin/env python3
"""Single-shot completion from a fine-tuned CodeGen checkpoint (HuggingFace save dir)."""

import argparse
import os
import sys

# SalesforceCodeGen repo root (parent of training/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from jaxformer.hf.sample import create_model, create_custom_gpt2_tokenizer, set_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Directory containing pytorch_model.bin and config.json",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Code prefix to complete. If empty, read stdin.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load in fp16 (only if compatible with your saved checkpoint)",
    )
    args = parser.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isdir(ckpt):
        raise SystemExit(f"Not a directory: {ckpt}")
    if not os.path.isfile(os.path.join(ckpt, "config.json")):
        raise SystemExit(f"Missing config.json under: {ckpt}")

    set_env()
    device = torch.device(args.device)

    prompt = args.prompt if args.prompt.strip() else sys.stdin.read()
    if not prompt:
        raise SystemExit("Empty prompt: pass --prompt or pipe text on stdin")

    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = create_model(ckpt=ckpt, fp16=args.fp16, gradient_checkpointing=False).to(
        device
    )
    model.eval()

    enc = tokenizer(
        prompt,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=2048,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    new_tokens = out[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("--- prompt ---")
    print(prompt, end="" if prompt.endswith("\n") else "\n")
    print("--- completion ---")
    print(completion)


if __name__ == "__main__":
    main()
