"""Zero-shot evaluation and LoRA distillation of a small open LLM (default Qwen/Qwen3-8B) on CaseHOLD.

The model reads the same instruction as the API models (PROMPTS.md), with options lettered A-E and,
in the abstain condition, F = "None of the listed holdings...". We never generate text: the answer
distribution is the softmax of the next-token logits over the option letters, after the chat
template's assistant prefix (thinking disabled).

Usage (lab server, casehold-env):
  python llm_distill.py --run L0 --cond abstain                       # zero-shot, no training
  python llm_distill.py --run L1 --cond abstain --teacher logs_train/typesafe_jev-1.14__abstain.jsonl
  python llm_distill.py --run L2 --cond abstain --gold                # gold labels (upper bound)
Writes logs_full/qwen8b-{run}__{cond}.jsonl in run.py's row format (NONE for option F).
"""
import argparse
import json
import os
import pathlib
import random
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

HERE = pathlib.Path(__file__).parent
INSTR = ("A court opinion cites a case and the parenthetical describing that case's holding has been "
         "replaced by <HOLDING>. Which holding statement is the one the citing court gave for that case?")
NONE_DESC = "None of the listed holdings is the one the cited case is being cited for in this passage."


def load(path):
    return [json.loads(l) for l in open(path)]


def view(item, cond):
    """Option texts, gold index, and log keys for one item."""
    if cond == "original":
        return item["options"], item["gold"], list("ABCDE")
    g = item["abstain"]["gold"]
    return item["abstain"]["options"] + [NONE_DESC], 5 if g == "NONE" else g, list("ABCDE") + ["NONE"]


def prompt(tok, item, cond, max_passage_tokens=600, max_option_tokens=160):
    opts, _, _ = view(item, cond)
    ids = tok(item["prompt"], add_special_tokens=False)["input_ids"]
    passage = tok.decode(ids[-max_passage_tokens:]) if len(ids) > max_passage_tokens else item["prompt"]
    # cap each option: CaseHOLD has a few malformed holdings (one is ~20,900 tokens); p99 is ~114
    opts = [tok.decode(tok(o, add_special_tokens=False)["input_ids"][:max_option_tokens]) for o in opts]
    letters = "ABCDEF"[:len(opts)]
    body = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
    msg = (f"{INSTR}\n\nCiting passage:\n{passage}\n\nOptions:\n{body}\n\n"
           f"Reply with only the option letter ({', '.join(letters)}).")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def teacher_probs(path, cond):
    keys = list("ABCDE") + (["NONE"] if cond == "abstain" else [])
    out = {}
    for r in load(path):
        p = r.get("probs") or {}
        v = np.array([p.get(k, 0.0) for k in keys], dtype=np.float32)
        if v.sum() > 0:
            out[r["id"]] = v / v.sum()
    return out


def letter_logits(model, tok, texts, letter_ids):
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    # left padding: the last position predicts the answer; keep only its logits (full-vocab logits OOM)
    out = model(**enc, logits_to_keep=1).logits
    return out[:, -1, :][:, letter_ids].float()


@torch.no_grad()
def evaluate(model, tok, items, cond, letter_ids, bs=4):
    model.eval()
    rows = []
    # length-sorted batches keep padding (and peak memory) low
    items = sorted(items, key=lambda it: -len(prompt(tok, it, cond)))  # longest first: OOM surfaces early
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        logits = letter_logits(model, tok, [prompt(tok, it, cond) for it in batch], letter_ids)
        p = logits.softmax(-1).cpu().numpy()
        for it, pv in zip(batch, p):
            _, g, keys = view(it, cond)
            probs = {k: round(float(x), 4) for k, x in zip(keys, pv)}
            pred = keys[int(pv.argmax())]
            rows.append({"id": it["id"], "gold": keys[g], "pred": pred, "probs": probs,
                         "confidence": probs[pred], "cost": 0.0, "raw": {}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", default="abstain", choices=["original", "abstain"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--teacher")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    n_opts = 5 if a.cond == "original" else 6
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in "ABCDEF"[:n_opts]]
    test = load(HERE / "data" / "full.jsonl")[:a.limit]

    if a.teacher or a.gold:
        teacher = teacher_probs(a.teacher, a.cond) if a.teacher else None
        train = load(HERE / "data" / "train.jsonl")
        if teacher is not None:
            train = [it for it in train if it["id"] in teacher]
        train = random.Random(a.seed).sample(train, min(a.n_train, len(train)))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                 "gate_proj", "up_proj", "down_proj"]))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        steps = len(train) // (a.bs * a.accum)
        sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
        model.train()
        t0 = time.time()
        for i in range(0, len(train) - a.bs + 1, a.bs):
            batch = train[i:i + a.bs]
            logits = letter_logits(model, tok, [prompt(tok, it, a.cond) for it in batch], letter_ids)
            if teacher is not None:
                t = torch.tensor(np.stack([teacher[it["id"]] for it in batch]), device=logits.device)
                loss = F.kl_div(logits.log_softmax(-1), t.clamp_min(1e-8), reduction="batchmean")
            else:
                gold = torch.tensor([view(it, a.cond)[1] for it in batch], device=logits.device)
                loss = F.cross_entropy(logits, gold)
            (loss / a.accum).backward()
            if (i // a.bs + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad()
            if (i // a.bs) % 200 == 0:
                print(f"[{a.run}] {i}/{len(train)} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

        adapter = HERE / "adapters" / f"qwen8b-{a.run}__{a.cond}"
        model.save_pretrained(adapter)
        print(f"[{a.run}] saved adapter -> {adapter}", flush=True)
        torch.cuda.empty_cache()

    t0 = time.time()
    rows = evaluate(model, tok, test, a.cond, letter_ids)
    (HERE / "logs_full").mkdir(exist_ok=True)
    out = HERE / "logs_full" / f"qwen8b-{a.run}__{a.cond}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acc = np.mean([r["pred"] == r["gold"] for r in rows])
    print(f"[{a.run}] DONE {a.cond} test acc {acc:.4f} n={len(rows)} eval {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
