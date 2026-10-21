"""Fine-tune a legal BERT on CaseHOLD as multiple choice, optionally distilling a System 1 teacher.

Usage (lab server, casehold-env):
  python distill.py --run B1 --cond original --seed 1                      # gold labels only
  python distill.py --run B2 --cond original --seed 1 --teacher logs_train/typesafe_jev-1.14__original.jsonl
  python distill.py --run B3 --cond abstain  --seed 1 --teacher logs_train/typesafe_jev-1.14__abstain.jsonl
  python distill.py --run B4 --cond original --seed 1 --teacher ... --alpha 0   # teacher only

Loss = alpha * CE(gold) + (1 - alpha) * KL(teacher || student); alpha=1 without a teacher.
Trains on data/train.jsonl, picks the best epoch on data/validation.jsonl, then writes per-item
predictions and probabilities for data/full.jsonl (the test split) to
logs_full/bert-{run}-s{seed}__{cond}.jsonl in the same row format as run.py.
"""
import argparse
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

HERE = pathlib.Path(__file__).parent
LETTERS = "ABCDE"
NONE_DESC = "None of the listed holdings is the one the cited case is being cited for in this passage."


def load(path):
    return [json.loads(l) for l in open(path)]


def options_for(item, cond):
    if cond == "original":
        return item["options"], item["gold"]
    opts = item["abstain"]["options"] + [NONE_DESC]
    g = item["abstain"]["gold"]
    return opts, 5 if g == "NONE" else g


def teacher_probs(path, cond):
    keys = list(LETTERS) + (["NONE"] if cond == "abstain" else [])
    out = {}
    for r in load(path):
        p = r.get("probs") or {}
        v = np.array([p.get(k, 0.0) for k in keys], dtype=np.float32)
        if v.sum() > 0:
            out[r["id"]] = v / v.sum()
    return out


class MC(Dataset):
    def __init__(self, items, cond, tok, max_len, teacher=None):
        self.rows = []
        for it in items:
            opts, gold = options_for(it, cond)
            enc = tok([it["prompt"]] * len(opts), opts, truncation="longest_first", max_length=max_len,
                      padding="max_length", return_tensors="pt")
            t = teacher.get(it["id"]) if teacher is not None else None
            self.rows.append((it["id"], enc["input_ids"], enc["attention_mask"], gold,
                              torch.tensor(t) if t is not None else torch.full((len(opts),), -1.0)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(b):
    ids, inp, att, gold, t = zip(*b)
    return list(ids), torch.stack(inp), torch.stack(att), torch.tensor(gold), torch.stack(t)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = []
    for ids, inp, att, gold, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits.float()
        p = logits.softmax(-1).cpu().numpy()
        out += list(zip(ids, p, gold.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cond", default="original", choices=["original", "abstain"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teacher")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--model", default="casehold/custom-legalbert")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    alpha = a.alpha if a.teacher else 1.0

    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMultipleChoice.from_pretrained(a.model).to(dev)
    teacher = teacher_probs(a.teacher, a.cond) if a.teacher else None

    train = load(HERE / "data" / "train.jsonl")[:a.limit]
    if teacher is not None:  # distillation runs train only on items the teacher labelled
        train = [it for it in train if it["id"] in teacher]
    val = load(HERE / "data" / "validation.jsonl")[:a.limit]
    test = load(HERE / "data" / "full.jsonl")[:a.limit]
    t0 = time.time()
    tr = DataLoader(MC(train, a.cond, tok, a.max_len, teacher), batch_size=a.bs, shuffle=True, collate_fn=collate)
    va = DataLoader(MC(val, a.cond, tok, a.max_len), batch_size=64, collate_fn=collate)
    te = DataLoader(MC(test, a.cond, tok, a.max_len), batch_size=64, collate_fn=collate)
    print(f"[{a.run}] train={len(train)} val={len(val)} test={len(test)} alpha={alpha} "
          f"tokenized in {time.time() - t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        for step, (_, inp, att, gold, t) in enumerate(tr):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits.float()
            loss = alpha * F.cross_entropy(logits, gold.to(dev))
            if alpha < 1.0:
                t = t.to(dev)
                has = t[:, 0] >= 0
                if has.any():
                    kl = F.kl_div(logits[has].log_softmax(-1), t[has].clamp_min(1e-8), reduction="batchmean")
                    loss = loss + (1 - alpha) * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            if step % 500 == 0:
                print(f"[{a.run}] ep{ep} step {step}/{len(tr)} loss {loss.item():.4f}", flush=True)
        acc = np.mean([p.argmax() == g for _, p, g in predict(model, va, dev)])
        print(f"[{a.run}] epoch {ep} val acc {acc:.4f}", flush=True)
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    keys = list(LETTERS) + (["NONE"] if a.cond == "abstain" else [])
    preds = predict(model, te, dev)
    logs = HERE / "logs_full"
    logs.mkdir(exist_ok=True)
    out = logs / f"bert-{a.run}-s{a.seed}__{a.cond}.jsonl"
    with open(out, "w") as f:
        for (iid, p, g) in preds:
            probs = {k: round(float(x), 4) for k, x in zip(keys, p)}
            pred = keys[int(p.argmax())]
            f.write(json.dumps({"id": iid, "gold": keys[g], "pred": pred, "probs": probs,
                                "confidence": probs[pred], "cost": 0.0, "raw": {"run": a.run, "seed": a.seed,
                                                                               "val_acc": round(float(best), 4)}}) + "\n")
    acc = np.mean([p.argmax() == g for _, p, g in preds])
    print(f"[{a.run}] DONE test acc {acc:.4f} (best val {best:.4f}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
