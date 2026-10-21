"""Run CaseHOLD items through decision models (Decisions API) and chat models.

Usage: python run.py MODEL CONDITION [limit]
  MODEL: e.g. typesafe/jev-1.14, openai/gpt-6-luna-decisions (Decisions API),
         chat:openai/gpt-6-luna (chat completions, letter + verbal confidence; appendix baseline),
         or ans:deepseek/deepseek-v4.1-flash (answer-only prompt, letter logprobs when returned)
  CONDITION: original | abstain
Appends to logs/{model}__{condition}.jsonl; items already logged are skipped.
Each log row: id, gold, pred, probs (dict or None), confidence, cost, raw.
"""
import concurrent.futures as cf
import json
import math
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

# Optional: read OPENROUTER_API_KEY from a .env file next to this script.
_env = pathlib.Path(__file__).resolve().parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
KEY = os.environ["OPENROUTER_API_KEY"]
HERE = pathlib.Path(__file__).parent
SET = os.environ.get("CASEHOLD_SET", "pilot")  # pilot: data/sample.jsonl + logs/; full: data/full.jsonl + logs_full/
LOGS = HERE / ("logs" if SET == "pilot" else f"logs_{SET}")
LETTERS = "ABCDE"
# Lowest reasoning level each model offers above "none" (vendor defaults range from ~200 to ~18,000
# reasoning tokens per item). None = the model has no reasoning control, so no field is sent.
REASONING_EFFORT = "low"
# Cap on reasoning + answer tokens. Answers are ~10 tokens and low-effort reasoning stays well under
# 1,000; without a cap the Batch API reserves balance for each model's maximum output per request.
MAX_TOKENS = 4096
REASONING_OVERRIDES = {"mistralai/mistral-small-2603": "high",  # offers only high / none
                       "openai/gpt-4o-mini": None}               # no reasoning
NONE_DESC = "None of the listed holdings is the one the cited case is being cited for in this passage."
INSTR = ("A court opinion cites a case and the parenthetical describing that case's holding has been "
         "replaced by <HOLDING>. Which holding statement is the one the citing court gave for that case?")


def post(path, body):
    req = urllib.request.Request(f"https://openrouter.ai/api/{path}", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def options_for(item, cond):
    opts = item["options"] if cond == "original" else item["abstain"]["options"]
    gold = item["gold"] if cond == "original" else item["abstain"]["gold"]
    crit = {LETTERS[i]: o for i, o in enumerate(opts)}
    if cond == "abstain":
        crit["NONE"] = NONE_DESC
    gold = gold if gold == "NONE" else LETTERS[gold]
    return crit, gold


def ask_decision(model, item, cond):
    crit, gold = options_for(item, cond)
    d = post("alpha/decisions", {"model": model, "state": {"citing_passage": item["prompt"]},
                                 "questions": {"holding": {"type": "choice", "instructions": INSTR,
                                                           "criteria": crit}}})
    a = d["answers"]["holding"]
    return {"gold": gold, "pred": a.get("choice"), "probs": a.get("probabilities"),
            "confidence": a.get("confidence"), "cost": d.get("usage", {}).get("cost"), "raw": d}


def ask_chat(model, item, cond):
    crit, gold = options_for(item, cond)
    opts = "\n".join(f"{k}. {v}" for k, v in crit.items())
    msg = (f"{INSTR}\n\nCiting passage:\n{item['prompt']}\n\nOptions:\n{opts}\n\n"
           'Reply with only JSON: {"answer": "<option key>", "confidence": <0-100, your probability '
           'that the answer is correct>}')
    d = post("v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": msg}],
                                     "temperature": 0, "usage": {"include": True}})
    text = d["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", text, re.S)
    pred, conf = None, None
    if m:
        try:
            j = json.loads(m.group(0))
            pred, conf = str(j.get("answer", "")).strip().upper(), j.get("confidence")
            conf = float(conf) / 100 if conf is not None else None
        except (ValueError, TypeError):
            pass
    return {"gold": gold, "pred": pred, "probs": None, "confidence": conf,
            "cost": d.get("usage", {}).get("cost"), "raw": {"text": text, "usage": d.get("usage")}}


def letter_probs(choice, keys):
    """Probability per option key from the logprobs of the first answer token, if returned."""
    content = (choice.get("logprobs") or {}).get("content") or []
    for tok in content:
        if tok["token"].strip().upper().rstrip(".") in keys:
            probs = {k: 0.0 for k in keys}
            for alt in tok.get("top_logprobs") or []:
                k = alt["token"].strip().upper().rstrip(".")
                if k in probs:
                    probs[k] += math.exp(alt["logprob"])
            z = sum(probs.values())
            return {k: round(v / z, 4) for k, v in probs.items()} if z else None
    return None


def answer_body(model, item, cond):
    """Chat-completions body for the answer-only prompt; shared by sync runs and batch.py.
    Structured output forces a bare {"answer": key}; some models otherwise write an explanation."""
    crit, gold = options_for(item, cond)
    opts = "\n".join(f"{k}. {v}" for k, v in crit.items())
    msg = (f"{INSTR}\n\nCiting passage:\n{item['prompt']}\n\nOptions:\n{opts}\n\n"
           f"Reply with only the option key ({', '.join(crit)}).")
    schema = {"type": "object", "additionalProperties": False, "required": ["answer"],
              "properties": {"answer": {"type": "string", "enum": list(crit)}}}
    effort = REASONING_OVERRIDES.get(model, REASONING_EFFORT)
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "temperature": 0,
            **({"reasoning": {"effort": effort}} if effort else {}), "usage": {"include": True},
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "holding_choice", "strict": True, "schema": schema}}}
    return body, crit, gold


def parse_answer(text):
    m = re.findall(r"\b(NONE|[A-E])\b", (text or "").upper())
    return m[-1] if m else None


def ask_answer(model, item, cond):
    """Answer-only prompt (how the tools are actually used): no elicited confidence."""
    body, crit, gold = answer_body(model, item, cond)
    try:
        d = post("v1/chat/completions", {**body, "logprobs": True, "top_logprobs": 10})
    except urllib.error.HTTPError as e:
        if e.code not in (400, 404):
            raise
        d = post("v1/chat/completions", body)  # model or its providers reject logprobs
    ch = d["choices"][0]
    text = ch["message"]["content"] or ""
    pred = parse_answer(text)
    probs = letter_probs(ch, set(crit))
    return {"gold": gold, "pred": pred, "probs": probs,
            "confidence": probs.get(pred) if probs and pred in probs else None,
            "cost": d.get("usage", {}).get("cost"), "raw": {"text": text, "usage": d.get("usage")}}


def main(model, cond, limit=None):
    items = [json.loads(l) for l in open(HERE / "data" / ("sample.jsonl" if SET == "pilot" else f"{SET}.jsonl"))][:limit]
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"{model.replace('/', '_').replace(':', '-')}__{cond}.jsonl"
    done = {json.loads(l)["id"] for l in open(log)} if log.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    fn = ask_chat if model.startswith("chat:") else ask_answer if model.startswith("ans:") else ask_decision
    name = model.split(":", 1)[-1] if model.startswith(("chat:", "ans:")) else model

    def one(it):
        try:
            return {"id": it["id"], **fn(name, it, cond)}
        except Exception as e:  # keep going; failed items are retried on the next run
            return {"id": it["id"], "error": repr(e)[:300]}

    ok = err = 0
    with cf.ThreadPoolExecutor(8) as ex, open(log, "a") as f:
        for r in ex.map(one, todo):
            if "error" in r:
                err += 1
                print("ERR", r["id"], r["error"], flush=True)
                continue
            ok += 1
            f.write(json.dumps(r) + "\n")
    print(f"{model} {cond}: wrote {ok}, errors {err}, skipped {len(done)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
