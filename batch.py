"""Run the answer-only prompt through OpenRouter's Batch API (~50% of sync price, async, <=24h; Google up to 96h).

Usage:
  python batch.py submit MODEL COND [limit] [offset]   e.g. submit google/gemini-4-argon abstain 50
  python batch.py status                               show every saved batch
  python batch.py collect                              write finished batches to the logs
Items listed for a model in data/moderation_excluded.json are left out of its batches (reported as refused).

Requests are built by run.answer_body (identical to sync runs, minus logprobs); results land in
the sync log file for "ans:MODEL", so analyze.py / combine.py treat batch and sync rows the same.
Batch ids are saved in batches/{SET}__{model}__{cond}__{offset}-{end}.json.
"""
import json
import pathlib
import sys
import urllib.error
import urllib.request

import run

BATCHES = run.HERE / "batches"


def api(method, path, body=None):
    req = urllib.request.Request(f"https://openrouter.ai/api/v1/{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {run.KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def items_for(limit=None, offset=0):
    name = "sample.jsonl" if run.SET == "pilot" else f"{run.SET}.jsonl"
    items = [json.loads(l) for l in open(run.HERE / "data" / name)]
    return items[offset:offset + limit if limit else None]


def excluded(model):
    """Item ids a provider's moderation refused for this model (data/moderation_excluded.json)."""
    f = run.HERE / "data" / "moderation_excluded.json"
    return set(json.loads(f.read_text()).get(model, [])) if f.exists() else set()


def submit(model, cond, limit=None, offset=0):
    skip = excluded(model)
    items = [it for it in items_for(limit, offset) if str(it["id"]) not in skip]
    reqs = []
    for it in items:
        body, _, _ = run.answer_body(model, it, cond)
        body.pop("model")  # inherited from the batch-level model
        body.pop("usage", None)
        reqs.append({"custom_id": str(it["id"]), "body": body})
    # endpoint/model must be serialized before requests (the API stream-parses the body)
    b = api("POST", "batches", {"endpoint": "/v1/chat/completions", "model": model, "requests": reqs})
    BATCHES.mkdir(exist_ok=True)
    tag = f"{run.SET}__{model.replace('/', '_')}__{cond}__{offset}-{offset + len(items)}"
    (BATCHES / f"{tag}.json").write_text(json.dumps(
        {"batch_id": b["id"], "model": model, "cond": cond, "set": run.SET, "n": len(items),
         "provider": b.get("provider"), "expires_at": b.get("expires_at"), "collected": False}, indent=1))
    print(f"submitted {b['id']} {model} {cond} n={len(items)} provider={b.get('provider')} status={b['status']}")


def manifests():
    return sorted(BATCHES.glob("*.json")) if BATCHES.exists() else []


def status():
    for f in manifests():
        m = json.loads(f.read_text())
        try:
            b = api("GET", f"batches/{m['batch_id']}")
        except urllib.error.HTTPError as e:
            print(f"{f.stem:70s} HTTP {e.code} (just submitted? retry shortly)")
            continue
        rc = b.get("request_counts") or {}
        cost = (b.get("usage") or {}).get("cost")
        print(f"{f.stem:70s} {b['status']:11s} {rc.get('completed', 0)}/{rc.get('total', '?')} "
              f"failed={rc.get('failed', 0)} provider={b.get('provider')} cost={cost} "
              f"collected={m['collected']}{'  error=' + str(b['error'])[:120] if b.get('error') else ''}")


def collect():
    for f in manifests():
        m = json.loads(f.read_text())
        if m["collected"]:
            continue
        b = api("GET", f"batches/{m['batch_id']}")
        if b["status"] != "completed":
            print(f"{f.stem}: {b['status']}, skipping")
            continue
        if m["set"] != run.SET:
            print(f"{f.stem}: belongs to CASEHOLD_SET={m['set']}, skipping")
            continue
        items = {str(it["id"]): it for it in items_for()}
        results = b.get("results") or []
        done_ok = [r for r in results if not r.get("error") and (r.get("response") or {}).get("status_code") == 200]
        per_item = ((b.get("usage") or {}).get("cost") or 0) / max(len(done_ok), 1)
        run.LOGS.mkdir(exist_ok=True)
        log = run.LOGS / f"ans-{m['model'].replace('/', '_').replace(':', '-')}__{m['cond']}.jsonl"
        have = {json.loads(l)["id"] for l in open(log)} if log.exists() else set()
        wrote = failed = 0
        with open(log, "a") as out:
            for r in results:
                it = items.get(r["custom_id"])
                if it is None or it["id"] in have:
                    continue
                if r not in done_ok:
                    failed += 1
                    continue
                body = r["response"]["body"]
                text = body["choices"][0]["message"].get("content") or ""
                gold = run.options_for(it, m["cond"])[1]
                out.write(json.dumps({"id": it["id"], "gold": gold, "pred": run.parse_answer(text), "probs": None,
                                      "confidence": None, "cost": (body.get("usage") or {}).get("cost", per_item),
                                      "raw": {"text": text, "usage": body.get("usage"), "batch": m["batch_id"]}}) + "\n")
                wrote += 1
        m["collected"] = failed == 0
        f.write_text(json.dumps(m, indent=1))
        print(f"{f.stem}: wrote {wrote}, failed {failed}, batch cost ${(b.get('usage') or {}).get('cost')}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "submit":
        submit(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None,
               int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    elif cmd == "status":
        status()
    elif cmd == "collect":
        collect()
