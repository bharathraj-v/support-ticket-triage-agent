"""Score the agent's decisions against the human-labelled examples from GET /examples.

Nothing is posted. Labels come from the helpdesk (written by humans), never from the model under test.
Usage: python -m eval.run_eval --tag before
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from app.agent import Agent
from app.config import load_settings
from app.faq import load_helpcentre
from app.helpdesk import Helpdesk
from app.llm import Llm
from app.state import State

HERE = Path(__file__).parent
FIELDS = ["language", "category", "priority", "action"]


def load_examples(hd: Helpdesk, refresh: bool = False) -> list[dict]:
    cache = HERE / "examples.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    examples = hd.examples()
    cache.write_text(json.dumps(examples, indent=1, ensure_ascii=False))
    return examples


def score(rows: list[dict]) -> dict:
    n = len(rows)
    acc = {f: sum(r["pred"][f] == r["label"][f] for r in rows) / n for f in FIELDS}
    must_handoff = [r for r in rows if r["label"]["action"] == "handoff"]
    must_not_handoff = [r for r in rows if r["label"]["action"] != "handoff"]
    return {
        "n": n,
        "accuracy": acc,
        # safety: a ticket humans send to a human must never get an automatic reply
        "handoff_recall": (sum(r["pred"]["action"] == "handoff" for r in must_handoff) / len(must_handoff)
                           if must_handoff else None),
        "unsafe_auto_replies": [r["id"] for r in must_handoff if r["pred"]["action"] == "reply"],
        # usefulness: tickets the help centre can answer should not all be escalated
        "over_handoffs": [r["id"] for r in must_not_handoff if r["pred"]["action"] == "handoff"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run", help="results saved to eval/results_<tag>.json")
    p.add_argument("--refresh", action="store_true", help="re-download the examples")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    s = load_settings()
    hc = load_helpcentre(s.faq_path)
    hd = Helpdesk(s.helpdesk_url, s.helpdesk_key, s.candidate_email)
    llm = Llm(s.gemini_api_key, s.gemini_model, hc, cache_path=s.llm_cache_path)
    agent = Agent(hd, llm, hc, State(HERE / ".eval_state.json"), HERE / ".eval_log.jsonl",
                  s.confidence_threshold, not args.no_verify)

    rows = []
    bar = tqdm(load_examples(hd, args.refresh), desc="eval", unit="ticket")
    for ex in bar:
        bar.set_postfix(id=ex["id"], llm_calls=llm.usage.calls, cached=llm.usage.cache_hits)
        outcome, meta = agent.decide(ex)
        pred = {k: outcome.payload[k] for k in ("language", "category", "priority")}
        pred["action"] = outcome.action
        rows.append({"id": ex["id"], "label": {k: ex["label"][k] for k in FIELDS}, "pred": pred,
                     "downgrades": outcome.downgrades, "check_notes": outcome.notes, "model_draft": outcome.payload.get("draft_reply"), "reason": outcome.payload["reason"],
                     "draft_reply": outcome.payload.get("draft_reply")})
    result = {"model": s.gemini_model, "threshold": s.confidence_threshold, "verify_replies": not args.no_verify,
              "score": score(rows), "usage": llm.usage.__dict__, "rows": rows}
    out = HERE / f"results_{args.tag}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))

    sc = result["score"]
    print(f"model={s.gemini_model} n={sc['n']}")
    for f in FIELDS:
        print(f"  {f:<9} {sc['accuracy'][f]:.0%}")
    print(f"  handoff recall {sc['handoff_recall']:.0%}   unsafe auto-replies: {sc['unsafe_auto_replies']}")
    print(f"  over-handoffs: {sc['over_handoffs']}")
    for r in rows:
        bad = [f for f in FIELDS if r["pred"][f] != r["label"][f]]
        if bad:
            print(f"  MISS {r['id']}: " + ", ".join(f"{f} {r['label'][f]}->{r['pred'][f]}" for f in bad))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
