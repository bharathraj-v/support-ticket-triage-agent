"""One command: python -m app.run [--reset]"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .agent import Agent
from .config import load_settings
from .faq import load_helpcentre
from .helpdesk import Helpdesk
from .llm import Llm
from .state import State


def build_agent(verify_replies: bool = True) -> tuple[Agent, Llm, Helpdesk, State]:
    s = load_settings()
    hc = load_helpcentre(s.faq_path)
    hd = Helpdesk(s.helpdesk_url, s.helpdesk_key, s.candidate_email)
    llm = Llm(s.gemini_api_key, s.gemini_model, hc, cache_path=s.llm_cache_path,
              min_interval=s.llm_min_interval)
    state = State(s.state_path)
    return Agent(hd, llm, hc, state, s.log_path, s.confidence_threshold, verify_replies, progress=True), llm, hd, state


def main() -> None:
    p = argparse.ArgumentParser(description="Triage open support tickets.")
    p.add_argument("--reset", action="store_true", help="re-open all practice tickets first and clear local state")
    p.add_argument("--dry-run", action="store_true", help="decide for all open tickets but post nothing; "
                   "writes dry_run.json for review")
    p.add_argument("--no-verify", action="store_true", help="skip the second-pass reply check")
    args = p.parse_args()
    agent, llm, hd, state = build_agent(not args.no_verify)
    if args.dry_run:
        rows = []
        for t in tqdm(list(hd.iter_tickets("open")), desc="dry-run", unit="ticket"):
            outcome, meta = agent.decide(t)
            rows.append({"id": t["id"], "subject": t["subject"], "body": t["body"], "action": outcome.action,
                         "payload": outcome.payload, "downgrades": outcome.downgrades, "faq_ids": outcome.faq_ids,
                         "rule_flags": meta.get("rule_flags"), "check_notes": outcome.notes})
        out = Path("dry_run.json")
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"wrote {out} ({len(rows)} tickets); nothing was posted; usage={llm.usage.__dict__}")
        return
    if args.reset:
        hd.reset()
        state.clear()
    summary = agent.run()
    print(json.dumps({"summary": summary.__dict__,
                      "llm_usage": llm.usage.__dict__, "helpdesk": hd.status()}, indent=2))


if __name__ == "__main__":
    main()
