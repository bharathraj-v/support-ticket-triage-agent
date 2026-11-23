"""One command: python -m app.run [--reset]"""
import argparse
import json

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
    llm = Llm(s.gemini_api_key, s.gemini_model, hc)
    state = State(s.state_path)
    return Agent(hd, llm, hc, state, s.log_path, s.confidence_threshold, verify_replies), llm, hd, state


def main() -> None:
    p = argparse.ArgumentParser(description="Triage open support tickets.")
    p.add_argument("--reset", action="store_true", help="re-open all practice tickets first and clear local state")
    p.add_argument("--no-verify", action="store_true", help="skip the second-pass reply check")
    args = p.parse_args()
    agent, llm, hd, state = build_agent(not args.no_verify)
    if args.reset:
        hd.reset()
        state.clear()
    summary = agent.run()
    print(json.dumps({"summary": summary.__dict__,
                      "llm_usage": llm.usage.__dict__, "helpdesk": hd.status()}, indent=2))


if __name__ == "__main__":
    main()
