# Support ticket triage agent

Reads open tickets from a helpdesk API (customers mostly write Dutch) and posts **one decision per ticket**:
`reply` (draft in the customer's language, from the help centre only), `handoff` (English note for a human) or
`close` (spam/empty). It also sets language, category, priority and a one-line reason.

```
helpdesk ─► redact IBAN/card ─► safety rules ─► Gemini (structured JSON) ─► validators ─► reply checker ─► post
  open        (code)           (code, forces      decision + FAQ ids       (code)        2nd LLM pass:        │
  tickets                       handoff)                                                 back-translate,     ▼
                                                                                         one repair try   state + log
```
Anything doubtful becomes a handoff: unsure tickets cost a human a minute, a wrong reply reaches a customer.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env            # fill in HELPDESK_KEY, CANDIDATE_EMAIL, GEMINI_API_KEY, GEMINI_MODEL
pytest -q                       # 39 offline tests
python -m eval.run_eval --tag mine   # score the 10 labelled examples (posts nothing)
python -m app.run --reset       # re-open practice tickets, handle all open tickets once
uvicorn app.main:app            # optional: same agent over HTTP (POST /run, GET /status)
```

## Decisions
1. **Dutch: work in Dutch directly, verify by back-translation.** No translate-first step (saves a call and avoids
   translation loss). I can't read Dutch, so every draft goes to a second model pass that checks the language,
   back-translates to English and tests each claim against the cited FAQ entries. Cost: one extra call per reply.
   It earns its keep: it caught "Colleagues" inside a Dutch sentence.
2. **Rules in code, not just in the prompt.** Refunds, cancellations, GDPR/AVG, security, legal threats and churn
   threats (NL/EN/DE keywords, `app/rules.py`) force a handoff whatever the model says, and raise priority.
   A reply must also cite a real FAQ id, clear a confidence threshold (0.7) and contain no card/IBAN, else handoff.
3. **Eval-driven, small and explainable.** Labels come from the helpdesk's human-labelled examples; the model under
   test never writes labels. Every LLM call is saved to `state/llm_cache.jsonl`, so runs resume and re-runs are free.

Assumptions: "unsure" = confidence < 0.7, a validator failure, or no FAQ id cited. Forced handoffs are at least P1
(security/legal) or P2 (money/contract/GDPR/churn). "Customer" in the priority scale means the whole company.
Built with Claude Code: I directed the design and reviewed every change; the prompts and rules were tuned against eval output.

## Reliability
- **503**: retry with exponential backoff and jitter (6 tries). **429**: sleep `Retry-After`; a client-side limiter
  keeps us at 4 req/s. **409**: never retried; the ticket is marked done.
- **Crash / re-run**: the decision is written to `state/decisions.json` (atomic write) *before* posting, marked
  posted after. A re-run only fetches open tickets, and re-posts a stored decision rather than re-deciding, so no
  customer gets two answers. A model failure escalates that ticket to a human instead of stopping the run.
- **Security**: keys only from environment variables (`.env` is gitignored). Ticket text is untrusted: it sits inside
  `<ticket>` tags with an instruction to treat it as data, output is schema-validated, and rules and validators run
  in code after the model. Card numbers/IBANs are masked before the model, the logs and replies. One log line per ticket
  in `logs/run.jsonl` (ids, decision, confidence, latency; no ticket text).

## Eval and fix
10 human-labelled examples, `gemini-3.5-flash-lite`:

| | Baseline | Final |
|---|---|---|
| Action | 70% | **100%** |
| Category | 100% | 100% |
| Language | 100% | 100% |
| Priority | 90% | 100% |
| Handoff recall / unsafe auto-replies | 100% / 0 | 100% / 0 |
| Over-handoffs | 3 (E01, E05, E07) | 0 |

**The fix:** all 3 over-handoffs came from the reply checker rejecting correct answers, because it judged them against
the FAQ alone and flagged facts the customer had written themselves. Giving it the ticket took action accuracy to 90%
(`results_after.json`). Remaining misses were then fixed with a one-shot repair of rejected drafts, and prompt
guidance for category, priority and totals (`results_final.json`). One tuning step regressed E06 before I fixed it.
**Caveat:** I tuned on the same 10 examples I score on, so 100% is optimistic; 10 tickets is also very small.

**Cost per 100 tickets:** the final eval used 21,816 input + 1,977 output tokens for 10 tickets (18 calls, including
replies' checks and repairs) = 218,160 + 19,770 per 100. At $0.30 / $2.50 per 1M tokens
(third-party listings, not checked against Google's page): 0.218 × 0.30 + 0.0198 × 2.50 ≈ **$0.115 (about €0.11)**.

## What I would improve
Next: a held-out labelled set so tuning and scoring use different tickets, and a different model for the reply checker
so it doesn't share the drafter's blind spots. Also a rate-aware worker pool; the run is sequential (~10 s/ticket).
**One thing I'd change in the brief:** define the priority scale for a *company* versus a *single user* ("one customer
blocked" was ambiguous: it decided E01 and E06), and define "unsure" numerically instead of leaving it open.
