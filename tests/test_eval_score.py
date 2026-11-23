from eval.run_eval import score


def row(i, label_action, pred_action, **kw):
    lab = {"language": "nl", "category": "question", "priority": "P4", "action": label_action}
    pred = {**lab, "action": pred_action, **kw}
    return {"id": i, "label": lab, "pred": pred}


def test_score_flags_unsafe_and_over_handoff():
    rows = [row("A", "handoff", "reply"), row("B", "reply", "handoff"), row("C", "reply", "reply")]
    s = score(rows)
    assert s["unsafe_auto_replies"] == ["A"] and s["over_handoffs"] == ["B"]
    assert s["handoff_recall"] == 0 and abs(s["accuracy"]["action"] - 1 / 3) < 1e-9
