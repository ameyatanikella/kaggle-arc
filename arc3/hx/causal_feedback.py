"""Non-blocking algebra/neutral prompt pair for the Part A comparison.

Only observed same-level transitions enter this report. In particular, a
candidate click outside the observed GF(2) row space is never called known to
be novel until its effect has actually been observed.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .algebra import COMMUTE, CYCLE, IDEM, INVERSE, INVOL, NOOP, OVERRIDE

RELATION_NAME = {NOOP: "noop", IDEM: "idem", INVOL: "invol",
                 INVERSE: "inverse", COMMUTE: "commute", OVERRIDE: "override"}
DISPLAY_NAME = {"idem": "idempotent", "invol": "involution", "commute": "commutes"}
WIDTH = 560


def _rank(vectors):
    pivots = {}
    for row in vectors:
        x = row
        while x:
            p = x.bit_length() - 1
            if p in pivots:
                x ^= pivots[p]
            else:
                pivots[p] = x
                break
    return len(pivots)


def _action(a):
    if a[0] == 6:
        return f"click({a[1]},{a[2]})"
    return f"A{a[0]}"


def _concrete(word):
    out = []
    for token in word:
        if isinstance(token, tuple) and token and isinstance(token[0], tuple):
            out.append(tuple(token[0]))
        elif isinstance(token, tuple) and token and isinstance(token[0], int):
            out.append(tuple(token))
        else:
            return None
    return tuple(out)


def evidence(session):
    """A compact, serializable report with concrete actions and local support."""
    level = session.log.level
    transitions = list(session.log.transitions(level=level))
    counts = Counter()
    click_vectors = []
    for pre, action, post, _ in transitions:
        f0, f1 = session.log.grid(pre), session.log.grid(post)
        if f0 is None or f1 is None:
            continue
        counts["actions"] += 1
        changed = (np.asarray(f0) != np.asarray(f1)) & ~session.vm.mask
        counts["changed"] += int(changed.any())
        if action[0] == 6:
            counts["clicks"] += 1
            click_vectors.append(int.from_bytes(np.packbits(changed).tobytes(), "big"))
    grouped = {}
    alg = session.alg if session.alg_level == level else None
    if alg is not None:
        for row in alg.relations(min_support=1, limit=300):
            if row["level"] != 2:
                continue
            relation = row["rel"]
            order = None
            if relation.startswith(CYCLE + "|"):
                order = int(relation.split("|")[1])
                if not 3 <= order <= 6:
                    continue
                kind = "order"
            elif relation in RELATION_NAME:
                kind = RELATION_NAME[relation]
            else:
                continue
            word = _concrete(row["word"])
            if word is None:
                continue
            key = (kind, word, order)
            item = grouped.setdefault(key, {"kind": kind, "actions": word, "order": order,
                                            "support": 0, "counter": 0})
            item["support"] += int(row["support"])
            item["counter"] += int(row["counter"])
    laws = list(grouped.values())
    laws.sort(key=lambda r: (r["counter"] != 0, -r["support"], r["kind"], r["actions"]))
    legal = session.log.steps[-1].legal if session.log.steps else ()
    tried = {a[0] for _, a, _, _ in transitions}
    untried_keys = sorted(a for a in legal if a not in tried and a not in (0, 6))
    return {"level": level + 1, "recorded_steps": len(session.log.steps),
            "actions": counts["actions"], "changed": counts["changed"],
            "clicks": counts["clicks"], "click_rank": _rank(click_vectors),
            "untried_keys": untried_keys, "laws": laws[:24]}


def render(session, neutral=False):
    """Return (bounded prompt text, machine-readable facts); no action gate."""
    facts = evidence(session)
    if neutral:
        body = (f"Progress summary, level {facts['level']}: {facts['actions']} actions observed; "
                f"{facts['changed']} changed the visible board. "
                f"{facts['clicks']} clicks were tried. Continue testing your current game plan "
                "and compare each result with what you expected. Record the board after each "
                "action and revise your own explanation when observations differ. "
                "This is a timing and progress reminder only; it contains no causal claim.")
    else:
        supported = [r for r in facts["laws"] if r["support"] and not r["counter"]][:2]
        contradicted = [r for r in facts["laws"] if r["counter"]][:1]

        def desc(row):
            word = " then ".join(_action(a) for a in row["actions"])
            name = f"order {row['order']}" if row["order"] is not None else \
                DISPLAY_NAME.get(row["kind"], row["kind"])
            return f"{word}: {name} (+{row['support']}/-{row['counter']})"

        def causal_body():
            return (f"Causal evidence, level {facts['level']}: "
                    f"supported locally [{'; '.join(map(desc, supported)) or 'none yet'}]; "
                    f"contradicted [{'; '.join(map(desc, contradicted)) or 'none yet'}]. "
                    f"Observed clicks: GF(2) effect rank {facts['click_rank']} from "
                    f"{facts['clicks']} clicks; unseen click effects are unknown. "
                    f"Untried keys: {','.join(map(str, facts['untried_keys'])) or 'none'}. "
                    "These relations apply only in tested contexts; test an uncertain action if it matters.")

        body = causal_body()
        while len(body) > WIDTH and (contradicted or supported):
            (contradicted if contradicted else supported).pop()
            body = causal_body()
        facts["shown_laws"] = supported + contradicted
    # Same character budget and line count in both arms. Proprietary MiniMax
    # tokenization is not available locally, so exact token parity is unknown.
    if len(body) > WIDTH:
        body = body[:WIDTH - 1] + "…"
    return body.ljust(WIDTH), facts if not neutral else None
