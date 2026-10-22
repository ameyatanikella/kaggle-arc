"""Prospective evidence and probe selection for competing causal models.

The bank deliberately does not parse pixels or execute actions. A perception layer
supplies named, observable features; candidate models predict only the features
they claim to know. Predictions are recorded *before* the action result arrives.
This lets a later controller use the same bank with hand-written programs, model-
proposed programs, or learned proposal policies without treating a retrospective
fit as a successful prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Hashable, Mapping, Protocol, Sequence


Observation = Mapping[str, Hashable]
Action = Hashable


@dataclass(frozen=True)
class Transition:
    before: Observation
    action: Action
    after: Observation


class CausalModel(Protocol):
    name: str

    def predict(
        self, before: Observation, action: Action, history: tuple[Transition, ...]
    ) -> Observation | None:
        """Return claimed next features; omit fields whose outcome is unknown."""


@dataclass
class Evidence:
    # Correct and wrong count individual *informative fields*, not whole steps.
    correct: int = 0
    wrong: int = 0
    vacuous: int = 0
    abstained: int = 0
    errors: int = 0
    event_opportunities: int = 0
    event_claims: int = 0

    @property
    def checked(self) -> int:
        return self.correct + self.wrong

    @property
    def event_coverage(self) -> float | None:
        """Share of changed observable fields the model claimed prospectively."""
        if self.event_opportunities == 0:
            return None
        return self.event_claims / self.event_opportunities


@dataclass(frozen=True)
class ProbeChoice:
    action: Action
    disagreement: float
    cost: float
    value: float


class CausalBeliefBank:
    """Keep partial models alive and score their next-action predictions.

    Weights use one likelihood increment per informative field against a
    50%-accurate binary reference, multiplied by observed event coverage.
    Correct no-change claims receive no credit, while a missed change is a
    wrong claim. They rank models, but are not calibrated probabilities:
    fields can be correlated and models may omit important effects. Inspect
    event_coverage alongside weights before trusting a model for control.
    A mismatch lowers a model's weight without deleting it, since the candidate
    language may only explain part of a real game.
    """

    def __init__(self, models: Sequence[CausalModel], priors: Mapping[str, float] | None = None):
        if not models:
            raise ValueError("at least one causal model is required")
        names = [model.name for model in models]
        if len(names) != len(set(names)):
            raise ValueError("causal model names must be unique")
        self.models = tuple(models)
        self.history: tuple[Transition, ...] = ()
        self.evidence = {name: Evidence() for name in names}
        self.priors = {name: float((priors or {}).get(name, 1.0)) for name in names}
        if any(not isfinite(value) or value <= 0 for value in self.priors.values()):
            raise ValueError("prior weights must be finite and positive")
        self._pending: tuple[Observation, Action, dict[str, Observation | None], set[str], frozenset[str] | None] | None = None

    def weights(self) -> dict[str, float]:
        """Relative weights from *prospective* checks only."""
        logits = {
            name: log(self.priors[name])
            + record.correct * log(0.85 / 0.5)
            + record.wrong * log(0.15 / 0.5)
            - 2.0 * record.errors
            + log((record.event_claims + 1) / (record.event_opportunities + 1))
            for name, record in self.evidence.items()
        }
        peak = max(logits.values())
        # Keep severely refuted models available for later regime changes or
        # corrections to a misspecified candidate language.
        raw = {name: max(exp(value - peak), 1e-12) for name, value in logits.items()}
        total = sum(raw.values())
        return {name: value / total for name, value in raw.items()}

    def _forecast(self, before: Observation, action: Action) -> tuple[dict[str, Observation | None], set[str]]:
        predictions: dict[str, Observation | None] = {}
        errors: set[str] = set()
        for model in self.models:
            try:
                claim = model.predict(before, action, self.history)
                predictions[model.name] = dict(claim) if claim else None
            except Exception:
                predictions[model.name] = None
                errors.add(model.name)
        return predictions, errors

    def forecast(self, before: Observation, action: Action) -> dict[str, Observation | None]:
        """Read candidate predictions without adding evidence or changing history."""
        claims, _ = self._forecast(before, action)
        return claims

    def choose_probe(
        self,
        before: Observation,
        actions: Sequence[Action],
        relevant_fields: Mapping[str, float] | None = None,
        action_costs: Mapping[Action, float] | None = None,
        cost_weight: float = 0.0,
    ) -> ProbeChoice | None:
        """Choose an action with high relevant disagreement after action cost.

        This is an information-seeking heuristic. It does not assume an action is
        safe or reversible; the caller must filter candidates for those concerns.
        Returning None means no action has positive net heuristic value; this
        includes cases without conflicting claims on relevant fields.
        """
        if cost_weight < 0:
            raise ValueError("cost_weight must be nonnegative")
        if not isfinite(cost_weight):
            raise ValueError("cost_weight must be finite")
        if relevant_fields is not None and any(not isfinite(v) or v < 0 for v in relevant_fields.values()):
            raise ValueError("field weights must be finite and nonnegative")
        if action_costs is not None and any(not isfinite(v) or v < 0 for v in action_costs.values()):
            raise ValueError("action costs must be finite and nonnegative")
        weights = self.weights()
        best: ProbeChoice | None = None
        for action in actions:
            claims, _ = self._forecast(before, action)
            disagreement = 0.0
            for i, left in enumerate(self.models):
                a = claims[left.name]
                if a is None:
                    continue
                for right in self.models[i + 1 :]:
                    b = claims[right.name]
                    if b is None:
                        continue
                    common = a.keys() & b.keys()
                    for field in common:
                        importance = 1.0 if relevant_fields is None else relevant_fields.get(field, 0.0)
                        if importance and a[field] != b[field]:
                            disagreement += weights[left.name] * weights[right.name] * importance
            if disagreement <= 0:
                continue
            cost = float((action_costs or {}).get(action, 1.0))
            choice = ProbeChoice(action, disagreement, cost, disagreement - cost_weight * cost)
            if choice.value > 0 and (best is None or choice.value > best.value):
                best = choice
        return best

    def commit(self, before: Observation, action: Action,
               scored_fields: Sequence[str] | None = None) -> None:
        """Save predictions and decision-relevant scoring fields before acting.

        If scored_fields is omitted, all observable fields count. The caller
        should provide a fixed relevant field set when incidental HUD or other
        nuisance changes would distort model selection.
        """
        if self._pending is not None:
            raise RuntimeError("observe the pending action before committing another")
        frozen_before = dict(before)
        claims, errors = self._forecast(frozen_before, action)
        self._pending = (frozen_before, action, claims, errors,
                         frozenset(scored_fields) if scored_fields is not None else None)

    def observe(self, after: Observation) -> Transition:
        """Grade saved claims per field and append the observed transition.

        A correct prediction that a field stays as it was is vacuous: it cannot
        build trust on idle steps. Predicting no change when it changes is wrong.
        """
        if self._pending is None:
            raise RuntimeError("commit an action before observing its result")
        before, action, claims, errors, scored_fields = self._pending
        frozen_after = dict(after)
        # A disappeared object is represented by a missing key or explicit
        # None. Treat those identically so deletion is a gradeable effect.
        changed_fields = {
            field for field in before.keys() | frozen_after.keys()
            if (scored_fields is None or field in scored_fields)
            and before.get(field) != frozen_after.get(field)
        }
        for name, record in self.evidence.items():
            record.event_opportunities += len(changed_fields)
            if name in errors:
                record.errors += 1
            elif claims[name] is None or not any(
                scored_fields is None or field in scored_fields for field in claims[name]
            ):
                record.abstained += 1
            else:
                claim = claims[name]
                record.event_claims += len(changed_fields & claim.keys())
                for field, value in claim.items():
                    if scored_fields is not None and field not in scored_fields:
                        continue
                    if frozen_after.get(field) != value:
                        record.wrong += 1
                    elif before.get(field) == value:
                        record.vacuous += 1
                    else:
                        record.correct += 1
        transition = Transition(before, action, frozen_after)
        self.history += (transition,)
        self._pending = None
        return transition
