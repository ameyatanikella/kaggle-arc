"""Evidence ledger for executable world-model components.

A Component is an executable hypothesis with a version, unknown parameters, the quantities it
explains, and an applicability condition over observation contexts. Supported kinds:
  * 'linear'     y = row(ctx) . theta over a field ('q' or 'gf2')
  * 'polynomial' y = poly(ctx)(theta) over Q (a Poly in the parameters)
  * 'unsupported' anything else: only ordinary prediction checking applies (predict(ctx, theta)).
Rank is always derived from THIS map (design rows / Jacobian), never from arbitrary features.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Optional


@dataclass
class Observation:
    id: str
    quantity: str
    value: Any
    context: dict


@dataclass
class Component:
    name: str
    version: int
    kind: str                                   # 'linear' | 'polynomial' | 'unsupported'
    params: list
    explains: Callable[[Observation], bool]     # which quantities this component predicts
    applies: Callable[[Observation], tuple]     # -> (ok: bool, reason: str)
    field: str = "q"                            # 'q' | 'gf2' (linear only)
    row: Optional[Callable[[dict], list]] = None
    poly: Optional[Callable[[dict], Any]] = None
    predict: Optional[Callable[[dict, list], Any]] = None
    design_key: Optional[Callable[[dict], Any]] = None   # identical keys -> identical structure
    context_signature: Optional[Callable[[list], Any]] = None
    symmetries: list = dc_field(default_factory=list)       # [(name, theta -> theta')] as Poly maps
    catalog: list = dc_field(default_factory=list)          # candidate contexts for experiments
    applicability_text: str = ""

    def assumptions(self, ctx_sig) -> list:
        return [("version", self.name), ("context", self.name, ctx_sig)]


@dataclass
class LedgerEntry:
    component: str
    version: int
    kind: str
    params: list
    applicability: str
    eligible: list = dc_field(default_factory=list)
    excluded: list = dc_field(default_factory=list)          # (obs id, reason)
    supporting: list = dc_field(default_factory=list)
    contradicting: list = dc_field(default_factory=list)
    rank: Optional[int] = None
    n_params: int = 0
    independent: list = dc_field(default_factory=list)       # a basis of observation constraints
    relations: list = dc_field(default_factory=list)         # {obs, relation, value, satisfied}
    circuits_complete: Optional[bool] = None
    unverifiable: list = dc_field(default_factory=list)      # obs in no circuit: cannot be checked
    identifiability: dict = dc_field(default_factory=dict)
    ambiguity: dict = dc_field(default_factory=dict)         # unidentified dirs, discrete, identity
    verdict: str = "unchecked"
    diagnoses: list = dc_field(default_factory=list)
    suspects: list = dc_field(default_factory=list)
    suggestions: list = dc_field(default_factory=list)
    fallback: bool = False
    costs: dict = dc_field(default_factory=dict)
