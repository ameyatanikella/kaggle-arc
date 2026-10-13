"""HX: decision-relevant automata learning and contextual action algebra.

Shared by the CPU host (arc3/agents/hx) and the LLM host (kaggle/b2 B2_HX switch).
Pure numpy; no game-specific knowledge. See docs/hx/architecture.md.

Modules
  events    faithful typed event log (decision / animation / win / init / death / reset)
  objects   object parse with abstention and cross-frame matching
  algebra   contextual action relations (noop, idempotent, involution, cycle, inverse,
            commute, override, absorb) with support, counterexamples and context refinement
  goals     goal hypotheses from level transitions (no dynamics model)
  models    candidate executable models (state, T, O, G) with prospective weighting
  probes    decision-valued probe selection over a model set
  homing    hidden-state detection and short homing / distinguishing tests
  memory    cross-level store keeping mechanics and goals separate
"""
