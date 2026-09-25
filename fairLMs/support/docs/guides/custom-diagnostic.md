# Writing a diagnostic

A diagnostic subclasses `DatasetDiagnostic` and implements two methods: `plan`,
which decides applicability *before* any computation, and `compute`, which
either produces a value or returns the same non-ready state.

```python
from typing import Any

from fairLMs.datasets.diagnostics import (
    ComponentPlan,
    ComponentResult,
    DatasetAuditSpec,
    DatasetDiagnostic,
    DiagnosticStatus,
)


class MyComponent(DatasetDiagnostic):
    """One-line summary of what this component measures."""

    name = "my_component"

    def plan(self, evidence: Any, spec: DatasetAuditSpec) -> ComponentPlan:
        if spec.task_family != "scored_rows":
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="task_family_mismatch",
                reason="my_component describes row-level scores only.",
            )
        return ComponentPlan(component=self.name, status=DiagnosticStatus.READY)

    def compute(self, evidence: Any, spec: DatasetAuditSpec) -> ComponentResult:
        plan = self.plan(evidence, spec)
        if plan.status is not DiagnosticStatus.READY:
            return ComponentResult(
                component=self.name,
                status=plan.status,
                reason_code=plan.reason_code,
                reason=plan.reason,
            )
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=float(...),
            details={"unit": "score_units"},
            assumptions=["Scores are comparable across groups."],
            provenance={"evidence_source": evidence.source},
        )
```

## The invariant the base class enforces

`ComponentResult` validates this for you, and it is the whole point of the
layer:

- a `ready` component **must** carry a finite numeric `value`;
- any non-ready component **must** use `value=None`; passing a numeric
  sentinel raises `ValueError` with the message *"a blocked component must use
  value=None, not a numeric sentinel"*;
- any non-ready component **must** supply both `reason_code` and `reason`, so a
  refusal is always explained.

You cannot accidentally report unavailable evidence as `0.0`.

## Choosing a status

| Status | Use when |
|---|---|
| `READY` | inputs are sufficient and the value was computed |
| `BLOCKED` | a required input was not supplied (a missing reference, a missing rule) |
| `NOT_APPLICABLE` | the evidence is structurally incapable of supporting this component |
| `FAILED` | computation raised; an execution status, illegal in a `plan` |

`FAILED` is rejected by `ComponentPlan.__post_init__`: a plan describes
applicability, not outcomes.

## Registering

Add the class to `DIAGNOSTIC_REGISTRY` in
`fairLMs/datasets/diagnostics/registry.py`, keyed on its `name`, and export it
from `fairLMs/datasets/diagnostics/__init__.py`.
Registration enrols it in the diagnostics contract suite:

```bash
pytest support/tests/test_diagnostics_contract.py
```

## Running it

Diagnostics are executed by `audit_representativeness` or `audit_scores`, which
assemble the per-component results into a `DiagnosticReport` and derive the
report-level status. Bump `DIAGNOSTIC_SCHEMA_VERSION` if you change the
serialized shape.
