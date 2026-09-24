"""The scikit-learn parameter protocol, shared by metrics and mitigators.

Both component families follow scikit-learn's estimator conventions:

* ``__init__`` takes **configuration only**, stores each argument verbatim under
  its own name, and performs no validation or computation. This is what makes
  :meth:`~ParameterizedComponent.get_params` /
  :meth:`~ParameterizedComponent.set_params` - and therefore cloning and
  parameter sweeps - possible.
* **Data** is passed to the compute entry point, never to ``__init__``.
* :meth:`~ParameterizedComponent.get_params` introspects ``__init__``, so
  subclasses get parameter introspection for free just by declaring named
  arguments.

Lives in the definitions core because :mod:`fairLMs.definitions` and
:mod:`fairLMs.mitigation` both build on it while definitions stays independent
of the mitigation layer.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List

__all__ = ["ParameterizedComponent"]


class ParameterizedComponent:
    """Mixin providing sklearn-style parameter introspection and ``repr``."""

    @classmethod
    def _param_names(cls) -> List[str]:
        """Names of this component's configuration parameters, from ``__init__``."""
        init = cls.__init__
        if init is ParameterizedComponent.__init__ or init is object.__init__:
            return []
        params = inspect.signature(init).parameters.values()
        names = [
            p.name
            for p in params
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        return sorted(names)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return this component's configuration as a dict.

        Mirrors ``sklearn.base.BaseEstimator.get_params``, so
        ``type(c)(**c.get_params())`` reconstructs an equivalent component and
        ``sklearn.base.clone`` works on fairLMs objects.

        Parameters
        ----------
        deep:
            When ``True``, also report the parameters of any nested object that
            exposes ``get_params``, under ``name__subname`` keys, which is
            sklearn's convention. Accepting this argument is what lets sklearn
            utilities (which always pass ``deep``) operate on these objects.
        """
        params: Dict[str, Any] = {}
        for name in self._param_names():
            value = getattr(self, name)
            params[name] = value
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for sub_name, sub_value in value.get_params(deep=True).items():
                    params[f"{name}__{sub_name}"] = sub_value
        return params

    def set_params(self, **params: Any) -> "ParameterizedComponent":
        """Set configuration parameters in place and return ``self``."""
        valid = self._param_names()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter {key!r} for {type(self).__name__}. "
                    f"Valid parameters are: {', '.join(valid) or '(none)'}."
                )
            setattr(self, key, value)
        return self

    def _reject_unknown_kwargs(self, kwargs: Dict[str, Any], *allowed: str) -> None:
        """Raise ``TypeError`` for kwargs this component does not understand.

        ``**kwargs`` signatures silently swallow typos, so a misspelled
        ``n_bootstrp=`` quietly uses the default instead of failing. Components
        call this to make unknown keys an error, as sklearn does.
        """
        unknown = sorted(set(kwargs) - set(allowed))
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.{self._compute_name}() got unexpected "
                f"keyword argument(s): {', '.join(unknown)}. "
                f"Accepted: {', '.join(sorted(allowed)) or '(none)'}."
            )

    #: Named in the unknown-keyword message so it points at the real entry point.
    _compute_name = "compute"

    def __repr__(self) -> str:
        params = self.get_params(deep=False)
        defaults = {}
        if params:
            sig = inspect.signature(type(self).__init__).parameters
            defaults = {
                k: v.default
                for k, v in sig.items()
                if v.default is not inspect.Parameter.empty
            }
        shown = [
            f"{k}={v!r}"
            for k, v in sorted(params.items())
            if k not in defaults or defaults[k] != v
        ]
        return f"{type(self).__name__}({', '.join(shown)})"
