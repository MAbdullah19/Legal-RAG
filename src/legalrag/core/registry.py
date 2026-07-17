"""Component registry (ADR 0002).

Every pipeline stage implementation registers itself under ``(stage, name)``.
Experiment configs name components; :func:`build` instantiates them with their
params. Adding a technique never edits the pipeline runner — it adds a class.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ComponentSpec(BaseModel):
    """A component selection from config: a ``name`` plus arbitrary params.

    In YAML a spec is written flat, e.g. ``{name: rrf, k: 60}``; every key other
    than ``name`` becomes a constructor param.
    """

    model_config = ConfigDict(extra="allow")
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Any) -> ComponentSpec:
        if isinstance(obj, ComponentSpec):
            return obj
        if isinstance(obj, str):
            return cls(name=obj)
        if isinstance(obj, Mapping):
            data = dict(obj)
            name = data.pop("name", None)
            if not name:
                raise ValueError(f"component spec missing 'name': {obj!r}")
            # allow either nested {name, params:{...}} or flat {name, k: 60}
            params = dict(data.pop("params", {}))
            params.update(data)
            return cls(name=str(name), params=params)
        raise TypeError(f"cannot read component spec from {type(obj).__name__}")


class RegistryError(KeyError):
    pass


class Registry:
    def __init__(self) -> None:
        self._by_stage: dict[str, dict[str, Callable[..., Any]]] = {}

    def register(self, stage: str, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            bucket = self._by_stage.setdefault(stage, {})
            if name in bucket:
                raise RegistryError(f"{stage}:{name} already registered")
            bucket[name] = cls
            return cls

        return deco

    def get(self, stage: str, name: str) -> Callable[..., Any]:
        try:
            return self._by_stage[stage][name]
        except KeyError:
            avail = ", ".join(self.available(stage)) or "<none>"
            raise RegistryError(
                f"unknown {stage} component {name!r}; available: {avail}"
            ) from None

    def build(self, stage: str, spec: Any) -> Any:
        spec = ComponentSpec.from_obj(spec)
        factory = self.get(stage, spec.name)
        return factory(**spec.params)

    def available(self, stage: str) -> list[str]:
        return sorted(self._by_stage.get(stage, {}))

    def stages(self) -> list[str]:
        return sorted(self._by_stage)


REGISTRY = Registry()


def register(stage: str, name: str) -> Callable[[type[T]], type[T]]:
    """Module-level decorator: ``@register(Stage.FUSION, "rrf")``."""
    return REGISTRY.register(stage, name)


def build(stage: str, spec: Any) -> Any:
    return REGISTRY.build(stage, spec)
