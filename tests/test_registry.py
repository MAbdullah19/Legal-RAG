"""Registry + ComponentSpec behaviour."""

from __future__ import annotations

import pytest

from legalrag.core.registry import ComponentSpec, Registry, RegistryError


def test_spec_from_flat_mapping() -> None:
    spec = ComponentSpec.from_obj({"name": "rrf", "k": 60})
    assert spec.name == "rrf"
    assert spec.params == {"k": 60}


def test_spec_from_nested_params() -> None:
    spec = ComponentSpec.from_obj({"name": "rrf", "params": {"k": 40}, "extra": 1})
    assert spec.params == {"k": 40, "extra": 1}


def test_spec_from_string() -> None:
    assert ComponentSpec.from_obj("none").name == "none"


def test_spec_requires_name() -> None:
    with pytest.raises(ValueError):
        ComponentSpec.from_obj({"k": 60})


def test_register_build_and_params() -> None:
    reg = Registry()

    @reg.register("fusion", "rrf")
    class _RRF:
        def __init__(self, k: int = 60) -> None:
            self.k = k

    built = reg.build("fusion", {"name": "rrf", "k": 42})
    assert isinstance(built, _RRF)
    assert built.k == 42


def test_duplicate_registration_rejected() -> None:
    reg = Registry()

    @reg.register("fusion", "rrf")
    class _A:
        pass

    with pytest.raises(RegistryError):

        @reg.register("fusion", "rrf")
        class _B:
            pass


def test_unknown_component_lists_available() -> None:
    reg = Registry()

    @reg.register("fusion", "rrf")
    class _A:
        pass

    with pytest.raises(RegistryError) as exc:
        reg.get("fusion", "nope")
    assert "rrf" in str(exc.value)
