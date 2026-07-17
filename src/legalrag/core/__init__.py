"""Jurisdiction-neutral engine core.

Nothing under :mod:`legalrag.core` may import from ``legalrag.jurisdictions``:
the dependency arrow points one way (core <- jurisdiction packs). Anything two
packs need gets promoted here behind a Protocol in :mod:`legalrag.core.interfaces`.
"""
