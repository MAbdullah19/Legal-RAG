"""Jurisdiction packs. Each pack supplies connectors, parsers, citation schemes,
and a court registry that populate the jurisdiction-neutral canonical model
(master plan §4). Core never imports from here — the dependency arrow points one
way (core <- packs). The shared JurisdictionPack abstraction is extracted on the
multi-jurisdiction track (P5) once US and PK both work.
"""
