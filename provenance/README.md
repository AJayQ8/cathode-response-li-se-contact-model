# Provenance map

`NUMERICAL_SOURCES.csv` maps current manuscript result groups to the released tables and selected saved evidence. It is intentionally limited to scientific data provenance and omits internal review/workflow ledgers.

The data tables and figures are current v04 saved deliverables. Evidence bundle entries are copied byte-for-byte from the frozen numerical archive only after validating each member against its `EVIDENCE_CONTENTS.json` hash index. Every bundle contains its own member hashes and records the frozen archive SHA-256. Two bundled operational manifests have only machine-local receipt/log path keys removed; their original source hash and sanitized export hash are both retained in the bundle manifest. `source/SOURCE_MANIFEST.csv` provides the same trace for included code snapshots.

Commit references distinguish the source calculation history from document packaging: Route A scientific source `e2015f4118b50aff0d94cfafd2d08b797eba1da9`; final saved read-off `8795b0eed` (as recorded in the evidence archive); challenge source `bec2881f69a5728ea637effe64560e474df5f77f`; and current v04 document source worktree commit `7c8b1d6e48030958ff93917e9e21447ebeedf7e3`. Historical v1.0.0 tag provenance is in `MANIFEST.md`.
