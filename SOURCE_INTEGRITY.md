# Source integrity and split isolation

- Source archive: `MP1_student_starter.zip`
- Source archive SHA-256: `755f40ca111f901c6f2672e3744b894f1af112bc70fa00323b19dbf36d20950b`
- Formal protocol start: 2026-09-17 (UTC+8)
- Primary formal seed: `7506`

This development workspace was created from the source archive while excluding
`__MACOSX`, `.DS_Store`, and `code/data/wikitext_test.txt`. Training and model
selection therefore run without the test text present in the development tree.
The official `common.py` and `evaluate.py` remain unchanged. A separate clean
archive extraction is reserved for evaluation after the predictor is frozen.

Earlier exploratory work outside this workspace accessed the public test split.
The clean formal run cannot erase that history. The final report must describe
this work as a preregistered, validation-controlled confirmation after prior
exploratory test exposure, not as a project in which the test set was never
seen.
