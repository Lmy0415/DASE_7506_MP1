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

The complete method, checkpoint, configuration and implementation were frozen
and hash-recorded before one final full-test evaluation, and no choice changed
afterward. An earlier discarded exploratory workspace had accessed the public
test split, but those runs are excluded from formal evidence. The submitted
holdout claim applies to the validation-only formal rerun and the frozen
predictor's single final evaluation.
