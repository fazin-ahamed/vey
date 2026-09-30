# STEF enlarged-evaluation provenance

The 2,000-decision pool behind the enlarged held evaluation, and how to
rebuild it. Raw rows are not committed; this is the recipe.

## Source

- File: `/home/fazinahamed/Documents/vey-data/decisionmix/d6/generic/train.jsonl`
- sha256: `25586dbfe6a72329b11eca51c8f9566c51e372f707746a475e2bf2e92b53e08e`
- Rows: 12,000
- Generator: `research/d6/generic_corpus.py`, hash-derived per-row RNG, writes
  12,000 train rows. Its labels are gold indices, not the scores used here.

## The training draw

- `random.seed(0)`, shuffle the 12,000 rows, take the first 2,500.
- Deduplicate by the option tuple. In practice none collide, leaving 2,500.
- `random.seed(7)`, shuffle, 90/10 split: 2,250 train, 250 held.
- These 2,500 option tuples are excluded from the dev pool.

## The dev pool

- The remaining 9,500 rows, none sharing an option tuple with the training draw.
- `random.seed(1)`, shuffle, take the first 2,000.
- First 50 are the calibration split, used only to fit alpha. The other 1,950
  are the test. The bootstrap never sees the calibration rows.

## The teacher

- `FieldScorer` from `vey.crux.field`, `ordinal_many(options, axes)`, device cpu.
- Artifact revision: `3eb1460a82c98fc99c350032b9cf29a74075a4a6` (DEFAULT_FIELD_REV,
  field.py:26), resolved through the published default, not a local override.
- Axes, in teacher-column order: food progress, open space, headroom.
- Returns continuous potentials, not ranks. A label that is an integer is a
  sign the call was wrong.

## The evaluation

- Seed 7, single run. Weights were not persisted, so a re-score means re-running
  the deterministic training.
- Bootstrap: 10,000 resamples over decisions, `random.Random(0)`.
- Tight band: absolute gap within 0.02 of 0.15.
- Tight pairs on the 1,950 test decisions: 1,386, across 884 decisions.
