# Synthetic human-validation fixture

`synthetic_human_validation.json` contains four entirely synthetic final
annotation records for software testing. These are not real judicial cases,
not expert judgments about real cases, and not the manuscript's 150-case
sample or 132-case subset.

Only one synthetic record meets all three criteria. Each other record fails
a different criterion; unresolved removal preservation is excluded.

From the repository root, run without installing the package:

    PYTHONPATH=src python -m unittest discover -s tests -p 'test_human_validation.py'

The selector accepts canonical final records only. It does not infer consensus,
map historical annotation columns automatically, or use predictions for
selection. Its aggregate summary does not expose case identifiers.

Private empirical records must be supplied separately. Synthetic tests verify
software behavior, not empirical reproduction.
