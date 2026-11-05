# The OOMem labs

Nine small, reproducible PySpark labs for examining Catalyst behavior behind
the article. Lab 1 is an interactive notebook; the remaining bounded
experiments retain their small Python runners.

The layout intentionally follows the other article labs: one shared
`common.py`, one numbered experiment per question, a pinned requirements file,
and no hidden production data or generated trace dump.

## Install

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Java 17 or later (Java 21 recommended) must be available on `PATH`.

## Run the labs

Open Lab 1 from the project root, then run the remaining bounded experiments:

1. `lab_01__the_unborn_query.ipynb` — the real-fixture logical-plan stress case that reaches `OutOfMemoryError` with Spark's default 1 GB driver heap.
2. `2_nulls_and_duplicates.py` — rewrite counterexamples.
3. `3_constraint_growth.py` — aliases and optimized plan shape.
4. `4_outer_vs_anti.py` — bounded planning comparison.
5. `5_nested_projects.py` — a `withColumn` staircase and nested `Project` nodes.
6. `6_expression_bloat.py` — repeated deep expressions and `semanticEquals`.
7. `7_union_fanout.py` — duplicated branch trees without a join.
8. `8_boolean_expression_bloat.py` — deeply nested predicates and `CASE` trees.
9. `9_aqe_phase_split.py` — AQE on/off with logical and physical phases separated.

```powershell
jupyter notebook lab_01__the_unborn_query.ipynb
python 2_nulls_and_duplicates.py
python 3_constraint_growth.py --alias-depth 6
python 4_outer_vs_anti.py --rows 240
python 5_nested_projects.py --depth 20
python 6_expression_bloat.py --depth 20 --width 8
python 7_union_fanout.py --branches 8 --rows 240
python 8_boolean_expression_bloat.py --depth 20
python 9_aqe_phase_split.py --rows 240 --branches 8
```

Every remaining script accepts the shared `--master`, `--shuffle-partitions`, `--rows`,
`--alias-depth`, `--depth`, `--branches`, `--width`, and `--app-name` options.
`common.py` owns Spark session setup, timing, multiset assertions, and
plan-shape summaries so each lab has the same operational skeleton.

The first lab is the failure reproduction: the original two-key `right_outer`
shape, the checked-in `.schema` fixtures and binary-key decoding, generated-width
projections, retained optimized-plan metadata, and the default 1 GB heap crossing
into `OutOfMemoryError` during logical planning. The remaining
labs isolate the pressure sources one at a time—constraints, nested projects,
expression canonicalization, union fan-out, boolean trees, and AQE—so the reader
can see why Catalyst spends CPU and heap before an executor is ever scheduled.
They use bounded inputs and do not tune Spark to make the numbers look better.

Each script creates its own local Spark session. Lab 1 reads the two CSV and
`.schema` fixtures from the original failing test and decodes the bracketed binary
keys; the other runners use deterministic bounded data
to isolate one Catalyst mechanism. No package, test suite, research dump or
generated result file is required. The assertions compare row multisets and
schemas where applicable. Lab 1 is intentionally destructive to its fresh local
driver: after the expected OOM, restart the notebook kernel before running
another lab.
