# Denial constraints for beers and flights

HoloClean ships denial constraints for **hospital only**. Its `testdata/` also contains
a `flight.csv`, but that is a different extraction from CARE's (57,246 rows against
2,376, eight columns against six, `scheduled_dept` against `sched_dep_time`), so its
constraints would not apply and its logs would not key against CARE's gold set. beers is
absent entirely.

These two files are therefore **not HoloClean's**: they are the schema-supported subset of
Ni et al.'s constraints (see below), and that difference is
recorded in `docs/REPRODUCE.md`: D3 rests on using the authors' constraints verbatim,
which stays true for hospital and stops being true here.

## How they were selected

From the **schema**, before looking at any result, and without consulting the gold
standard; then reconciled against Ni et al.'s files (every rule we kept is in theirs). The only data-level check performed was structural: does the left-hand side
of each dependency actually repeat? A determinant that is unique per row can never be
violated, so the constraint would be inert and its presence would misrepresent the
constraint set's strength.

```
beers    brewery_id   558 distinct over 2,410 rows, 431 repeat, largest group 62
flights  flight       100 distinct over 2,376 rows, all repeat, largest group 29
         id           2,410 distinct, 0 repeat   <- rejected, cannot fire
```

## Relation to Ni et al.'s constraint files

Ni et al. (PVLDB 2024, `WelkinNi/Automatic-Data-Repair`) ship denial constraints for both
tables. Ours are a **subset of theirs**, not a competing set: every constraint below
appears in their file, and the omissions are listed with reasons so that a reader
comparing the two files knows exactly what was left out and why.

## flights: four of Ni's six constraints

```
flight -> sched_dep_time,  sched_arr_time,  act_dep_time,  act_arr_time
```

A flight number on a given day has one scheduled departure, one scheduled arrival, one
actual departure and one actual arrival. The benchmark's rows are the *same flights
reported by ~24 different sources*, so two rows agreeing on `flight` and disagreeing on
a time is exactly the error this dataset contains.

Omitted from Ni's file, with reasons:

- `sched_arr_time -> act_arr_time`: a scheduled arrival does not determine an actual
  arrival; flights arrive late. Discrepancies between the two are the *content* of this
  benchmark, not errors, so the rule would fire on correct cells and inject wrong repairs.
- `EQ(t1.sched_dep_time, t2.sched_arr_time) & ...`: compares a departure column to an
  arrival column across tuples. We read this as a slip in the upstream file (no schema
  reading makes a departure time determine an arrival time) and did not carry it over.

## beers: two of Ni's five constraints

```
brewery_id -> city,  state
```

A brewery has one location.

Omitted from Ni's file, with reasons:

- `brewery_name -> brewery_id`, `brewery_id -> brewery_name`,
  `beer_name, brewery_name -> brewery_id`: all three constrain `brewery_name` or
  `beer_name`. CARE's beers schema spells these columns with hyphens (`brewery-name`,
  `beer-name`) while the CSV uses underscores, so `prepare_log.py` drops every repair to
  them as out-of-schema. Keeping the rules would spend HoloClean's domain modelling on
  columns whose repairs are then discarded and produce a log that looks larger than the
  part of it CARE can audit. This is a schema-naming artefact of our pipeline, not a
  judgement about the rules.

Also rejected on data grounds (not in Ni's file, listed for completeness):

- `id -> anything`: `id` is unique per row (2,410 distinct over 2,410 rows), so the
  constraint can never fire.
- `city -> state`: **false in the United States.** Portland is in Oregon and in Maine;
  Springfield is in a dozen states.

**Consequence to expect:** with two constraints over two columns, HoloClean only proposes
repairs to `city` and `state` on beers. The run is narrow, and a narrow run is the honest
outcome of what this schema (as CARE spells it) supports, not a tuning failure. In the
paper this pair is degenerate (711 of 878 changes are null imputations on non-error
cells) and is excluded from Table 1, so nothing above changes a reported number.
