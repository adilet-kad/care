# Denial constraints for beers and flights

HoloClean ships denial constraints for hospital only. Its `testdata/flight.csv` is a
different extraction (57,246 rows, eight columns) from the suite's (2,376 rows, six
columns), and beers is absent. These files are the subset of Ni et al.'s constraints
(PVLDB 2024, `WelkinNi/Automatic-Data-Repair`) that CARE's schema supports. They were
chosen from the schema, without the gold standard; the only data check was whether the
determinant repeats.

```
beers    brewery_id   558 distinct over 2,410 rows
flights  flight       100 distinct over 2,376 rows
```

**flights** (four of Ni et al.'s six): `flight -> sched_dep_time, sched_arr_time,
act_dep_time, act_arr_time`. Omitted: `sched_arr_time -> act_arr_time` (flights arrive
late, so the rule fires on correct cells) and a rule equating a departure column with an
arrival column.

**beers** (two of five): `brewery_id -> city, state`. Omitted: the three rules on
`brewery_name` and `beer_name`, which CARE's schema spells `brewery-name` and `beer-name`,
so their repairs would be dropped as out of schema. Also rejected: `id -> *` (`id` is
unique) and `city -> state` (false in the US).

With these constraints HoloClean repairs only `city` and `state` on beers; most of its
changes there are null imputations outside the gold standard, so the pair is not scored.
