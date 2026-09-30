"""sanitize_reasoning.py -- strip verbalised reasoning out of a proposer log's value field.

An instruction-tuned model asked for a cell value sometimes answers with a derivation
instead: the first verify-correct run wrote

    Bucyrus -> Bucyrus (no change needed) -> Bucyrus

verbatim into beers row 1655's `city`. A value is a cell's contents, not an argument
about them, so a log row like that is unusable: it can never match gold, and applied to
a database it would write the reasoning into the table.

`export_jellyfish.py::_clean` now cuts at the first reasoning marker, but the logs from
the first run were exported before that fix. Re-running costs hours of GPU for ~10
cells, so this applies the same rule to an existing log instead. The transformation is
deterministic and depends only on the recorded value, so a sanitised log is exactly what
the fixed exporter would have written for those rows -- with one caveat stated plainly:
the raw completion is not recoverable from the log, so if a cut leaves nothing the row
becomes an abstention, which is what `_clean` would also have produced.

Every change is printed, and the original is kept as `<log>.preclean.bak`. Rows that do
not match a marker are copied byte-for-byte.

    python tools/sanitize_reasoning.py csvs/jellyfish_vc_*_pred.csv
    python tools/sanitize_reasoning.py csvs/jellyfish_vc_beers_pred.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil

# Must stay identical to export_jellyfish.py::_REASON_CUT.
REASON_CUT = re.compile(
    r"\s*(?:->|=>|→|\bno change needed\b|\bi\.e\.\b|\bbecause\b).*",
    re.IGNORECASE | re.DOTALL)
TRIM = " .;:–—-"


def clean(value: str) -> str:
    t = (value or "").strip().split("\n")[0].strip()
    t = REASON_CUT.sub("", t).strip().strip(TRIM)
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", help="proposer logs (globs are expanded)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = sorted({p for g in args.logs for p in glob.glob(g)})
    if not paths:
        raise SystemExit(f"no logs matched {args.logs}")

    grand_cut = grand_void = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rdr = csv.DictReader(fh)
            fields = rdr.fieldnames or []
            rows = list(rdr)
        if "value" not in fields:
            print(f"[skip] {path}: no `value` column")
            continue

        cut = void = 0
        for r in rows:
            old = r["value"] or ""
            if not old.strip():
                continue                       # already an abstention
            new = clean(old)
            if new == old.strip():
                continue
            cut += 1
            if not new:
                void += 1
                # nothing survived: the model gave reasoning only -> abstention, and
                # the confidence must go too, or it reads as "sure about nothing".
                r["value"] = ""
                if "confidence" in r:
                    r["confidence"] = ""
            else:
                r["value"] = new
            print(f"  {os.path.basename(path)} row {r.get('row_id')} "
                  f"{r.get('column')}: {old!r} -> {new!r}"
                  + ("   [-> abstention]" if not new else ""))

        print(f"{path}: {cut} value(s) cut, {void} became abstentions "
              f"({len(rows)} rows total)" + ("  [dry run]" if args.dry_run else ""))
        grand_cut += cut
        grand_void += void
        if cut and not args.dry_run:
            shutil.copy(path, path + ".preclean.bak")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

    print(f"\ntotal: {grand_cut} cut, {grand_void} became abstentions, "
          f"across {len(paths)} log(s)")
    if grand_cut and not args.dry_run:
        print("originals kept as *.preclean.bak; re-run prepare_log.py on each log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
