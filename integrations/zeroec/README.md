# ZeroEC

ZeroEC (Wu et al., ICDE 2025; github.com/YangChen32768/ZeroEC, commit d76ef45). Copy
these files into the ZeroEC repository root and run them there.

- `run_zeroec.py` runs the release's `correction.py` unchanged except for what it
  hard-codes: dataset paths, output folder, prompt folder name, model, endpoint and key,
  embedding path and device, and the `dashscope` token counter. It records which cells
  went to the correction prompt, the 10 user-labelled rows, and token usage, with a hard
  cost budget.
- Detection is oracle (clean ≠ dirty), as in ZeroEC's main experiments.
- ZeroEC's tables equal CARE's cell by cell, up to case, whitespace and null tokens.
- `export_zeroec.py` writes `csvs/zeroec_<table>_pred.csv`: changed cells outside the
  labelled rows; `source` is `zeroec_llm` or `zeroec_rule`; `confidence` is the share of
  runs that agree (empty with one run).

Every stage uses gpt-4o-mini (ZeroEC's paper used GPT-4o and GPT-3.5-turbo); embeddings
run on CPU with the same int8 MiniLM.

    bash run_zeroec.sh build
    bash run_zeroec.sh mock hospital          # pipeline test, no API calls
    bash run_zeroec.sh dry hospital           # cost bound, no API calls
    bash run_zeroec.sh run hospital 0 1 2
    bash run_zeroec.sh export hospital
    # in CARE:
    bash integrations/sweep_offline_logs.sh zeroec hospital
