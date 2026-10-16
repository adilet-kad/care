#!/usr/bin/env bash
# run_generate_uc.sh -- generate BClean+ user constraints for rayyan and tax.
#
# Runs the authors' generate_UC.py (flan-t5-xxl zero-shot column typing, patterns from
# summary_sample.csv) on _care_input/<ds>_for_uc.csv and writes UC_json/<ds>_schema.json.
# Launched by run_bclean_plus.sh ("uc" stage) inside the GPU container. The first run
# downloads google/flan-t5-xxl (~45 GB) into ./flan-t5-xxl.

set -uo pipefail
DATASETS=${*:-"rayyan tax"}

# imports generate_UC.py needs at module level
pip install -q tdda==2.2.17 python-dotenv retry fuzzywuzzy Levenshtein langchain==0.0.305 \
    sentence-transformers accelerate bitsandbytes openai matplotlib seaborn 2>&1 | tail -1

if [ ! -f flan-t5-xxl/config.json ]; then
  python - <<'PY' || exit 1
from huggingface_hub import snapshot_download
snapshot_download("google/flan-t5-xxl", local_dir="flan-t5-xxl",
                  ignore_patterns=["*.msgpack", "*.h5", "*.ckpt", "*.bin"])
PY
fi

# Wrapper: point src/const.py's ARCHETYPE_PATH at this repository, and with
# FLAN_BF16=1 load flan-t5-xxl in bfloat16 instead of 8-bit (bitsandbytes).
run_uc() {
  python - "$@" <<'PY'
import os, runpy, sys, types
sys.path.insert(0, os.getcwd())
import src.const
src.const.ARCHETYPE_PATH = os.getcwd()
if os.environ.get("FLAN_BF16") == "1":
    import src as pkg
    code = open("src/model.py").read()
    old = ('AutoModelForSeq2SeqLM.from_pretrained("flan-t5-xxl", device_map="auto", '
           'torch_dtype=torch.float16, load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)')
    assert code.count(old) == 1, "flan-t5-xxl loader not found in src/model.py"
    code = code.replace(old, 'AutoModelForSeq2SeqLM.from_pretrained("flan-t5-xxl", '
                             'device_map="auto", torch_dtype=torch.bfloat16)')
    mod = types.ModuleType("src.model"); mod.__file__ = "src/model.py"; mod.__package__ = "src"
    sys.modules["src.model"] = mod
    exec(compile(code, "src/model.py", "exec"), mod.__dict__)
    pkg.model = mod
    print("  flan-t5-xxl: bfloat16 load (FLAN_BF16=1)", flush=True)
sys.argv = ["generate_UC.py"] + sys.argv[1:]
runpy.run_path("generate_UC.py", run_name="__main__")
PY
}

for ds in $DATASETS; do
  [ -f "_care_input/${ds}_for_uc.csv" ] || { echo "_care_input/${ds}_for_uc.csv missing"; continue; }
  run_uc --model_name="flan-t5-xxl-zs" \
      --input_files="_care_input/${ds}_for_uc.csv" \
      --save_path="./UC_json/${ds}_schema.json" \
      --input_labels="skip-eval-return" --label_set="custom" \
      --custom-labels Identifier Code PersonName OrganizationName LocationName ObjectName \
        Address DateTime Age Count Measurement Money Percentage Score Category Text \
        LanguageName Email PhoneNumber URL Boolean Year \
      --response --generate_schema --summary_csv_path="summary_sample.csv" \
    || echo "ERROR: generate_UC failed for $ds"
  ls -la "UC_json/${ds}_schema.json" 2>/dev/null
done
