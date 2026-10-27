import re

with open("btc_prediction_experiment/config/experiment.yaml") as f:
    text = f.read()

for name in ["train", "validation", "final_test"]:
    pattern = rf"^    {re.escape(name)}:\s*target_dates:\s*\[\s*['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\s*\]"
    m = re.search(pattern, text, re.MULTILINE)
    if m:
        print(f"{name}: start={m.group(1)}, end={m.group(2)}")
    else:
        print(f"{name}: NO MATCH")