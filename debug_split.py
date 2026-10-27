import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.yaml"

def _parse_split_ranges(config_text):
    """Parse train/validation/final_test date ranges from experiment.yaml."""
    ranges = {}
    for name in ("train", "validation", "final_test"):
        pattern = rf"^    {re.escape(name)}:\s*target_dates:\s*\[\s*['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\s*\]"
        m = re.search(pattern, config_text, re.MULTILINE)
        if m:
            ranges[name] = (m.group(1), m.group(2))
    return ranges

config_text = CONFIG_PATH.read_text(encoding="utf-8-sig")
split_ranges = _parse_split_ranges(config_text)
print("Split ranges:", split_ranges)

# Test with 2022-06-15
target_date = "2022-06-15"
td = datetime.strptime(target_date, "%Y-%m-%d").date()
print(f"Target date: {td}")

for name, (start_str, end_str) in split_ranges.items():
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    print(f"  {name}: start={start}, end={end}, td in range: {start <= td <= end}")