import json
from pathlib import Path

p = Path("tutorials/quick_visualize_q.ipynb")
nb = json.load(open(p, encoding="utf-8"))

old = "between the two are expected.\n"
new = (
    "between the two are expected. In noisy data, the two can even disagree entirely.\n"
)

for cell in nb["cells"]:
    if cell.get("id") == "troubleshooting":
        src = (
            cell["source"]
            if isinstance(cell["source"], str)
            else "".join(cell["source"])
        )
        assert old in src, "String not found!"
        cell["source"] = src.replace(old, new, 1)
        break

json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Done")
