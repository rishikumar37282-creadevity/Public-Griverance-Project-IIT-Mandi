"""Build data/category_tree.json from the CPGRAMS CategoryCode_Mapping.xlsx.

Keeps only the org codes present in grievances.csv and stores each node's
code, description, parent, and stage — used by the app to suggest the most
specific CPGRAMS sub-category for a complaint via keyword matching.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    grievances = pd.read_csv(DATA / "grievances.csv")
    org_codes = sorted(grievances["org_code"].unique())
    xl = pd.read_excel(DATA / "CategoryCode_Mapping.xlsx")
    xl = xl[xl["OrgCode"].isin(org_codes)].copy()

    tree = {}
    for org, grp in xl.groupby("OrgCode"):
        nodes = []
        for _, r in grp.iterrows():
            nodes.append({
                "code": int(r["Code"]),
                "desc": str(r["Description"]).strip(),
                "parent": None if pd.isna(r["Parent"]) else int(r["Parent"]),
                "stage": int(r["Stage"]),
            })
        tree[org] = nodes

    out = DATA / "category_tree.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False)
    total = sum(len(v) for v in tree.values())
    print(f"Wrote {out} with {total} nodes across {len(tree)} organisations.")


if __name__ == "__main__":
    main()
