import json, sys
for i, claims in enumerate(json.load(open(sys.argv[1])), 1):
    print(f"\nDataset {i}")
    for c in claims:
        print(" -", c.get("claim"))