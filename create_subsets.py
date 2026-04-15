import json, random, os

random.seed(42)

with open("training_data.jsonl") as f:
    examples = [json.loads(l) for l in f]

tc1 = [e for e in examples if e["instruction"].startswith("Parse the following")]
tc2 = [e for e in examples if e["instruction"].startswith("You will receive a JSON")]
tc3 = [e for e in examples if e["instruction"].startswith("You are a pay equity compliance auditor")]

print(f"TC1: {len(tc1)}, TC2: {len(tc2)}, TC3: {len(tc3)}")

for n_total in [20, 40, 60]:
    per_task = n_total // 3
    subset = (random.sample(tc1, per_task) +
              random.sample(tc2, per_task) +
              random.sample(tc3, per_task))
    random.shuffle(subset)
    path = f"train_n{n_total}.jsonl"
    with open(path, "w") as f:
        for e in subset:
            f.write(json.dumps(e) + "\n")
    print(f"n={n_total}: {len(subset)} examples → {path}")