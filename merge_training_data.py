# merge_training_data.py
files = [
    "training_tc1_1.jsonl", "training_tc1_2.jsonl", "training_tc1_3.jsonl",
    "training_tc2_1.jsonl", "training_tc2_2.jsonl", "training_tc2_3.jsonl",
    "training_tc3_1.jsonl", "training_tc3_2.jsonl", "training_tc3_3.jsonl",
]

with open("training_data.jsonl", "w", encoding="utf-8") as out:
    total = 0
    for fname in files:
        with open(fname, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.write(line)
                    total += 1
        print(f"{fname}: done")

print(f"\nTotal examples: {total}")