import subprocess, sys

python = sys.executable

runs = [
    ("train_n20.jsonl", "./output_peft/llama-ablation-n20"),
    ("train_n40.jsonl", "./output_peft/llama-ablation-n40"),
    ("train_n60.jsonl", "./output_peft/llama-ablation-n60"),
]

for data_path, output_dir in runs:
    print(f"\n{'='*60}")
    print(f"Starting: {data_path} → {output_dir}")
    print(f"{'='*60}\n")
    subprocess.run([
        python, "finetune_peft (Llama).py",
        f"--data_path={data_path}",
        f"--output_dir={output_dir}",
    ], check=True)

print("\nAll ablation runs complete.")