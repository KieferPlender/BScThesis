import json
import os

# Files to Clean (We clean ALL versions to keep them synced)
FILES = [
    # Originals
    'data/mt_bench/model_answers/vicuna_13b.jsonl',
    'data/mt_bench/model_answers/gpt_4_turbo.jsonl',
    # Neutralized (If you already generated them)
    'data/mt_bench/neutralized_answers/vicuna_13b_neutral.jsonl',
    'data/mt_bench/neutralized_answers/gpt_4_turbo_neutral.jsonl',
    # Back-Translated (If you already generated them)
    'data/mt_bench/neutralized_answers/vicuna_13b_backtrans.jsonl',
    'data/mt_bench/neutralized_answers/gpt_4_turbo_backtrans.jsonl'
]

# The IDs of the questions that failed (Visually confirmed by you)
BAD_IDS = [100, 101, 103] 

def clean_file(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping missing file: {filepath}")
        return

    print(f"Cleaning {os.path.basename(filepath)}...")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    clean_lines = []
    for line in lines:
        item = json.loads(line)
        if item['question_id'] not in BAD_IDS:
            clean_lines.append(line)
            
    # Overwrite the file with the clean version
    with open(filepath, 'w') as f:
        f.writelines(clean_lines)
        
    print(f" -> Removed {len(lines) - len(clean_lines)} bad rows. Remaining: {len(clean_lines)}")

if __name__ == "__main__":
    print(f"🚫 Removing Bad Questions: {BAD_IDS}")
    for f in FILES:
        clean_file(f)
    print("\n✅ Cleanup Complete. You are ready for Step 4.")