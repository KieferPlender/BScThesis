import json
import random
import os

# Files to inspect
ORIGINAL_FILE = 'data/mt_bench/model_answers/vicuna_13b.jsonl'
NEUTRAL_FILE = 'data/mt_bench/neutralized_answers/vicuna_13b_neutral.jsonl'
BACKTRANS_FILE = 'data/mt_bench/neutralized_answers/vicuna_13b_backtrans.jsonl'

def load_random_sample():
    if not os.path.exists(NEUTRAL_FILE):
        print("❌ Run Step 3 first! Files not found.")
        return

    # Load all lines
    with open(ORIGINAL_FILE) as f: orig = [json.loads(line) for line in f]
    with open(NEUTRAL_FILE) as f: neut = [json.loads(line) for line in f]
    
    # Check Backtrans if it exists
    bt = []
    if os.path.exists(BACKTRANS_FILE):
        with open(BACKTRANS_FILE) as f: bt = [json.loads(line) for line in f]

    # Pick a random index
    idx = random.randint(0, len(orig) - 1)
    
    q_id = orig[idx]['question_id']
    
    print(f"\n🔍 INSPECTING SAMPLE (Question ID: {q_id})")
    print("="*80)
    
    print(f"🔵 ORIGINAL (Vicuna):\n{orig[idx]['choices'][0]['turns'][-1]}")
    print("-" * 80)
    
    print(f"🟢 NEUTRALIZED (Paraphrased):\n{neut[idx]['choices'][0]['turns'][-1]}")
    print("-" * 80)
    
    if bt:
        print(f"🟣 BACK-TRANSLATED (En->De->En):\n{bt[idx]['choices'][0]['turns'][-1]}")
        print("="*80)

if __name__ == "__main__":
    load_random_sample()