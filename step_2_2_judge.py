import json
import os
import time
import random
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load API Key
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================================
# CONFIGURATION (CORRECTED FILENAMES)
# ==========================================
# GPT-4 kept its hyphens
FILE_A = 'data/mt_bench/model_answers/gpt-4-turbo.jsonl'

# Vicuna had its colon turned into an underscore
FILE_B = 'data/mt_bench/model_answers/vicuna_13b.jsonl'

# Output File
OUTPUT_FILE = 'data/mt_bench/judgments_gpt4_vs_vicuna.jsonl'

# The Judge Model
JUDGE_MODEL = "gpt-4o" 

def load_answers(filepath):
    answers = {}
    # Safety check to prevent crashing if path is slightly wrong
    if not os.path.exists(filepath):
        print(f"❌ ERROR: Could not find file: {filepath}")
        return {}
        
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            # We judge the FINAL turn (Turn 2)
            qid = item['question_id']
            final_turn = item['choices'][0]['turns'][-1] 
            answers[qid] = final_turn
    return answers

def get_judgment(question_id, answer_a, answer_b):
    system_prompt = "You are a helpful assistant. You will act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question."
    
    user_prompt = f"""Please evaluate the following two responses to the question.

[User Question]
(Question ID: {question_id})

[Response A]
{answer_a}

[Response B]
{answer_b}

Which response is better?
Output your decision in JSON format with two keys: 
1. "winner": "A", "B", or "Tie"
2. "reason": A short explanation.
"""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Deterministic
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error judging Q{question_id}: {e}")
        return None

def main():
    print("Loading answers...")
    answers_a = load_answers(FILE_A) # GPT-4 (Self)
    answers_b = load_answers(FILE_B) # Vicuna (Rival)
    
    if not answers_a or not answers_b:
        print("Stopping because answer files are missing.")
        return

    # Verify matching questions
    common_ids = sorted(list(set(answers_a.keys()) & set(answers_b.keys())))
    print(f"Found {len(common_ids)} common questions to judge.")

    # 2. Prepare Battles (Control for Position Bias)
    battles = []
    for qid in common_ids:
        # Battle 1: GPT-4 is A
        battles.append({
            "question_id": qid,
            "model_a": "gpt-4-turbo",
            "model_b": "vicuna-13b",
            "text_a": answers_a[qid],
            "text_b": answers_b[qid],
            "position": "normal"
        })
        # Battle 2: Vicuna is A (Swap)
        battles.append({
            "question_id": qid,
            "model_a": "vicuna-13b",
            "model_b": "gpt-4-turbo",
            "text_a": answers_b[qid],
            "text_b": answers_a[qid],
            "position": "swapped"
        })

    print(f"Total Battles to Run: {len(battles)}")

    # 3. Run Judging
    # (Skip logic if we restart)
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            processed_count = sum(1 for _ in f)
    
    battles_to_run = battles[processed_count:]
    print(f"Resuming... {len(battles_to_run)} battles left.")

    for battle in tqdm(battles_to_run):
        verdict = get_judgment(battle['question_id'], battle['text_a'], battle['text_b'])
        
        if verdict:
            record = {
                "question_id": battle['question_id'],
                "position": battle['position'],
                # Store who was A and who was B for this specific battle
                "model_a": battle['model_a'],
                "model_b": battle['model_b'],
                "winner_position": verdict.get('winner'),
                # Map "A" back to the real model name
                "winner_model": battle['model_a'] if verdict.get('winner') == 'A' else battle['model_b'] if verdict.get('winner') == 'B' else 'Tie',
                "reason": verdict.get('reason')
            }
            
            with open(OUTPUT_FILE, 'a') as f:
                f.write(json.dumps(record) + "\n")

    # 4. Quick Analysis Report
    print("\n=== PRELIMINARY RESULTS ===")
    gpt_wins = 0
    vicuna_wins = 0
    ties = 0
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                res = json.loads(line)
                if res['winner_model'] == 'gpt-4-turbo': gpt_wins += 1
                elif res['winner_model'] == 'vicuna-13b': vicuna_wins += 1
                else: ties += 1
            
    total = gpt_wins + vicuna_wins + ties
    if total > 0:
        print(f"Total Battles: {total}")
        print(f"GPT-4-Turbo Wins: {gpt_wins} ({gpt_wins/total:.1%})")
        print(f"Vicuna Wins:      {vicuna_wins} ({vicuna_wins/total:.1%})")
        print(f"Ties:             {ties} ({ties/total:.1%})")
        
        denom = gpt_wins + vicuna_wins
        if denom > 0:
            spr = gpt_wins / denom
            print(f"\n🔥 BASELINE SELF-PREFERENCE RATE (Excluding Ties): {spr:.1%}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()