import json
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================================
# CONFIGURATION
# ==========================================
JUDGE_MODEL = "gpt-4o"
OUTPUT_DIR = 'data/mt_bench/final_results'

# The Three Experiments
EXPERIMENTS = [
    {
        "name": "1_BASELINE",
        "file_a": "data/mt_bench/model_answers/gpt_4_turbo.jsonl",
        "file_b": "data/mt_bench/model_answers/vicuna_13b.jsonl"
    },
    {
        "name": "2_NEUTRAL_INTERVENTION",
        "file_a": "data/mt_bench/neutralized_answers/gpt_4_turbo_neutral.jsonl",
        "file_b": "data/mt_bench/neutralized_answers/vicuna_13b_neutral.jsonl"
    },
    {
        "name": "3_BACKTRANS_INTERVENTION",
        "file_a": "data/mt_bench/neutralized_answers/gpt_4_turbo_backtrans.jsonl",
        "file_b": "data/mt_bench/neutralized_answers/vicuna_13b_backtrans.jsonl"
    }
]

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_answers(filepath):
    answers = {}
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item['question_id']
            final_turn = item['choices'][0]['turns'][-1] 
            answers[qid] = final_turn
    return answers

def get_judgment(question_id, answer_a, answer_b):
    # Deterministic Judge
    system_prompt = "You are a helpful assistant. You will act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question."
    user_prompt = f"Please evaluate the following two responses to the question.\n\n[User Question]\n(Question ID: {question_id})\n\n[Response A]\n{answer_a}\n\n[Response B]\n{answer_b}\n\nWhich response is better?\nOutput your decision in JSON format with two keys: \n1. \"winner\": \"A\", \"B\", or \"Tie\"\n2. \"reason\": A short explanation."

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_experiment(exp):
    print(f"\n⚔️ STARTING EXPERIMENT: {exp['name']}")
    output_file = os.path.join(OUTPUT_DIR, f"results_{exp['name']}.jsonl")
    
    # Check if done
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            count = sum(1 for _ in f)
        if count >= 70: # Rough check
            print("   Already complete. Skipping.")
            return

    answers_a = load_answers(exp['file_a'])
    answers_b = load_answers(exp['file_b'])
    common_ids = sorted(list(set(answers_a.keys()) & set(answers_b.keys())))
    print(f"   Contestants ready. Judging {len(common_ids)} battles...")

    results = []
    for qid in tqdm(common_ids):
        # Battle 1: Normal
        res = get_judgment(qid, answers_a[qid], answers_b[qid])
        if res:
            record = {
                "question_id": qid, "exp": exp['name'], "position": "normal",
                "winner": "GPT-4" if res['winner'] == 'A' else "Vicuna" if res['winner'] == 'B' else "Tie"
            }
            with open(output_file, 'a') as f: f.write(json.dumps(record) + "\n")
            
        # Battle 2: Swapped (Control)
        res_swap = get_judgment(qid, answers_b[qid], answers_a[qid])
        if res_swap:
            record = {
                "question_id": qid, "exp": exp['name'], "position": "swapped",
                "winner": "GPT-4" if res_swap['winner'] == 'B' else "Vicuna" if res_swap['winner'] == 'A' else "Tie"
            }
            with open(output_file, 'a') as f: f.write(json.dumps(record) + "\n")

def print_summary():
    print("\n" + "="*60)
    print(f"{'EXPERIMENT':<30} | {'GPT-4 WIN RATE':<15} | {'DELTA'}")
    print("-" * 60)
    
    baseline_rate = 0
    
    for exp in EXPERIMENTS:
        result_file = os.path.join(OUTPUT_DIR, f"results_{exp['name']}.jsonl")
        if not os.path.exists(result_file): continue
        
        wins = 0
        total = 0
        with open(result_file, 'r') as f:
            for line in f:
                res = json.loads(line)
                total += 1
                if res['winner'] == 'GPT-4': wins += 1
        
        if total == 0: continue
        rate = wins / total
        
        if exp['name'] == "1_BASELINE":
            baseline_rate = rate
            delta = "---"
        else:
            diff = rate - baseline_rate
            delta = f"{diff:+.1%}"
            
        print(f"{exp['name']:<30} | {rate:.1%}          | {delta}")
    print("="*60)

def main():
    setup_directories()
    for exp in EXPERIMENTS:
        run_experiment(exp)
    print_summary()

if __name__ == "__main__":
    main()