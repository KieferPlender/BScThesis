import json
import os
from tqdm import tqdm
from openai import OpenAI
from bert_score import score
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILES = [
    'data/mt_bench/model_answers/gpt-4-turbo.jsonl',
    'data/mt_bench/model_answers/vicuna_13b.jsonl'
]
OUTPUT_DIR = 'data/mt_bench/neutralized_answers'

# Quality Threshold
BERTSCORE_THRESHOLD = 0.85 

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# --- TECHNIQUE A: NEUTRAL PARAPHRASING (SMART) ---
def rewrite_neutral(text):
    system_prompt = (
        "Rewrite the following text to be concise, objective, and stylistically neutral. "
        "Remove distinct mannerisms. Do NOT add new information or improve the logic. "
        "Preserve the original meaning exactly."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        return None

# --- TECHNIQUE B: BACK-TRANSLATION (DUMB) ---
def back_translate(text):
    # Step 1: English -> German
    try:
        res_de = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Translate the following text to German. Output ONLY the translation:\n\n{text}"}],
            temperature=0.3,
        )
        text_de = res_de.choices[0].message.content
        
        # Step 2: German -> English
        res_en = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Translate the following text back to English. Output ONLY the translation:\n\n{text_de}"}],
            temperature=0.3,
        )
        return res_en.choices[0].message.content
    except Exception:
        return None

# --- FIDELITY CHECK ---
def check_fidelity(original, candidate):
    # Returns F1 Score
    if not candidate: return 0.0
    try:
        P, R, F1 = score([candidate], [original], lang="en", verbose=False)
        return F1.item()
    except:
        return 0.0

def process_file(filepath, method_name, method_func):
    filename = os.path.basename(filepath)
    new_filename = filename.replace(".jsonl", f"_{method_name}.jsonl")
    output_path = os.path.join(OUTPUT_DIR, new_filename)
    
    if os.path.exists(output_path):
        print(f"Skipping {new_filename} (exists)")
        return

    print(f"\n🧪 Running {method_name.upper()} on {filename}...")
    new_records = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        item = json.loads(line)
        new_choices = []
        for choice in item['choices']:
            new_turns = []
            for turn_text in choice['turns']:
                # 1. Apply Intervention
                new_text = method_func(turn_text)
                
                # 2. Quality Gate
                fidelity = check_fidelity(turn_text, new_text)
                
                if fidelity >= BERTSCORE_THRESHOLD:
                    new_turns.append(new_text)
                else:
                    # Fallback to original if intervention failed quality check
                    new_turns.append(turn_text)
            
            new_choice = choice.copy()
            new_choice['turns'] = new_turns
            new_choices.append(new_choice)
        
        item['choices'] = new_choices
        item['model_id'] = item['model_id'] + f"_{method_name}"
        new_records.append(item)

    with open(output_path, 'w') as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")

def main():
    setup_directories()
    
    for fpath in INPUT_FILES:
        # Run Technique A
        process_file(fpath, "neutral", rewrite_neutral)
        
        # Run Technique B
        process_file(fpath, "backtrans", back_translate)
        
    print("\nStep 3 Complete. Generated Neutral and Back-Translated files.")

if __name__ == "__main__":
    main()