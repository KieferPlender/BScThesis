from dotenv import load_dotenv  
load_dotenv()

import json
import os
import time
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'data/mt_bench/question.jsonl'
OUTPUT_DIR = 'data/mt_bench/model_answers'

# Define your "Contestants" and where they live
MODELS = {
    "gpt-4-turbo": {
        "type": "cloud", 
        "api_key": os.environ.get("OPENAI_API_KEY"), 
        "base_url": None 
    },
    "gpt-3.5-turbo": {
        "type": "cloud",
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": None
    },
    "vicuna:13b": {  # This matches the Ollama tag
        "type": "local",
        "api_key": "ollama",        # Ollama doesn't care about keys
        "base_url": "http://localhost:11434/v1" # Your local Mac server
    }
}

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_answer(model_name, config, messages, max_tokens=1024):
    # Create a specific client for this model (Cloud vs Local)
    try:
        if config["type"] == "local":
            # Local Client (Ollama)
            client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
        else:
            # Cloud Client (OpenAI)
            client = OpenAI(api_key=config["api_key"])

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
            seed=42
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating for {model_name}: {e}")
        return None

def run_generation():
    setup_directories()
    
    # 1. Load Questions
    questions = []
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                questions.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: Could not find {INPUT_FILE}")
        print("Did you save the jsonl text I gave you into 'data/mt_bench/question.jsonl'?")
        return
            
    print(f"Loaded {len(questions)} questions from MT-Bench.")

    # 2. Loop through each Model
    for model_name, config in MODELS.items():
        print(f"\n🤖 Generating answers for: {model_name} ({config['type']})...")
        
        # File naming: replace ':' with '_' for Vicuna filename
        safe_name = model_name.replace(":", "_")
        output_file = f"{OUTPUT_DIR}/{safe_name}.jsonl"
        
        if os.path.exists(output_file):
            print(f"   Skipping (File exists: {output_file})")
            continue

        answers = []
        for q in tqdm(questions):
            # MT-Bench has 2 turns per question
            conversation_history = []
            choices = []
            
            # --- TURN 1 ---
            prompt_1 = q['turns'][0]
            conversation_history.append({"role": "user", "content": prompt_1})
            
            ans_1 = get_answer(model_name, config, conversation_history)
            if ans_1 is None: break # Skip if error
            
            conversation_history.append({"role": "assistant", "content": ans_1})
            choices.append({"index": 0, "turns": [ans_1], "model": model_name})
            
            # --- TURN 2 ---
            prompt_2 = q['turns'][1]
            conversation_history.append({"role": "user", "content": prompt_2})
            
            ans_2 = get_answer(model_name, config, conversation_history)
            if ans_2 is None: break
            
            choices[0]['turns'].append(ans_2)
            
            # Save record
            record = {
                "question_id": q["question_id"],
                "category": q["category"],
                "model_id": model_name,
                "choices": choices,
                "tstamp": time.time()
            }
            answers.append(record)

        # 3. Save to File
        if answers:
            with open(output_file, 'w') as f:
                for ans in answers:
                    f.write(json.dumps(ans) + "\n")
            print(f"Saved {len(answers)} answers to {output_file}")

if __name__ == "__main__":
    run_generation()
