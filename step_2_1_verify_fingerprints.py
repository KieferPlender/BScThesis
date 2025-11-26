import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import html
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scipy.sparse import hstack

# ==========================================
# CONFIGURATION
# ==========================================
# Where your new answers are
MT_BENCH_DIR = 'data/mt_bench/model_answers'
OUTPUT_DIR = 'figures/step_2_1'

# Mapping: Filename (from Step 2) -> Training Label (from Step 1 LMSYS)
NAME_MAPPING = {
    'gpt-4-turbo': 'gpt-4',          # MATCHES FILE: gpt-4-turbo.jsonl
    'gpt-3.5-turbo': 'gpt-3.5-turbo', # MATCHES FILE: gpt-3.5-turbo.jsonl
    'vicuna_13b': 'vicuna-13b'       # MATCHES FILE: vicuna_13b.jsonl
}

# Family Definitions (Same as Step 1)
FAMILY_MAPPING = {
    'vicuna-13b': 'Llama-1 Family',
    'gpt-4': 'OpenAI Family',
    'gpt-3.5-turbo': 'OpenAI Family'
}

# Target Models for Training (We only need to train on models relevant to the experiment + some noise)
TRAINING_MODELS = [
    'vicuna-13b', 'koala-13b', 'alpaca-13b', # Llama Family
    'gpt-4', 'gpt-3.5-turbo',                # OpenAI Family
    'claude-v1', 'palm-2'                    # Distractors to keep classifier robust
]

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# ==========================================
# DATA CLEANING (MUST MATCH STEP 0 EXACTLY)
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\bdiv\s+div\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# LOADERS
# ==========================================
def load_training_data():
    print("Loading LMSYS Training Data (The 'Expert' Knowledge)...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    data = []
    # Train on 2000 samples per model to be fast but accurate
    target_n = 2000 
    
    # Group by model first
    temp_storage = {m: [] for m in TRAINING_MODELS}
    
    for row in tqdm(dataset, desc="Scanning LMSYS"):
        for key in ['model_a', 'model_b']:
            model = row[key]
            if model in TRAINING_MODELS:
                if len(temp_storage[model]) < target_n:
                    if len(row[f'conversation_{key[-1]}']) > 1:
                        raw = row[f'conversation_{key[-1]}'][1]['content']
                        cleaned = clean_text(raw)
                        if len(cleaned) > 50:
                            temp_storage[model].append(cleaned)
                            
    # Flatten
    for model, texts in temp_storage.items():
        for t in texts:
            data.append({'text': t, 'model': model, 'type': 'TRAIN'})
            
    return pd.DataFrame(data)

def load_mt_bench_data():
    print("\nLoading MT-Bench Generated Answers (The 'Test Subjects')...")
    data = []
    
    for filename in os.listdir(MT_BENCH_DIR):
        name_key = filename.replace('.jsonl', '')
        
        if name_key in NAME_MAPPING:
            train_label = NAME_MAPPING[name_key]
            print(f"Found {filename} -> Mapped to '{train_label}'")
            
            path = os.path.join(MT_BENCH_DIR, filename)
            with open(path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    # Extract all turns from the assistant
                    for choice in item['choices']:
                        for turn_text in choice['turns']:
                            cleaned = clean_text(turn_text)
                            if len(cleaned) > 20:
                                data.append({
                                    'text': cleaned,
                                    'model': train_label,
                                    'type': 'TEST'
                                })
    return pd.DataFrame(data)

# ==========================================
# FEATURE PIPELINE
# ==========================================
class POSTaggingTransformer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
    def fit(self, X, y=None): return self
    def transform(self, X):
        print("Generating POS Tags...")
        pos_sequences = []
        for doc in tqdm(self.nlp.pipe(X, batch_size=1000), total=len(X)):
            pos_seq = " ".join([token.pos_ for token in doc])
            pos_sequences.append(pos_seq)
        return pos_sequences

def extract_features(df_train, df_test):
    print("\nVectorizing Features (Fitting on TRAIN, Transforming TEST)...")
    
    # Combine for text processing convenience (but fit only on train)
    train_texts = [str(t) for t in df_train['text'].tolist()]
    test_texts = [str(t) for t in df_test['text'].tolist()]
    
    # 1. Words
    print("- Word N-Grams...")
    vec_word = TfidfVectorizer(ngram_range=(2, 4), max_features=2000, min_df=5)
    X_word_train = vec_word.fit_transform(train_texts)
    X_word_test = vec_word.transform(test_texts) # CRITICAL: Transform only
    
    # 2. Chars
    print("- Char N-Grams...")
    vec_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=2000, min_df=5)
    X_char_train = vec_char.fit_transform(train_texts)
    X_char_test = vec_char.transform(test_texts)
    
    # 3. POS
    print("- POS N-Grams...")
    pos = POSTaggingTransformer()
    # We process all texts to get POS strings
    pos_train_str = pos.transform(train_texts)
    pos_test_str = pos.transform(test_texts)
    
    vec_pos = TfidfVectorizer(ngram_range=(2, 4), max_features=1000)
    X_pos_train = vec_pos.fit_transform(pos_train_str)
    X_pos_test = vec_pos.transform(pos_test_str)
    
    # Stack
    X_train = hstack([X_word_train, X_char_train, X_pos_train])
    X_test = hstack([X_word_test, X_char_test, X_pos_test])
    
    return X_train, X_test

# ==========================================
# EXECUTION
# ==========================================
def main():
    setup_directories()
    
    # 1. Load
    df_train = load_training_data()
    df_test = load_mt_bench_data()
    print(f"\nTraining Samples: {len(df_train)}")
    print(f"Testing Samples (MT-Bench): {len(df_test)}")
    
    # 2. Features
    X_train, X_test = extract_features(df_train, df_test)
    
    # 3. Train
    print(f"\nTraining XGBoost on {len(df_train)} samples...")
    # Need to encode labels based on TRAIN set
    le = LabelEncoder()
    y_train_enc = le.fit_transform(df_train['model'])
    
    clf = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, n_jobs=-1)
    clf.fit(X_train, y_train_enc)
    
    # 4. Predict on NEW Data
    print("Predicting on MT-Bench Data...")
    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    
    # 5. Family Analysis
    print("\nChecking Family Fingerprints...")
    y_true_fam = [FAMILY_MAPPING[m] for m in df_test['model']]
    y_pred_fam = []
    
    # Handle prediction labels that might not be in FAMILY_MAPPING (e.g. 'claude' predicted)
    for p in y_pred:
        if p in FAMILY_MAPPING:
            y_pred_fam.append(FAMILY_MAPPING[p])
        else:
            y_pred_fam.append("Other")
            
    acc = accuracy_score(y_true_fam, y_pred_fam)
    
    print("\n" + "="*50)
    print(f"🔬 FINGERPRINT SURVIVAL RATE (Family Accuracy): {acc:.2%}")
    print("="*50)
    
    # Save Matrix
    labels = sorted(list(set(y_true_fam)))
    cm = confusion_matrix(y_true_fam, y_pred_fam, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title(f'MT-Bench Fingerprint Verification\nAccuracy: {acc:.2%}')
    plt.ylabel('True Source (MT-Bench)')
    plt.xlabel('Predicted Style (LMSYS)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/mt_bench_verification.png')
    
    if acc > 0.50:
        print("\n✅ SUCCESS: Fingerprints survived the transition to MT-Bench.")
        print("   You may proceed to Step 2.2 (The Judge).")
    else:
        print("\n⚠️ WARNING: Accuracy is low. The styles might be too similar in the new dataset.")

if __name__ == "__main__":
    main()