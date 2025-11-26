import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import html
from tqdm import tqdm
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scipy.sparse import hstack
import os

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_MODELS = [
    'vicuna-13b', 'koala-13b', 'oasst-pythia-12b', 'gpt-3.5-turbo', 
    'alpaca-13b', 'gpt-4', 'claude-v1', 'RWKV-4-Raven-14B', 
    'chatglm-6b', 'palm-2'
]
OUTPUT_DIR = 'figures/step_1'
TARGET_SAMPLE_SIZE = 2900 

# Define your Families here
FAMILY_MAPPING = {
    # Family A: Llama-1 Derivatives (The "Open Source Siblings")
    'vicuna-13b': 'Llama-1 Family',
    'koala-13b': 'Llama-1 Family',
    'alpaca-13b': 'Llama-1 Family',
    
    # Family B: OpenAI (The "RLHF Siblings")
    'gpt-4': 'OpenAI Family',
    'gpt-3.5-turbo': 'OpenAI Family',
    
    # Family C: Others (Distinct Architectures)
    'claude-v1': 'Anthropic/Others',
    'palm-2': 'Anthropic/Others',
    'chatglm-6b': 'Anthropic/Others',
    'RWKV-4-Raven-14B': 'Anthropic/Others',
    'oasst-pythia-12b': 'Anthropic/Others' # Pythia is distinct from Llama
}

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\bdiv\s+div\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_balance_data():
    print("Loading dataset...")
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    df = pd.DataFrame(dataset)
    all_samples = {m: [] for m in TARGET_MODELS}
    
    print("Extracting and cleaning texts...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        for key in ['model_a', 'model_b']:
            model = row[key]
            if model in TARGET_MODELS:
                if len(row[f'conversation_{key[-1]}']) > 1:
                    raw_text = row[f'conversation_{key[-1]}'][1]['content']
                    cleaned = clean_text(raw_text)
                    if len(cleaned) > 50: 
                        all_samples[model].append(cleaned)

    balanced_data = []
    min_available = min([len(v) for k, v in all_samples.items()])
    final_n = min(min_available, TARGET_SAMPLE_SIZE)
    
    print(f"\nBalancing classes to N={final_n} per model...")
    for model, texts in all_samples.items():
        selected = np.random.choice(texts, final_n, replace=False)
        for t in selected:
            balanced_data.append({'model': model, 'text': t})
            
    return pd.DataFrame(balanced_data)

# ==========================================
# 2. FEATURE EXTRACTION
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
        # Explicit string conversion
        X_str = [str(x) for x in X]
        pos_sequences = []
        for doc in tqdm(self.nlp.pipe(X_str, batch_size=1000), total=len(X)):
            pos_seq = " ".join([token.pos_ for token in doc])
            pos_sequences.append(pos_seq)
        return pos_sequences

def extract_features(df):
    print("\nExtracting N-gram Features...")
    texts = [str(t) for t in df['text'].tolist()]
    
    print("- Vectorizing Word N-grams (2-4)...")
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 4), max_features=2000, min_df=5)
    X_word = word_vectorizer.fit_transform(texts)
    
    print("- Vectorizing Character N-grams (3-5)...")
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=2000, min_df=5)
    X_char = char_vectorizer.fit_transform(texts)
    
    pos_transformer = POSTaggingTransformer()
    pos_texts = pos_transformer.transform(texts)
    print("- Vectorizing POS N-grams (2-4)...")
    pos_vectorizer = TfidfVectorizer(ngram_range=(2, 4), max_features=1000)
    X_pos = pos_vectorizer.fit_transform(pos_texts)
    
    print("Combining Feature Sets...")
    X_combined = hstack([X_word, X_char, X_pos])
    return X_combined, word_vectorizer, char_vectorizer, pos_vectorizer

# ==========================================
# 3. CLASSIFICATION & FAMILY ANALYSIS
# ==========================================
def train_and_evaluate(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    print(f"\nTraining Classifier on {X_train.shape[0]} samples...")
    clf = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, eval_metric='mlogloss', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # --- STANDARD ACCURACY ---
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*40)
    print(f"✅ MODEL ACCURACY (Specific): {acc:.2%}")
    print("="*40)
    
    # --- FAMILY ACCURACY (THE NEW PART) ---
    print("\nCalculating Family-Level Accuracy...")
    
    # 1. Decode back to names (e.g., 0 -> 'alpaca')
    y_test_names = le.inverse_transform(y_test)
    y_pred_names = le.inverse_transform(y_pred)
    
    # 2. Map to Families
    y_test_fam = [FAMILY_MAPPING[m] for m in y_test_names]
    y_pred_fam = [FAMILY_MAPPING[m] for m in y_pred_names]
    
    # 3. Calculate New Score
    fam_acc = accuracy_score(y_test_fam, y_pred_fam)
    print("\n" + "="*40)
    print(f"👨‍👩‍👧‍👦 FAMILY ACCURACY: {fam_acc:.2%}")
    print("="*40)
    
    # 4. Plot Family Confusion Matrix
    labels = sorted(list(set(FAMILY_MAPPING.values())))
    cm = confusion_matrix(y_test_fam, y_pred_fam, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: Model Families\n(Family Accuracy: {fam_acc:.2%})', fontsize=14)
    plt.ylabel('True Family')
    plt.xlabel('Predicted Family')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_families.png', dpi=300)
    print(f"Family Matrix saved to {OUTPUT_DIR}/confusion_matrix_families.png")
    
    return clf

def main():
    setup_directories()
    df = load_and_balance_data()
    X, _, _, _ = extract_features(df)
    train_and_evaluate(X, df['model'])
    print(f"Step 1 Complete.")

if __name__ == "__main__":
    main()