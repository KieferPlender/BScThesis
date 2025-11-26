import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import html 
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect, LangDetectException

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_MODELS = [
    'vicuna-13b', 'koala-13b', 'oasst-pythia-12b', 'gpt-3.5-turbo', 
    'alpaca-13b', 'gpt-4', 'claude-v1', 'RWKV-4-Raven-14B', 
    'chatglm-6b', 'palm-2'
]
OUTPUT_DIR = 'figures/step_0'
SAMPLE_SIZE = 3000  # Target samples per model (will limit to min class size)

# Function Word Categories
FUNCTION_WORDS = {
    'Pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
    'Prepositions': ['in', 'on', 'at', 'to', 'from', 'with', 'by', 'of', 'for', 'about'],
    'Articles_Determiners': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'their'],
    'Conjunctions': ['and', 'or', 'but', 'because', 'although', 'while', 'if', 'when', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', 'thus'],
    'Auxiliary_Verbs': ['be', 'am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 
                        'will', 'would', 'can', 'could', 'should', 'may', 'might'],
    'Particles': ['not', 'up', 'off', 'out'], 
    'Quantifiers': ['some', 'many', 'few', 'all', 'each']
}

# Flatten the list for easy iteration
ALL_FUNCTION_WORDS = [word for category in FUNCTION_WORDS.values() for word in category]

# Universal POS Tags
ALL_POS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
]

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# ==========================================
# DATA CLEANING & LOADING
# ==========================================

def clean_text(text):
    if not isinstance(text, str): return ""
    
    # 1. Unescape HTML entities (converts &lt;div&gt; to <div>)
    text = html.unescape(text)
    
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Explicitly kill the artifact if it survives
    text = re.sub(r'\bdiv\s+div\b', ' ', text, flags=re.IGNORECASE)
    
    # 4. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def load_and_filter_data():
    print("Loading dataset...")
    # Load dataset from HuggingFace
    try:
        dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame() # Return empty if fails
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)
    
    print("Flattening dataset...")
    rows = []
    
    # We use a progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Flattening"):
        # Model A
        if row['model_a'] in TARGET_MODELS:
            for turn in row['conversation_a']:
                if turn['role'] == 'assistant':
                    cleaned = clean_text(turn['content'])
                    if cleaned:
                        rows.append({'model': row['model_a'], 'text': cleaned})
        
        # Model B
        if row['model_b'] in TARGET_MODELS:
            for turn in row['conversation_b']:
                if turn['role'] == 'assistant':
                    cleaned = clean_text(turn['content'])
                    if cleaned:
                        rows.append({'model': row['model_b'], 'text': cleaned})
                    
    df_filtered = pd.DataFrame(rows)
    print(f"Total responses for Top 10 models (Unfiltered): {len(df_filtered)}")
    
    # Filter for English only
    print("Filtering for English text (this make take a few minutes)...")
    
    tqdm.pandas(desc="Detecting Language")
    df_filtered = df_filtered[df_filtered['text'].progress_apply(is_english)]
    print(f"English responses: {len(df_filtered)}")
    
    # Balance the dataset
    min_count = df_filtered['model'].value_counts().min()
    print(f"Smallest class size: {min_count}")
    
    # Use the smaller of min_count or our target SAMPLE_SIZE
    final_n = min(min_count, SAMPLE_SIZE)
    
    print(f"Balancing dataset to {final_n} samples per model...")
    df_balanced = df_filtered.groupby('model').apply(lambda x: x.sample(final_n, random_state=42)).reset_index(drop=True)
    
    print(f"Final Balanced dataset size: {len(df_balanced)}")
    return df_balanced

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def analyze_style(df):
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    results = []
    
    print("Analyzing text features (POS & Stylometry)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc = nlp(row['text'])
        total_tokens = len(doc)
        
        if total_tokens < 5: continue # Skip extremely short junk responses
        
        # 1. POS Counts (Normalized per 1000 tokens)
        pos_counts = Counter([token.pos_ for token in doc])
        pos_data = {}
        for tag in ALL_POS_TAGS:
            pos_data[f'pos_{tag}'] = (pos_counts.get(tag, 0) / total_tokens) * 1000
        
        # 2. Lexical Diversity
        unique_tokens = len(set([token.text.lower() for token in doc]))
        ttr = unique_tokens / total_tokens 
        avg_sent_len = total_tokens / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0
        
        # 3. Function Words (Normalized per 1000 tokens)
        text_lower = row['text'].lower()
        token_texts = [token.text.lower() for token in doc]
        token_counts = Counter(token_texts)
        
        fw_data = {}
        for fw in ALL_FUNCTION_WORDS:
            # Normalization Formula Applied
            raw_count = token_counts[fw]
            normalized_freq = (raw_count / total_tokens) * 1000
            fw_data[fw] = normalized_freq
        
        entry = {
            'model': row['model'],
            'ttr': ttr,
            'avg_sent_len': avg_sent_len
        }
        entry.update(pos_data)
        entry.update(fw_data)
        results.append(entry)
        
    return pd.DataFrame(results)

def analyze_ngrams(df):
    print("Analyzing N-grams...")
    model_texts = df.groupby('model')['text'].apply(lambda x: " ".join(x)).to_dict()
    
    ngram_results = {}
    
    for model, text in model_texts.items():
        # Unigrams (Top 20)
        vec_uni = CountVectorizer(ngram_range=(1, 1), max_features=20, stop_words='english')
        X_uni = vec_uni.fit_transform([text])
        freq_uni = zip(vec_uni.get_feature_names_out(), X_uni.toarray()[0])
        ngram_results[f'{model}_unigrams'] = sorted(freq_uni, key=lambda x: x[1], reverse=True)
        
        # Bigrams (Top 20)
        vec_bi = CountVectorizer(ngram_range=(2, 2), max_features=20)
        X_bi = vec_bi.fit_transform([text])
        freq_bi = zip(vec_bi.get_feature_names_out(), X_bi.toarray()[0])
        ngram_results[f'{model}_bigrams'] = sorted(freq_bi, key=lambda x: x[1], reverse=True)
        
    return ngram_results

# ==========================================
# PLOTTING & SAVING
# ==========================================

def plot_ngrams(ngram_results):
    print("Saving N-gram text report...")
    with open(f'{OUTPUT_DIR}/top_ngrams_report.txt', 'w') as f:
        for key, val in ngram_results.items():
            f.write(f"\n--- {key} ---\n")
            for word, count in val:
                f.write(f"{word}: {count}\n")
    print(f"N-gram analysis saved to {OUTPUT_DIR}/top_ngrams_report.txt")

def plot_results(df_stats):
    # Set theme
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['figure.dpi'] = 300 
    
    # 1. POS Distribution Plots
    print("Generating POS plots...")
    
    groups = {
        'Content': ['pos_NOUN', 'pos_VERB', 'pos_ADJ', 'pos_ADV', 'pos_PROPN'],
        'Function': ['pos_DET', 'pos_ADP', 'pos_PRON', 'pos_AUX', 'pos_CCONJ'],
        'Misc': ['pos_PUNCT', 'pos_NUM', 'pos_SYM']
    }
    
    for group_name, tags in groups.items():
        plt.figure(figsize=(14, 8))
        valid_tags = [t for t in tags if t in df_stats.columns]
        
        pos_melted = pd.melt(df_stats, id_vars=['model'], value_vars=valid_tags, 
                           var_name='POS Type', value_name='Frequency')
        pos_melted['POS Type'] = pos_melted['POS Type'].str.replace('pos_', '')
        
        sns.boxplot(x='POS Type', y='Frequency', hue='model', data=pos_melted, showfliers=False)
        plt.title(f'POS Distribution: {group_name} Words (per 1000 tokens)', fontweight='bold')
        plt.ylabel('Frequency (per 1k tokens)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/pos_{group_name.lower()}.png', bbox_inches='tight')
        plt.close()
    
    # 2. Complexity Plots
    print("Generating Complexity plots...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    sns.boxplot(x='model', y='ttr', data=df_stats, ax=axes[0], showfliers=False)
    axes[0].set_title('Lexical Diversity (Type-Token Ratio)', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(x='model', y='avg_sent_len', data=df_stats, ax=axes[1], showfliers=False)
    axes[1].set_title('Syntactic Complexity (Avg Sentence Length)', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/complexity_metrics.png', bbox_inches='tight')
    plt.close()
    
    # 3. Function Words
    print("Generating Function Word plots...")
    for category, words in FUNCTION_WORDS.items():
        valid_words = [w for w in words if w in df_stats.columns]
        if not valid_words: continue
        
        fw_melted = pd.melt(df_stats, id_vars=['model'], value_vars=valid_words, 
                            var_name='Word', value_name='Frequency')
        plt.figure(figsize=(16, 8))
        sns.barplot(x='Word', y='Frequency', hue='model', data=fw_melted, 
                    palette="tab10", errorbar=('ci', 95))
        plt.title(f'Function Word Usage: {category} (per 1000 tokens)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fw_{category}.png', bbox_inches='tight')
        plt.close()

def save_statistics(df_stats):
    print("Saving summary statistics CSV...")
    # Calculate numeric summary
    numeric_cols = df_stats.select_dtypes(include=[np.number]).columns
    summary = df_stats.groupby('model')[numeric_cols].agg(['mean', 'std'])
    summary.to_csv(f'{OUTPUT_DIR}/summary_stats_normalized.csv')
    
    # === NEW ADDITION FOR STEP 1 ===
    print("Saving FULL feature dataset for Step 1...")
    df_stats.to_csv(f'{OUTPUT_DIR}/full_feature_dataset.csv', index=False)
    print(f"Saved {len(df_stats)} rows to {OUTPUT_DIR}/full_feature_dataset.csv")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    setup_directories()
    
    # 1. Load Data
    df_data = load_and_filter_data()
    if df_data.empty:
        print("No data loaded. Exiting.")
        return

    # 2. Extract Features (Normalized)
    df_stats = analyze_style(df_data)
    
    # 3. N-gram Analysis
    ngram_results = analyze_ngrams(df_data)
    plot_ngrams(ngram_results)
    
    # 4. Plotting & Saving
    plot_results(df_stats)
    save_statistics(df_stats)
    
    print(f"Step 0 Analysis Complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()