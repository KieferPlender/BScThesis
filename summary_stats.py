import pandas as pd
from step0_eda import load_and_filter_data, analyze_style

def print_summary():
    print("Loading and analyzing data for summary...")
    df_data = load_and_filter_data()
    df_stats = analyze_style(df_data)
    
    print("\n--- Summary Statistics by Model ---")
    # Group by model and calculate mean for all numeric columns
    summary = df_stats.groupby('model').mean()
    print(summary.to_string())
    
    print("\n--- Standard Deviation ---")
    std_dev = df_stats.groupby('model').std()
    print(std_dev.to_string())

if __name__ == "__main__":
    print_summary()
