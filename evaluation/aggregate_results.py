import json
import pandas as pd
import glob
import os
from pathlib import Path


def load_experiment_results(results_dir):
    json_files = glob.glob(f"{results_dir}/*.json")
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return None
    
    print(f"Found {len(json_files)} result files")
    
    results = []
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = data.get('config', {})
            metrics = data.get('metrics', {})
            
            row = {
                'strategy': config.get('strategy'),
                'top_k': config.get('top_k'),
                'search_type': config.get('search_type'),
                'threshold': config.get('threshold'),
                'total_questions': config.get('total_questions'),
            }
            
            if 'retriever' in metrics:
                retriever = metrics['retriever']
                row.update({
                    'hit_rate': retriever.get('hit_rate'),
                    'avg_recall': retriever.get('avg_recall'),
                    'avg_ndcg': retriever.get('avg_ndcg'),
                    'mrr': retriever.get('mrr'),
                    'map': retriever.get('map')
                })
            
            if 'generator' in metrics:
                generator = metrics['generator']
                row.update({
                    'avg_faithfulness': generator.get('avg_faithfulness'),
                    'avg_key_facts_coverage': generator.get('avg_key_facts_coverage'),
                    'avg_answer_relevance': generator.get('avg_answer_relevance')
                })
            
            if 'end_to_end' in metrics:
                e2e = metrics['end_to_end']
                row.update({
                    'avg_context_recall': e2e.get('avg_context_recall'),
                    'avg_answer_correctness': e2e.get('avg_answer_correctness'),
                    'avg_latency_ms': e2e.get('avg_latency_ms'),
                    'min_latency_ms': e2e.get('min_latency_ms'),
                    'max_latency_ms': e2e.get('max_latency_ms'),
                    'p95_latency_ms': e2e.get('p95_latency_ms'),
                    'p99_latency_ms': e2e.get('p99_latency_ms'),
                    'estimated_cost_per_query': e2e.get('estimated_cost_per_query')
                })
            
            row['source_file'] = os.path.basename(json_file)
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    column_order = [
        'strategy', 'top_k', 'search_type', 'threshold',
        'hit_rate', 'avg_recall', 'avg_ndcg', 'mrr', 'map',
        'avg_faithfulness', 'avg_key_facts_coverage', 'avg_answer_relevance',
        'avg_context_recall', 'avg_answer_correctness',
        'avg_latency_ms', 'min_latency_ms', 'max_latency_ms', 
        'p95_latency_ms', 'p99_latency_ms',
        'estimated_cost_per_query', 'total_questions', 'source_file'
    ]
    
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]
    
    return df


def save_aggregated_results(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"\nSaved aggregated results to: {output_file}")
    print(f"Total experiments: {len(df)}")


def print_summary(df):
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\nTotal Experiments: {len(df)}")
    print(f"\nStrategies tested: {df['strategy'].nunique()}")
    print(df['strategy'].value_counts().to_string())
    
    print(f"\nTop-k values tested: {sorted(df['top_k'].unique())}")
    
    if 'search_type' in df.columns:
        print(f"\nSearch types tested: {df['search_type'].unique()}")
    
    print("\n" + "="*80)
    print("METRIC COVERAGE")
    print("="*80)
    
    metric_cols = [
        'hit_rate', 'avg_recall', 'avg_ndcg', 'mrr', 'map',
        'avg_faithfulness', 'avg_key_facts_coverage', 'avg_answer_relevance',
        'avg_context_recall', 'avg_answer_correctness',
        'avg_latency_ms', 'estimated_cost_per_query'
    ]
    
    for col in metric_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"{col:30s}: {non_null}/{len(df)} experiments")
    
    print("\n" + "="*80)
    print("TOP PERFORMERS BY METRIC")
    print("="*80)
    
    if 'hit_rate' in df.columns and df['hit_rate'].notna().any():
        best = df.loc[df['hit_rate'].idxmax()]
        print(f"\nBest Hit Rate: {best['hit_rate']:.3f}")
        print(f"  Strategy: {best['strategy']}, top_k: {best['top_k']}")
    
    if 'avg_ndcg' in df.columns and df['avg_ndcg'].notna().any():
        best = df.loc[df['avg_ndcg'].idxmax()]
        print(f"\nBest NDCG: {best['avg_ndcg']:.3f}")
        print(f"  Strategy: {best['strategy']}, top_k: {best['top_k']}")
    
    if 'avg_faithfulness' in df.columns and df['avg_faithfulness'].notna().any():
        best = df.loc[df['avg_faithfulness'].idxmax()]
        print(f"\nBest Faithfulness: {best['avg_faithfulness']:.3f}")
        print(f"  Strategy: {best['strategy']}, top_k: {best['top_k']}")
    
    if 'avg_answer_correctness' in df.columns and df['avg_answer_correctness'].notna().any():
        best = df.loc[df['avg_answer_correctness'].idxmax()]
        print(f"\nBest Answer Correctness: {best['avg_answer_correctness']:.3f}")
        print(f"  Strategy: {best['strategy']}, top_k: {best['top_k']}")
    
    if 'avg_latency_ms' in df.columns and df['avg_latency_ms'].notna().any():
        best = df.loc[df['avg_latency_ms'].idxmin()]
        print(f"\nBest (Lowest) Latency: {best['avg_latency_ms']:.2f}ms")
        print(f"  Strategy: {best['strategy']}, top_k: {best['top_k']}")
    
    print("\n" + "="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--results_dir', default='experiments/results_hybrid', 
                       help='Directory containing result JSON files')
    parser.add_argument('--output', default='experiments/combined_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--summary', action='store_true',
                       help='Print detailed summary')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    df = load_experiment_results(args.results_dir)
    
    if df is None or len(df) == 0:
        print("No results to aggregate")
        return
    
    save_aggregated_results(df, args.output)
    
    if args.summary:
        print_summary(df)
    else:
        print("\nQuick Summary:")
        print(df[['strategy', 'top_k', 'hit_rate', 'avg_ndcg', 'avg_faithfulness']].to_string())
        print(f"\nRun with --summary flag for detailed analysis")


if __name__ == "__main__":
    main()