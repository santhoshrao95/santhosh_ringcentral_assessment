import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Dict
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

def load_ground_truth(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['all_questions']


def call_search_api(query, strategy, top_k, retrieve_only, search_type, base_url):
    url = f"{base_url}/search"
    payload = {
        "query": query,
        "strategy": strategy,
        "retrieve_only": retrieve_only,
        "top_k": top_k,
        "search_type": search_type
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def is_relevant_chunk(retrieved_chunk_text, ground_truth_chunks, embedding_model, threshold):
    retrieved_emb = embedding_model.encode(retrieved_chunk_text, convert_to_tensor=True)
    
    for gt_chunk in ground_truth_chunks:
        gt_emb = embedding_model.encode(gt_chunk, convert_to_tensor=True)
        similarity = util.cos_sim(retrieved_emb, gt_emb).item()
        
        if similarity >= threshold:
            return True
    
    return False


def calculate_hit_rate(results, ground_truths, embedding_model, threshold, top_k):
    hits = 0
    for result, gt in zip(results, ground_truths):
        retrieved_chunks = result.get('retrieved_chunks', [])[:top_k]
        gt_chunks = gt['relevant_chunks']
        
        for chunk in retrieved_chunks:
            if is_relevant_chunk(chunk['text'], gt_chunks, embedding_model, threshold):
                hits += 1
                break
    
    return hits / len(results) if results else 0.0


def calculate_recall(results, ground_truths, embedding_model, threshold, top_k):
    recalls = []
    for result, gt in zip(results, ground_truths):
        retrieved_chunks = result.get('retrieved_chunks', [])[:top_k]
        gt_chunks = gt['relevant_chunks']
        
        if not gt_chunks:
            continue
        
        relevant_found = 0
        for chunk in retrieved_chunks:
            if is_relevant_chunk(chunk['text'], gt_chunks, embedding_model, threshold):
                relevant_found += 1
        
        recall = relevant_found / len(gt_chunks)
        recalls.append(recall)
    
    return sum(recalls) / len(recalls) if recalls else 0.0


def calculate_ndcg(results, ground_truths, embedding_model, threshold, top_k):
    ndcg_scores = []
    for result, gt in zip(results, ground_truths):
        retrieved_chunks = result.get('retrieved_chunks', [])[:top_k]
        gt_chunks = gt['relevant_chunks']
        
        relevance_scores = []
        for chunk in retrieved_chunks:
            if is_relevant_chunk(chunk['text'], gt_chunks, embedding_model, threshold):
                relevance_scores.append(1)
            else:
                relevance_scores.append(0)
        
        if not relevance_scores or sum(relevance_scores) == 0:
            ndcg_scores.append(0.0)
            continue
        
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
        
        ideal_relevances = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def calculate_mrr(results, ground_truths, embedding_model, threshold, top_k):
    rr_scores = []
    for result, gt in zip(results, ground_truths):
        retrieved_chunks = result.get('retrieved_chunks', [])[:top_k]
        gt_chunks = gt['relevant_chunks']
        
        rr = 0.0
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if is_relevant_chunk(chunk['text'], gt_chunks, embedding_model, threshold):
                rr = 1.0 / rank
                break
        
        rr_scores.append(rr)
    
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


def calculate_map(results, ground_truths, embedding_model, threshold, top_k):
    ap_scores = []
    for result, gt in zip(results, ground_truths):
        retrieved_chunks = result.get('retrieved_chunks', [])[:top_k]
        gt_chunks = gt['relevant_chunks']
        
        if not gt_chunks:
            continue
        
        relevant_count = 0
        precision_sum = 0.0
        
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if is_relevant_chunk(chunk['text'], gt_chunks, embedding_model, threshold):
                relevant_count += 1
                precision_at_k = relevant_count / rank
                precision_sum += precision_at_k
        
        ap = precision_sum / len(gt_chunks) if gt_chunks else 0.0
        ap_scores.append(ap)
    
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


def calculate_faithfulness(answer, chunks, groq_client):
    context = "\n".join([chunk['text'] for chunk in chunks])
    
    prompt = f"""Context: {context}

Answer: {answer}

Is the answer factually consistent with and supported by the context? 
Consider:
- Are all facts in the answer present in the context?
- Does the answer contradict the context?
- Does the answer add information not in the context?

Answer only: YES or NO"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a factual consistency checker."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        answer_text = response.choices[0].message.content.strip().upper()
        print(answer_text)
        return 1.0 if "YES" in answer_text else 0.0
    except:
        return None


def calculate_key_facts_coverage(answer, key_facts):
    generated_lower = answer.lower()
    facts_present = 0
    
    for key_fact in key_facts:
        if key_fact.lower() in generated_lower:
            facts_present += 1
    
    return facts_present / len(key_facts) if key_facts else 0.0


def calculate_answer_relevance(answer, query, embedding_model):
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    answer_emb = embedding_model.encode(answer, convert_to_tensor=True)
    
    relevance = util.cos_sim(query_emb, answer_emb).item()
    return relevance


def calculate_context_recall(query, chunks, ground_truth_answer, groq_client):
    context = "\n".join([chunk['text'] for chunk in chunks])
    
    prompt = f"""Question: {query}

Retrieved Context: {context}

Ground Truth Answer: {ground_truth_answer}

Can the ground truth answer be fully generated using only the information in the retrieved context?

Answer only: YES or NO"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are evaluating context completeness."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        answer_text = response.choices[0].message.content.strip().upper()
        print("Answer correct")
        return 1.0 if "YES" in answer_text else 0.0
    except:
        return None


def calculate_f1_score(generated, reference):
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    
    common = gen_tokens & ref_tokens
    
    if len(gen_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_answer_correctness(generated_answer, ground_truth_answer, embedding_model):
    f1 = calculate_f1_score(generated_answer, ground_truth_answer)
    
    gen_emb = embedding_model.encode(generated_answer, convert_to_tensor=True)
    ref_emb = embedding_model.encode(ground_truth_answer, convert_to_tensor=True)
    
    semantic_sim = util.cos_sim(gen_emb, ref_emb).item()
    
    correctness = 0.5 * f1 + 0.5 * semantic_sim
    return correctness, f1, semantic_sim


def estimate_cost(answer, chunks):
    def estimate_tokens(text):
        return len(text) / 4
    
    context_tokens = sum(estimate_tokens(chunk['text']) for chunk in chunks)
    answer_tokens = estimate_tokens(answer)
    
    input_cost = (context_tokens / 1_000_000) * 0.05
    output_cost = (answer_tokens / 1_000_000) * 0.08
    
    return input_cost + output_cost


def save_results(output_file, config, individual_results, metrics):
    output = {
        "config": config,
        "individual_results": individual_results,
        "metrics": metrics
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--output', required=True, help='Path to output results JSON')
    parser.add_argument('--metrics', nargs='+', required=True, 
                    choices=['retriever', 'generator', 'end_to_end'],
                    help='Metrics to evaluate')
    parser.add_argument('--top_k', type=int, default=5, help='Number of chunks to retrieve')
    parser.add_argument('--strategy', default='basic_recursive', help='Chunking strategy')
    parser.add_argument('--search_type', default='hybrid', 
                    choices=['hybrid', 'semantic', 'keyword'],
                    help='Search type: hybrid, semantic, or keyword')
    parser.add_argument('--base_url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--threshold', type=float, default=0.85, 
                    help='Similarity threshold for chunk matching')
    
    args = parser.parse_args()
    
    print("Loading ground truth...")
    ground_truths = load_ground_truth(args.ground_truth)
    
    print("Loading models...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    needs_generation = 'generator' in args.metrics or 'end_to_end' in args.metrics
    retrieve_only = not needs_generation
    
    print(f"Evaluating {len(ground_truths)} questions...")
    
    individual_results = []
    
    for idx, gt in enumerate(ground_truths, 1):
        print(f"Processing: {idx}/{len(ground_truths)} {gt['id']}")
        
        try:
            response = call_search_api(
                gt['query'], 
                args.strategy, 
                args.top_k, 
                retrieve_only,
                args.search_type,
                args.base_url
            )
            
            result = {
                "query_id": gt['id'],
                "query": gt['query'],
                "car_model": gt['car_model'],
                "ground_truth": {
                    "expected_answer": gt['expected_answer'],
                    "key_facts": gt['key_facts'],
                    "relevant_chunks": gt['relevant_chunks']
                },
                "system_response": {
                    "retrieved_chunks": response.get('citations', []),
                    "generated_answer": response.get('answer', ''),
                    "latency_ms": response.get('metadata', {}).get('processing_time_ms', 0)
                },
                "scores": {}
            }
            
            individual_results.append(result)
            
        except Exception as e:
            print(f"Error processing {gt['id']}: {str(e)}")
            continue
    
    metrics_output = {}
    
    if 'retriever' in args.metrics:
        print("Calculating retriever metrics...")
        
        hit_rate = calculate_hit_rate(
            [r['system_response'] for r in individual_results],
            ground_truths,
            embedding_model,
            args.threshold,
            args.top_k
        )
        
        avg_recall = calculate_recall(
            [r['system_response'] for r in individual_results],
            ground_truths,
            embedding_model,
            args.threshold,
            args.top_k
        )
        
        avg_ndcg = calculate_ndcg(
            [r['system_response'] for r in individual_results],
            ground_truths,
            embedding_model,
            args.threshold,
            args.top_k
        )
        
        mrr = calculate_mrr(
            [r['system_response'] for r in individual_results],
            ground_truths,
            embedding_model,
            args.threshold,
            args.top_k
        )
        
        map_score = calculate_map(
            [r['system_response'] for r in individual_results],
            ground_truths,
            embedding_model,
            args.threshold,
            args.top_k
        )
        
        metrics_output['retriever'] = {
            "hit_rate": round(hit_rate, 3),
            "avg_recall": round(avg_recall, 3),
            "avg_ndcg": round(avg_ndcg, 3),
            "mrr": round(mrr, 3),
            "map": round(map_score, 3)
        }
    
    if 'generator' in args.metrics:
        print("Calculating generator metrics...")
        
        faithfulness_scores = []
        coverage_scores = []
        relevance_scores = []
        
        for result, gt in zip(individual_results, ground_truths):
            answer = result['system_response']['generated_answer']
            chunks = result['system_response']['retrieved_chunks']
            
            faith = calculate_faithfulness(answer, chunks, groq_client)
            if faith is not None:
                faithfulness_scores.append(faith)
                result['scores']['faithfulness'] = faith
            
            coverage = calculate_key_facts_coverage(answer, gt['key_facts'])
            coverage_scores.append(coverage)
            result['scores']['key_facts_coverage'] = coverage
            
            relevance = calculate_answer_relevance(answer, gt['query'], embedding_model)
            relevance_scores.append(relevance)
            result['scores']['answer_relevance'] = relevance
        
        metrics_output['generator'] = {
            "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else None,
            "avg_key_facts_coverage": round(sum(coverage_scores) / len(coverage_scores), 3) if coverage_scores else 0.0,
            "avg_answer_relevance": round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else 0.0
        }
    
    if 'end_to_end' in args.metrics:
        print("Calculating end_to_end metrics...")
        
        context_recall_scores = []
        correctness_scores = []
        latencies = []
        costs = []
        
        for result, gt in zip(individual_results, ground_truths):
            answer = result['system_response']['generated_answer']
            chunks = result['system_response']['retrieved_chunks']
            
            ctx_recall = calculate_context_recall(
                gt['query'], 
                chunks, 
                gt['expected_answer'], 
                groq_client
            )
            if ctx_recall is not None:
                context_recall_scores.append(ctx_recall)
                result['scores']['context_recall'] = ctx_recall
            
            correctness, f1, sem_sim = calculate_answer_correctness(
                answer, 
                gt['expected_answer'], 
                embedding_model
            )
            correctness_scores.append(correctness)
            result['scores']['answer_correctness'] = correctness
            result['scores']['f1_score'] = f1
            result['scores']['semantic_similarity'] = sem_sim
            
            latencies.append(result['system_response']['latency_ms'])
            
            cost = estimate_cost(answer, chunks)
            costs.append(cost)
            result['scores']['estimated_cost'] = cost
        
        # Add these lines for p95 and p99
        p95_latency = np.percentile(latencies, 95) if latencies else 0.0
        p99_latency = np.percentile(latencies, 99) if latencies else 0.0
        
        metrics_output['end_to_end'] = {
            "avg_context_recall": round(sum(context_recall_scores) / len(context_recall_scores), 3) if context_recall_scores else None,
            "avg_answer_correctness": round(sum(correctness_scores) / len(correctness_scores), 3) if correctness_scores else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(p95_latency, 2),  # Add this
            "p99_latency_ms": round(p99_latency, 2),  # Add this
            "estimated_cost_per_query": round(sum(costs) / len(costs), 6) if costs else 0.0
        }
    
    config = {
        "top_k": args.top_k,
        "strategy": args.strategy,
        "search_type": args.search_type,
        "threshold": args.threshold,
        "metrics_evaluated": args.metrics,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(ground_truths)
    }
    
    save_results(args.output, config, individual_results, metrics_output)
    
    print(f"\nResults saved to: {args.output}")
    print("\n=== METRICS ===")
    print(json.dumps(metrics_output, indent=2))


if __name__ == "__main__":
    main()