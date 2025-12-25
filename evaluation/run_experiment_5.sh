# Create a bash script: run_experiments.sh

strategies=("landingai_based")
top_k_values=(5)

for strategy in "${strategies[@]}"; do
    for k in "${top_k_values[@]}"; do
        /Users/santhosh/Documents/study_projects/ringcentral_assessment/sentence_transformer/bin/python evaluate.py \
            --ground_truth ground_truth_dataset.json \
            --output results/${strategy}_k${k}.json \
            --metrics retriever generator end_to_end \
            --strategy $strategy \
            --top_k $k \
            --search_type semantic \
            --threshold 0.7
        
        echo "Completed: $strategy with k=$k"
    done
done