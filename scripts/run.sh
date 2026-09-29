#!/bin/bash

cd /mnt/NexusRAG || exit 1

# Common parameters
EMBEDDING_MODEL="/mnt/model/sentence-transformers/all-mpnet-base-v2"
LLM_MODEL="gpt-4o-mini"

MAX_WORKERS=16

# Dataset runner
run_dataset() {
    local DATASET=$1
    local SPACY_MODEL=$2
    local MAX_ITERATION=$3
    local THRESHOLD=$4
    local TOP_K_SENTENCE=$5
    local PRECOMPUTE_THRESHOLD=$6
    local COOCCUR_ALPHA=$7

    echo "=========================================="
    echo "Running dataset: ${DATASET} PRECOMPUTE_THRESHOLD=${PRECOMPUTE_THRESHOLD} COOCCUR_ALPHA=${COOCCUR_ALPHA}"
    echo "=========================================="

    python run.py \
        --spacy_model "${SPACY_MODEL}" \
        --embedding_model "${EMBEDDING_MODEL}" \
        --dataset_name "${DATASET}" \
        --llm_model "${LLM_MODEL}" \
        --max_workers "${MAX_WORKERS}" \
        --max_iterations "${MAX_ITERATION}" \
        --iteration_threshold "${THRESHOLD}" \
        --top_k_sentence "${TOP_K_SENTENCE}" \
        --top_k_entity_cooccur 5 \
        --precompute_threshold "${PRECOMPUTE_THRESHOLD}" \
        $([ -n "${COOCCUR_ALPHA}" ] && echo "--cooccur_alpha ${COOCCUR_ALPHA}") 
}

run_dataset "2wikimultihop" "en_core_web_trf" 3 0.4 1 0.5 0.5 && \
run_dataset "hotpotqa" "en_core_web_trf" 3 0.4 1 0.5 0.5 && \
run_dataset "medical" "en_core_web_trf" 3 0.5 3 0.5 0.5 && \
run_dataset "musique" "en_core_web_trf" 5 0.1 4 0.5 0.5

echo "=========================================="
echo "All datasets finished!"
echo "=========================================="
