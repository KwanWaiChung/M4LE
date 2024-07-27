seed=111
n_examples=200

# qa
python data/create_qa_data.py \
    --seed ${seed} \
    --buckets "[1000, 2000, 4000, 6000, 8000, 16000, 32000, 64000, 128000]" \
    --one_shot \
    --n_examples ${n_examples}

# topic retrieval
python data/create_topic_retrieval_data.py \
    --buckets "[1000, 2000, 4000, 6000, 8000, 16000, 32000, 64000, 128000]" \
    --n_examples ${n_examples} \
    --seed ${seed} \
    --one_shot


# book summarization
python data/create_booksum_data.py \
    --buckets "[1000, 2000, 4000, 6000, 8000, 16000, 32000, 64000, 128000]" \
    --one_shot \
    --n_examples ${n_examples} \
    --seed ${seed}

# summarization
python data/create_summarization_data.py \
    --buckets "[1000, 2000, 4000, 6000, 8000, 16000, 32000, 64000, 128000]" \
    --n_examples ${n_examples} \
    --seed ${seed}

# classification
python data/create_classification_data.py \
    --buckets "[1000, 2000, 4000, 6000, 8000, 16000, 32000, 64000, 128000]" \
    --n_examples ${n_examples} \
    --seed ${seed}

# translation
python data/create_translation_data.py \
    --buckets "[1000, 2000, 4000, 6000, 8000, 16000, 32000, 64000, 128000]" \
    --n_examples ${n_examples} \
    --one_shot \
    --seed ${seed}

# nli
python data/create_nli_data.py \
    --buckets "[1000, 2000, 4000, 6000, 8000, 16000, 32000, 64000, 128000]" \
    --n_examples ${n_examples} \
    --one_shot \
    --seed ${seed}