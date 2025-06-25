# ---------- Environment ----------
nproc_per_node=8

# ---------- Environment ----------
export NPROC_PER_NODE=$nproc_per_node   # Used by torchrun / SWIFT
export USE_LIBUV=0                      # Disable libuv TCPStore on Windows

# ---------- Training ----------
swift sft \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset data/processed/hyperbook_infonce.jsonl \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 20 \
    --num_train_epochs 5 \
    --save_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true
