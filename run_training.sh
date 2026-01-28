# #!/bin/bash
export WANDB_BASE_URL="https://api.bandw.top"
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,2,3
# 设置内存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "可见GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv -i 1,2,3

echo ""
echo "========================================"
echo "开始训练..."
echo "========================================"
# python train_alignment_system.py \
#     --config config/config_bi_example.yaml \

# python train_alignment_system.py \
#     --config config/config_bi_example_copy.yaml \

# python train_alignment_system.py \
#     --config config/config_bi_threshold.yaml \


python train_alignment_system.py \
    --config config/config_bi_threshold.yaml \



