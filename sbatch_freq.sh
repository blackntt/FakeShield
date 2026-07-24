#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=mps:a100:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
REQUIRED_VRAM=20720

# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f
module load shared python312
# *** Kích hoạt venv (Sửa đường dẫn theo user)
umask 000

CHECK_OUT=$(nvidia-smi -i 0,1,2,3,4,5,6,7 \
  --query-gpu=index,utilization.gpu,memory.total,memory.used \
  --format=csv,noheader,nounits |
  awk -F', *' -v req="$REQUIRED_VRAM" '{
  idx=$1; util=$2; total=$3; used=$4; free=total-used;

  if (free >= 1*req) {
    score = free - util*100
    if (best=="" || score>best) {
      best=score; best_idx=idx
    }
  }
}
END { if (best_idx!="") print best_idx }')

# if CHECK_OUT is empty, exit with error
if [ -z "$CHECK_OUT" ]; then
  echo "========================================================"
  echo "❌ ERROR: No available GPU matches your VRAM requirements!"
  echo "   Required Minimum VRAM: ${REQUIRED_VRAM} MB"
  echo "   Exiting task allocation to prevent a CPU fallback freeze."
  echo "========================================================"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $CHECK_OUT"
export LD_PRELOAD=$(find $(python -c 'import site; print(site.getsitepackages()[0])')/nvidia -name "libcusolver.so.11" | head -n 1)

# =========================================================
# KHỞI TẠO PRIVATE MPS SERVER
# =========================================================
# Tạo thư mục riêng biệt cho job này
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job-exvfd-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job-exvfd-$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

#training.real_loss_fomulas=uniform
PYTHON="/datastore/cndt_toannt/miniconda/envs/fakeshield_env_new/bin/python"
PYTHON2="/datastore/cndt_toannt/miniconda/envs/fakeshield_env/bin/python"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
OUTPUT_DIR=./playground/DTE-FDM_train_result
DATA_PATH=dataset.json
IMAGE_PATH=dataset
WEIGHT_PATH=./weight/fakeshield-v1-22b/DTE-FDM
FREQ_MODEL_PATH=/datastore/cndt_toannt/FakeShield/weight/noiseprint.th
CLIP_PATH=/datastore/cndt_toannt/FakeShield/weight/clip

mkdir -p $OUTPUT_DIR
"$PYTHON" ./DTE-FDM/llava/train/train_mem.py \
  --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
  --deepspeed ./scripts/DTE-FDM/zero3.json \
  --model_name_or_path $WEIGHT_PATH \
  --version v1 \
  --data_path $DATA_PATH \
  --image_folder $IMAGE_PATH \
  --vision_tower $CLIP_PATH \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 1 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_steps 800 \
  --save_total_limit 1 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --noiseprint_projector_path $FREQ_MODEL_PATH \
  --noiseprint_projector_mm_hidden_size 4096 \
  --noiseprint_projector_num_tokens 64
