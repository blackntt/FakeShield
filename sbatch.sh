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
WEIGHT_PATH=./weight/fakeshield-v1-22b
IMAGE_PATH=./playground/images/Sp_D_CND_A_pla0005_pla0023_0281.jpg
DTE_FDM_OUTPUT=./playground/DTE-FDM_output.jsonl
MFLM_OUTPUT=./playground/MFLM_output

"$PYTHON" -m llava.serve.cli \
  --model-path ${WEIGHT_PATH}/DTE-FDM \
  --DTG-path ${WEIGHT_PATH}/DTG.pth \
  --image-path ${IMAGE_PATH} \
  --output-path ${DTE_FDM_OUTPUT}
#
"$PYTHON" ./MFLM/cli_demo.py \
  --version ${WEIGHT_PATH}/MFLM \
  --DTE-FDM-output ${DTE_FDM_OUTPUT} \
  --MFLM-output ${MFLM_OUTPUT}
