#!/bin/bash
CUDA_DEV=$1
N_SPLITS=$2
THIS_SPLIT_CSV=$3

#QVARS=80
#MLEN=1
#PEMBED='pos_embedding'
##FEAT_LAYERS='[(128, 7, 2)] + [(64, 3, 2)] * 2 + [(32, 3, 2)] * 2'
#FEAT_LAYERS='[(128, 7, 7), (64, 5, 5), (16, 3, 2)]'
N_INIT_JOBS=1

#INPUT_RESULTS='../results/results_pretrain_2pair_2301016_2/'
#INPUT_RESULTS='../results/results_pretrain_2pair_231020_orig/'
#INPUT_RESULTS='../results/results_pretrain_2pair_231024/'
#INPUT_RESULTS='../results/results_pretrain_2pair_231104/'

INPUT_RESULTS='../results/results_pretrain_2pair_240113/'

export RESULTS_DIR=../results/sm_240113_10ksamp
#export RESULTS_DIR=../results/ft_${TASK}_6pair_230930_2;

if [ $THIS_SPLIT_CSV = "ALL" ]; then
  SLURM_JOB_NAME="sm_${THIS_SPLIT_CSV}_of_${N_SPLITS}"
  SLURM_OUTPUT_FILE="${SLURM_JOB_NAME}_output.out"

  THIS_SPLIT_CSV=$(seq -s "," 0 $(( N_SPLITS - 1 )))
  TMPARR=(${THIS_SPLIT_CSV//,/ })
  N_CSV_SPLITS=${#TMPARR[@]}
else
  SLURM_JOB_NAME="sm_${THIS_SPLIT_CSV//,/_}_of_${N_SPLITS}"
  SLURM_OUTPUT_FILE="${SLURM_JOB_NAME}_output.out"
  TMPARR=(${THIS_SPLIT_CSV//,/ })
  N_CSV_SPLITS=${#TMPARR[@]}
fi

echo "N SPLITS: $N_CSV_SPLITS"

#SBATCH --cpus-per-task=$N_CSV_SPLITS
#SBATCH --mem=$(( 150 * N_CSV_SPLITS))G
#cat<<EOT
sbatch<<EOT
#!/bin/bash
#SBATCH -o ${SLURM_OUTPUT_FILE}
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --mem=500G
#SBATCH --nodes=1

export RESULTS_DIR=$RESULTS_DIR

SPLITS_CSV=$THIS_SPLIT_CSV

for SPLIT in \${SPLITS_CSV//,/ }
do
    sub_file=${SLURM_JOB_NAME}_split_\${SPLIT}.out
    echo "Running split \$SPLIT in background"

    ./shell_scripts/grid_mi2.sh $RESULTS_DIR $CUDA_DEV --pretrained_result_dir ${INPUT_RESULTS} \
    --model_output_key='bce_loss' \
    --n_init_jobs=${N_INIT_JOBS} \
    --n_splits=$N_SPLITS --this_split=\$SPLIT >> "${SLURM_OUTPUT_FILE}.part\${SPLIT}" 2>&1 &

    PIDS[\$SPLIT]=\$!
done

# wait for all pids
for pid in \${PIDS[*]}; do
    #echo "Waiting on \$pid"
    #wait \$pid
    echo "Waiting one.."
    wait -n
    echo "...joined!"
done

exit 0
EOT