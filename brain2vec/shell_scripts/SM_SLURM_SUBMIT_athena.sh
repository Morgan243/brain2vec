#!/bin/bash
CUDA_DEV=$1
N_SPLITS=$2
THIS_SPLIT=$3

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

export RESULTS_DIR=../results/sm_240113
#export RESULTS_DIR=../results/ft_${TASK}_6pair_230930_2;


 sbatch<<EOT
#!/bin/bash
#SBATCH -o sm_${THIS_SPLIT}_of_${N_SPLITS}_output.out
#SBATCH --job-name=sm_${THIS_SPLIT}_of_${N_SPLITS}
#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --mem=300G
#SBATCH --nodes=1

#module load Cuda11.4
#module load gcc/11.2.1
#module load mkl/2021.2.0

export RESULTS_DIR=$RESULTS_DIR

./shell_scripts/grid_mi2.sh $RESULTS_DIR $CUDA_DEV --pretrained_result_dir ${INPUT_RESULTS} \
--model_output_key='bce_loss' \
--n_init_jobs=${N_INIT_JOBS} \
--n_splits=$N_SPLITS --this_split=$THIS_SPLIT
exit 0
EOT