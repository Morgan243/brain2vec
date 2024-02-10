#!/bin/bash
CUDA_DEV=$1
TASK=$2
N_SPLITS=$3
THIS_SPLIT=$4

QVARS=20
MLEN=2
PEMBED='pos_embedding'
N_ENCODER_LAYERS=4
FEAT_LAYERS='[(128, 7, 2)] + [(64, 3, 2)] * 2 + [(32, 3, 2)] * 2'
#FEAT_LAYERS='[(128, 7, 7), (64, 5, 5), (16, 3, 2)]'
N_INIT_JOBS=2

# - 1 pair -
INPUT_RESULTS='../results/results_pretrain_1pair_230916/'
export RESULTS_DIR=../results/ft_${TASK}_1pair_230916;

# - 6 pair -
#INPUT_RESULTS='../results/results_pretrain_6pair_230930/'
#export RESULTS_DIR=../results/ft_${TASK}_6pair_230930_2;

#SCRIPTS_PATH='./'
#SBATCH -e ft_speech_$2_error.out

sbatch <<EOT
#!/bin/bash
#SBATCH -o ft_${TASK}_${THIS_SPLIT}_of_${N_SPLITS}_output.out
#SBATCH --job-name=${TASK}_${THIS_SPLIT}_of_${N_SPLITS}
#SBATCH --cpus-per-task=2
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --mem=150G

module load Cuda11.4

export RESULTS_DIR=$RESULTS_DIR

./shell_scripts/ft_$TASK.sh --input_results_dir ${INPUT_RESULTS} --result_dir $RESULTS_DIR --existing_results_dir $RESULTS_DIR --dropout=0. --dropout_2d_rate=.75 --batch_norm=True --device $CUDA_DEV --n_init_jobs=${N_INIT_JOBS} --input_results_query='quant_num_vars.eq($QVARS) & n_encoder_layers.eq($N_ENCODER_LAYERS) & mask_length.eq($MLEN) & feature_extractor_layers.eq("$FEAT_LAYERS") & positional_encoding_method.eq("$PEMBED")' --n_splits=$N_SPLITS --this_split=$THIS_SPLIT
exit 0
EOT