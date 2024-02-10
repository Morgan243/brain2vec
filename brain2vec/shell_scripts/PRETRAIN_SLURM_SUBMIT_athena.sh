#!/bin/bash
CUDA_DEV=$1
NPAIR=$2
N_SPLITS=$3
THIS_SPLIT=$4

QVARS=20
MLEN=1
#PEMBED='combined'
PEMBED='pos_embedding'
N_ENCODER_LAYERS=6
FEAT_LAYERS='[(128, 7, 2)] + [(64, 3, 2)] * 2 + [(32, 3, 2)] * 2'
#FEAT_LAYERS='[(128, 7, 7), (64, 5, 5), (16, 3, 2)]'
N_INIT_JOBS=2

#INPUT_RESULTS='../results/results_pretrain_1pair_230916/'
#export RESULTS_DIR=../results/ft_${TASK}_1pair_230916;

#INPUT_RESULTS='../results/results_pretrain_6pair_230930/'
#export RESULTS_DIR=../results/ft_${TASK}_6pair_230930;
#export RESULTS_DIR="../results/results_pretrain_${NPAIR}pair_231110/"

# Mem test
export RESULTS_DIR="../results/results_pretrain_${NPAIR}pair_240204_memtest/"

#export RESULTS_DIR='../results/results_pretrain_6pair_240115_integrated'
#echo "WARNING - using hardcoded INPUT RESULTS pointed at integrated: $RESULTS_DIR"

#SCRIPTS_PATH='./'
#SBATCH -e ft_speech_$2_error.out

sbatch <<EOT
#!/bin/bash
#SBATCH -o pt_${NPAIR}_${THIS_SPLIT}_of_${N_SPLITS}_output.out
#SBATCH --job-name=pt_${NPAIR}_${THIS_SPLIT}_of_${N_SPLITS}
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1

#module load Cuda11.4
#module load gcc/11.2.1
#module load mkl/2021.2.0


export RESULTS_DIR=$RESULTS_DIR
RESULTS_PATH=${RESULTS_DIR}
export RESULTS_PATH

./shell_scripts/pretrain_${NPAIR}pair_grid.sh $RESULTS_DIR ${CUDA_DEV} \
--existing_results_dir ${RESULTS_DIR} --n_init_jobs=${N_INIT_JOBS} \
--experiment_component_grids_str="{'model.quant_num_vars':[${QVARS}],'model.n_encoder_layers':[${N_ENCODER_LAYERS}],'model.positional_encoding_method':['${PEMBED}'],'model.mask_length':[${MLEN}],'model.feature_extractor_layers':['${FEAT_LAYERS}']}" \
--n_splits=$N_SPLITS --this_split=$THIS_SPLIT

exit 0
EOT