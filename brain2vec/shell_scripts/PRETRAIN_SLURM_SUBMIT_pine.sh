#!/bin/bash
CUDA_DEV=$1
NPAIR=$2
N_SPLITS=$3
THIS_SPLIT=$4

QVARS=320
MLEN=2
PEMBED='combined'
N_ENCODER_LAYERS=6
FEAT_LAYERS='[(128, 7, 2)] + [(64, 3, 2)] * 2 + [(32, 3, 2)] * 2'
#FEAT_LAYERS='[(128, 7, 7), (64, 5, 5), (16, 3, 2)]'
N_INIT_JOBS=2

#INPUT_RESULTS='../results/results_pretrain_1pair_230916/'
#export RESULTS_DIR=../results/ft_${TASK}_1pair_230916;

#INPUT_RESULTS='../results/results_pretrain_6pair_230930/'
#export RESULTS_DIR=../results/ft_${TASK}_6pair_230930;
export RESULTS_DIR='../results/results_pretrain_6pair_231223/'

#SCRIPTS_PATH='./'
#SBATCH -e ft_speech_$2_error.out

sbatch <<EOT
#!/bin/bash
#SBATCH -o pt_${NPAIR}_${THIS_SPLIT}_of_${N_SPLITS}_output.out
#SBATCH --job-name=pt_${NPAIR}_${THIS_SPLIT}_of_${N_SPLITS}
#SBATCH --cpus-per-task=2
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --mem=64G

module load Cuda11.4

export RESULTS_DIR=$RESULTS_DIR
RESULTS_PATH=RESULTS_DIR
export RESULTS_PATH

./shell_scripts/pretrain_${NPAIR}pair_grid.sh $RESULTS_DIR $CUDA_DEV \
--existing_results_dir $RESULTS_DIR --n_init_jobs=${N_INIT_JOBS} \
--experiment_component_grids_str="{'model.quant_num_vars':[${QVARS}],'model.n_encoder_layers':[${N_ENCODER_LAYERS}],'model.positional_encoding_method':['${PEMBED}'],'model.mask_length':[${MLEN}],'model.feature_extractor_layers':['${FEAT_LAYERS}']}" \
--n_splits=$N_SPLITS --this_split=$THIS_SPLIT

exit 0
EOT