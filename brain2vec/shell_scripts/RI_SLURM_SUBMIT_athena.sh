#!/bin/bash
CUDA_DEV=$1
METHOD=$2
N_SPLITS=$3
THIS_SPLIT=$4

QVARS=20
MLEN=2
N_ENCODER_LAYERS=8
PEMBED='pos_embedding'
#PEMBED='combined'
FEAT_LAYERS='[(128, 7, 2)] + [(64, 3, 2)] * 2 + [(32, 3, 2)] * 2'
#FEAT_LAYERS='[(128, 7, 7), (64, 5, 5), (16, 3, 2)]'
N_INIT_JOBS=5

#INPUT_RESULTS='../results/results_pretrain_1pair_230916/'
#export RESULTS_DIR=../results/ft_${TASK}_1pair_230916;
#N_CH=48

#INPUT_RESULTS='../results/results_pretrain_6pair_230930/'
#INPUT_RESULTS='../results/results_pretrain_6pair_231202/'
#INPUT_RESULTS='../results/results_pretrain_6pair_231207/'
#INPUT_RESULTS='../results/results_pretrain_6pair_231223/'
INPUT_RESULTS='../results/results_pretrain_6pair_240115_integrated/'

#INPUT_RESULTS='../results/results_pretrain_6pair_231108/'
#INPUT_RESULTS='../results/results_pretrain_6pair_231104/'
#INPUT_RESULTS='../results/results_pretrain_6pair_231110/'
#export RESULTS_DIR="../results/ri_231223/"
export RESULTS_DIR="../results/ri_240115/"
#export RESULTS_DIR=../results/ft_${TASK}_6pair_230930_2;

#SCRIPTS_PATH='./'
#SBATCH -e ft_speech_$2_error.out

FLATTEN_SENSORS=True
#METHOD='1d_linear'
N_CH_TO_SAMPLE=None
echo "METHOD: $METHOD"
if [ "$METHOD" == "2d_linear" ]; then
  echo "2D Mode"
  FLATTEN_SENSORS=False
  GRID="{'attacker_model.n_layers':[3],'attacker_model.dropout':[.75],'attacker_model.dropout_2d_rate':[.0],'attacker_model.linear_hidden_n':[128],'task.n_channels_to_sample':[1,2,4,8,16,32,64]}"
  #N_CH_TO_SAMPLE=$N_CH
  #N_CH_OPT_STR="--n_channels_to_sample=$N_CH_TO_SAMPLE"
fi

#--input_results_query='quant_num_vars.eq($QVARS) & n_encoder_layers.eq($N_ENCODER_LAYERS) & mask_length.eq($MLEN) & feature_extractor_layers.eq("$FEAT_LAYERS") & positional_encoding_method.eq("$PEMBED")' \
echo "REMINDER THAT INPUT RESULT QUERY IS NOT APPLIED!"

 sbatch<<EOT
#!/bin/bash
#SBATCH -o ri_${THIS_SPLIT}_of_${N_SPLITS}_output.out
#SBATCH --job-name=ri_${THIS_SPLIT}_of_${N_SPLITS}
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
#SBATCH --mem=180G

module load Cuda11.4

export RESULTS_DIR=$RESULTS_DIR

./shell_scripts/grid_reid.sh $RESULTS_DIR $CUDA_DEV --input_results_dir ${INPUT_RESULTS} \
--dropout=0. --dropout_2d_rate=.75 --batch_norm=True \
--n_init_jobs=${N_INIT_JOBS} \
--flatten_sensors_to_samples=$FLATTEN_SENSORS --fine_tuning_method=$METHOD \
--experiment_component_grids_str="$GRID" \
--n_splits=$N_SPLITS --this_split=$THIS_SPLIT
exit 0
EOT