RESULT_PATH=$1
DEVICE=$2
#PRETRAINED_RESULT_DIR="../results/results_2pair_100epoch_230326/"
#PRETRAINED_RESULT_DIR="../results/results_2pair_100epoch_230419/"
#PRETRAINED_RESULT_DIR="../results_local/results_pretrain_2pair_small_230521/"
#PRETRAINED_RESULT_DIR="../results/results_pretrain_6pair_230628/"

echo "RESULT_PATH: $RESULT_PATH"
echo "DEVICE: $DEVICE"
#echo "PRETRAINED MODELS TO USE: $PRETRAINED_RESULT_DIR"
#--pretrained_result_dir=$PRETRAINED_RESULT_DIR \

GRID="{'attacker_model.n_layers':[3],'attacker_model.dropout':[.75],'attacker_model.linear_hidden_n':[128]}"
#GRID="{'attacker_model.n_layers':[3],'attacker_model.dropout':[.75,.5],'attacker_model.linear_hidden_n':[128],'batch_size':[4096]}"
#GRID="{'attacker_model.n_layers':[2],'attacker_model.dropout_2d_rate':.75}"
#GRID="{'attacker_model.dropout':[0.5,0.0],'attacker_model.n_layers':[3,5],'attacker_model.linear_hidden_n':[128]}"
#GRID='{"attacker_model.dropout":[.5],"attacker_model.n_layers":range(1,3),"attacker_model.linear_hidden_n":[16, 32, 64, 128]}'

METHOD='1d_linear'
#METHOD='2d_linear'
#--n_channels_to_sample=2 \
#--batch_norm=True -
#--input_results_query='quant_num_vars == 20 and n_encoder_layers== 8' \

mkdir -p $RESULT_PATH;

#time python grid/base.py \
time python -m brain2vec.grid.grid_on_results \
--experiment_base_instance='info_leakage' \
--result_file='dummy' \
--task='reid' \
--experiment_component_grids_str=$GRID \
--lr_adjust_patience=10 \
--device=$DEVICE \
--n_dl_workers=8 \
--n_dl_eval_workers=8 \
--batch_size=1024 --batch_size_eval=4096 \
--pipeline_params='{"rnd_stim__n":20000}' --early_stopping_patience=15 \
--n_epochs=100 \
--fine_tuning_method=$METHOD \
--flatten_sensors_to_samples=True \
--sensor_columns='good_for_participant' \
--result_dir=$RESULT_PATH \
--existing_results_dir=$RESULT_PATH \
--batch_norm=True \
--train_sets='*' "${@:3}"
