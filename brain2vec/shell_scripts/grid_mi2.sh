RESULT_PATH=$1
DEVICE=$2
#PRETRAINED_RESULT_DIR="../results/results_2pair_100epoch_230326/"
#PRETRAINED_RESULT_DIR="../results/results_2pair_100epoch_230419/"
PRETRAINED_RESULT_DIR="../results_local/results_pretrain_2pair_small_230521/"

echo "RESULT_PATH: $RESULT_PATH"
echo "DEVICE: $DEVICE"
echo "PRETRAINED MODELS TO USE: $PRETRAINED_RESULT_DIR"

GRID="{'attacker_model.dropout':[0.5,0.0],'attacker_model.n_layers':[3,5],'attacker_model.linear_hidden_n':[128]}"

#GRID='{"attacker_model.dropout":[.5],"attacker_model.n_layers":range(1,3),"attacker_model.linear_hidden_n":[16, 32, 64, 128]}'

METHOD='1d_linear'
#METHOD='2d_linear'
#--n_channels_to_sample=2 \
#--batch_norm=True -

mkdir -p $RESULT_PATH;

time python grid/base.py --experiment_base_instance='info_leakage' \
--task='shadow_classifier_mi' \
--experiment_component_grids_str=$GRID \
--lr_adjust_patience=15 \
--device=$DEVICE \
--n_dl_workers=0 \
--n_dl_eval_workers=0 \
--batch_size=1024 --batch_size_eval=4096 \
--pipeline_params='{"rnd_stim__n":300}' --early_stopping_patience=25 \
--method=$METHOD --fine_tuning_method=$METHOD \
--flatten_sensors_to_samples=True \
--sensor_columns='good_for_participant' \
--result_dir=$RESULT_PATH \
--pretrained_result_dir=$PRETRAINED_RESULT_DIR \
--existing_results_dir=$RESULT_PATH \
--sample_tuples_for=test_sets --sample_choose_n=3 "${@:3}"
