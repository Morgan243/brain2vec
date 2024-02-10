RESULT_PATH=$1
DEVICE=$2
# Needs to be 3-pair
PRETRAINED_RESULT_DIR="../results/results_100/"

echo "RESULT_PATH: $RESULT_PATH"
echo "DEVICE: $DEVICE"
echo "PRETRAINED MODELS TO USE: $PRETRAINED_RESULT_DIR"

GRID="{'task.dataset.test_sets':[0,1,2],'attacker_model.n_layers':[1,2,3],'attacker_model.linear_hidden_n':[8,16,32]}"

mkdir -p $RESULT_PATH

python -m brain2vec.grid.base \
--experiment_base_instance=info_leakage \
--experiment_component_grids_str=$GRID \
--task=one_model_mi \
--lr_adjust_patience=15 \
--device=$DEVICE \
--dropout=0.5 \
--batch_norm=True \
--batch_size=1024 \
--batch_size_eval=4096 \
--pipeline_params='{"rnd_stim__n":100000}' \
--sensor_columns="good_for_participant" \
--early_stopping_patience=20 \
--method="2d_linear" \
--fine_tuning_method="2d_linear" \
--n_channels_to_sample=2 \
--sample_tuples_for="train_sets" \
--sample_choose_n=3 \
--sensor_columns="good_for_participant" \
--pretrained_result_dir=$PRETRAINED_RESULT_DIR \
--result_dir=$RESULT_PATH "${@:3}"


