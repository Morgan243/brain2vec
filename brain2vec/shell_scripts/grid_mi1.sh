RESULT_PATH=$1
DEVICE=$2
#PRETRAINED_RESULT_DIR="../results/results_2pair_100epoch_230326/"
#PRETRAINED_RESULT_DIR="../results/results_100/"
PRETRAINED_RESULT_DIR="../results/results_2pair_100epoch_230419/"

echo "RESULT_PATH: $RESULT_PATH"
echo "DEVICE: $DEVICE"
echo "PRETRAINED MODELS TO USE: $PRETRAINED_RESULT_DIR"

mkdir -p $RESULT_PATH
GRID="{'task.dataset.test_sets':[0,1,2],'attacker_model.n_layers':[1,2],'attacker_model.linear_hidden_n':[16,32]}"

python -m brain2vec.grid.base \
--experiment_base_instance=info_leakage \
--experiment_component_grids_str=$GRID \
--device=$DEVICE \
--n_dl_workers=0 \
--n_dl_eval_workers=0 \
--task=one_model_mi \
--lr_adjust_patience=15 \
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


#--existing_results_dir=$RESULT_PATH \
#--train_sets='UCSD-4,UCSD-5,UCSD-10' \
#--result_file=../results/results_100/20230212_2357_8750c623-b58d-4384-af3b-6db2f23b9bea.json
