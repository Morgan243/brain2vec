RESULT_PATH=$1
DEVICE=$2

echo "RESULT_PATH: $RESULT_PATH"
echo "DEVICE: $DEVICE"

mkdir -p $RESULT_PATH
#GRID="{'task.dataset.test_sets':[0,1,2],'attacker_model.n_layers':[1,2],'attacker_model.linear_hidden_n':[16,32]}"
#GRID="{'model.quant_num_vars':[20,40,80,160],'model.n_encoder_layers':[4,6,8]}"

#GRID="{'model.quant_num_vars':[10],'model.n_encoder_layers':[4]}"
#GRID="{'model.quant_num_vars':[160],'model.n_encoder_layers':[8]}"
#GRID="{'model.quant_num_vars':[20],'model.n_encoder_layers':[8]}"
GRID="{'model.quant_num_vars':[40],'model.n_encoder_layers':[10]}"

#GRID="{'model.quant_num_vars':[160],'model.n_encoder_layers':[8],'model.ras_pos_encoding':[False],'model.positional_encoding_method':['position']}"

python -m brain2vec.grid.base \
--experiment_base_instance=pretrain \
--experiment_component_grids_str=$GRID \
--device=$DEVICE \
--n_dl_workers=0 \
--n_dl_eval_workers=0 \
--task=semi_supervised \
--dataset=hvs \
--lr_adjust_patience=10 \
--batch_size=1024 \
--batch_size_eval=4096 \
--early_stopping_patience=15 \
--pipeline_params='{"rnd_stim__n":10000,"rnd_stim__slice_selection":slice(0,0.4)}' \
--sensor_columns="good_for_participant" \
--sample_tuples_for="train_sets" \
--sample_choose_n=6 \
--sensor_columns="good_for_participant" \
--result_dir=$RESULT_PATH "${@:3}"

