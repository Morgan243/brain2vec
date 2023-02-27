# Preatrain: 0, 1, 2
# Membership classifier: 0,1 vs. 3,4
# Test Membership cls: 2 vs 5, 6


# -
# Pretrain
#EXEC="time python -m brain2vec.experiments.pretrain"

echo "RESULT PATH: $RESULT_PATH"

mkdir -p $RESULT_PATH/models
# --dataset.n_dl_workers=39 --dataset.n_dl_eval_workers=39 \
#DEFAULT_CLI_ARGS=" --dataset.flatten_sensors_to_samples=True \
#--dataset.extra_output_keys='sensor_ras_coord_arr' \
#--dataset.batch_size=1024 --dataset.batch_size_eval=2048 \
#--task.learning_rate=0.001 --task=semi_supervised \
#--task.n_epochs=10 --task.device=cuda --dataset=hvs --task.lr_adjust_patience=10 \
#--model=brain2vec \
#--dataset.pre_processing_pipeline=random_sample \
#--result_dir=$RESULT_PATH --save_model_path=$RESULT_PATH/models "

N_SPLITS=6
export CHOOSE_N_FOR_PRETRAIN=3

for ((i = 0 ; i < $N_SPLITS ; i++)); do
  echo "Running $i index"
  export N_SPLITS
  export THIS_SPLIT=$i
  export RESULT_PATH
  if [ $i -gt 2 ]
  then
    export DEVICE="cuda:0"
  else
    export DEVICE="cuda:2"
  fi

  screen -S "split_$i" -dm bash -c 'bash --init-file <(./CMD.sh)'
done
