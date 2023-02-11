# Preatrain: 0, 1, 2
# Membership classifier: 0,1 vs. 3,4
# Test Membership cls: 2 vs 5, 6


# -
# Pretrain
EXEC="time python -m brain2vec.experiments.pretrain"

echo "RESULT PATH: $RESULT_PATH"

mkdir -p $RESULT_PATH/models
# --dataset.n_dl_workers=39 --dataset.n_dl_eval_workers=39 \
DEFAULT_CLI_ARGS=" --dataset.flatten_sensors_to_samples=True \
--dataset.extra_output_keys='sensor_ras_coord_arr' \
--dataset.batch_size=1024 --dataset.batch_size_eval=2048 \
--task.learning_rate=0.001 --task=semi_supervised \
--task.n_epochs=10 --task.device=cuda --dataset=hvs --task.lr_adjust_patience=10 \
--model=brain2vec \
--dataset.pre_processing_pipeline=random_sample \
--result_dir=$RESULT_PATH --save_model_path=$RESULT_PATH/models "


pretraining_sets=("UCSD-4,UCSD-5,UCSD-10" "UCSD-4,UCSD-5,UCSD-18" "UCSD-4,UCSD-5,UCSD-19" "UCSD-4,UCSD-5,UCSD-22" "UCSD-4,UCSD-5,UCSD-28" "UCSD-4,UCSD-10,UCSD-18" "UCSD-4,UCSD-10,UCSD-19" "UCSD-4,UCSD-10,UCSD-22" "UCSD-4,UCSD-10,UCSD-28" "UCSD-4,UCSD-18,UCSD-19" "UCSD-4,UCSD-18,UCSD-22" "UCSD-4,UCSD-18,UCSD-28" "UCSD-4,UCSD-19,UCSD-22" "UCSD-4,UCSD-19,UCSD-28" "UCSD-4,UCSD-22,UCSD-28" "UCSD-5,UCSD-10,UCSD-18" "UCSD-5,UCSD-10,UCSD-19" "UCSD-5,UCSD-10,UCSD-22" "UCSD-5,UCSD-10,UCSD-28" "UCSD-5,UCSD-18,UCSD-19" "UCSD-5,UCSD-18,UCSD-22" "UCSD-5,UCSD-18,UCSD-28" "UCSD-5,UCSD-19,UCSD-22" "UCSD-5,UCSD-19,UCSD-28" "UCSD-5,UCSD-22,UCSD-28" "UCSD-10,UCSD-18,UCSD-19" "UCSD-10,UCSD-18,UCSD-22" "UCSD-10,UCSD-18,UCSD-28" "UCSD-10,UCSD-19,UCSD-22" "UCSD-10,UCSD-19,UCSD-28" "UCSD-10,UCSD-22,UCSD-28" "UCSD-18,UCSD-19,UCSD-22" "UCSD-18,UCSD-19,UCSD-28" "UCSD-18,UCSD-22,UCSD-28" "UCSD-19,UCSD-22,UCSD-28")

# shellcheck disable=SC2068
for set_str in ${pretraining_sets[@]}; do
  echo $set_str
  eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$set_str"
  #for ((i = 0 ; i < 3 ; i++)); do
  #  echo "Welcome $i times."
  #  eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$set_str --dataset.test_sets=$i"
  #done
done

#A="UCSD-4"
#B="UCSD-5"
#C="UCSD-10"
#D="UCSD-18"
#E="UCSD-19"
#F="UCSD-22"
#G="UCSD-28"
#
##eval "$EXEC $DEFAULT_CLI_ARGS $@"
##eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$B,$C"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$B,$C"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$B,$D"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$B,$E"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$B,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$B,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$C,$D"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$C,$E"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$C,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$C,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$D,$E"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$D,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$D,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$E,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$E,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$A,$F,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$C,$D"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$C,$E"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$C,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$C,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$D,$E"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$D,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$D,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$E,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$E,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$B,$F,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$C,$D,$E"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$C,$D,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$C,$D,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$C,$E,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$C,$E,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$C,$F,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$D,$E,$F"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$D,$E,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$D,$F,$G"
#eval "$EXEC $DEFAULT_CLI_ARGS --dataset.train_sets=$E,$F,$G"
#
#