RESULT_PATH=$1
DEVICE=$2

PRETRAINED_RESULT_DIR="../results/results_2pair_100epoch_230419/"

TEST_SETS='UCSD-4,UCSD-5,UCSD-10'
DEVICE='cuda:7'
N_CHANNELS_TO_SAMPLE=3

time python experiments/info_leakage_eval.py \
    --task='shadow_classifier_mi' \
    --test_sets=$TEST_SETS \
    --device=$DEVICE \
    --n_dl_workers=0 \
    --n_dl_eval_workers=0 \
    --lr_adjust_patience=7 \
    --n_layers=3 \
    --method='1d_linear' \
    --fine_tuning_method='1d_linear' \
    --flatten_sensors_to_samples=True \
    --pretrained_result_dir=$PRETRAINED_RESULT_DIR \
    --sensor_columns='good_for_participant' \
    --task_dataset_dir=$RESULT_PATH \
    --batch_size_eval=10000 \
    --batch_size=10000  "${@:3}"
