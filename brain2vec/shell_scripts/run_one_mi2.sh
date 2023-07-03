
RESULT_PATH=$1
DEVICE=$2
PRETRAINED_RESULT_DIR="../results/results_2pair_100epoch_230326/"

echo "RESULT_PATH: $RESULT_PATH"
echo "DEVICE: $DEVICE"
echo "PRETRAINED MODELS TO USE: $PRETRAINED_RESULT_DIR"

mkdir -p $RESULT_PATH;

#time python -m brain2vec.experiments.info_leakage_eval \
time python grid/base.py \
    --experiment_component_grids_str='{}' \
    --experiment_base_instance='info_leakage' \
    --task='shadow_classifier_mi' \
    --lr_adjust_patience=15 \
    --device=$DEVICE \
    --n_layers=1 \
    --linear_hidden_n=32 \
    --dropout=0.5 \
    --batch_norm=True \
    --batch_size=1024 \
    --batch_size_eval=4096 \
    --pipeline_params='{"rnd_stim__n":10000}' \
    --early_stopping_patience=20 \
    --weight_decay=0.0 \
    --method='2d_linear' \
    --fine_tuning_method='2d_linear' \
    --n_channels_to_sample=2 \
    --sensor_columns='good_for_participant' \
    --result_dir=$RESULT_PATH  \
    --test_sets='UCSD-4,UCSD-5,UCSD-10' \
    --pretrained_result_dir=$PRETRAINED_RESULT_DIR \
    --existing_results_dir=$RESULT_PATH \
    --sample_tuples_for=test_sets \
    --sample_choose_n=3
