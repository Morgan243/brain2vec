
#RESULTS_DIR="../results/results_2pair"
#SHADOW_0='20230215_1737_ff27ac5d-20a3-4c3b-9ba7-9e5ed3ab705b.json'
#SHADOW_1='20230215_0907_39966c4e-b105-4cf3-ac5a-ec75984dfa51.json'
#TARGET='20230215_0022_d15fa8a5-77f0-441c-a60a-1cc1768bc6ce.json'

RESULTS_DIR="../../results/results_2pair_50epoch"
SHADOW_0='20230217_1439_0fe41c4c-9ca4-4d76-ade1-b0fab925550d.json'
SHADOW_1='20230216_2135_063426a2-dc66-40d6-8193-28aef0012cad.json'
TARGET='20230216_0730_6333d547-bb0b-4552-9b9a-af1c08f6196f.json'


python -m brain2vec.experiments.info_leakage_eval \
  --task='shadow_classifier_mi' \
	--pretrained_shadow_model_result_input_0.result_file="$RESULTS_DIR/$SHADOW_0" \
	--pretrained_shadow_model_result_input_1.result_file="$RESULTS_DIR/$SHADOW_1" \
	--pretrained_target_model_result_input.result_file="$RESULTS_DIR/$TARGET" \
  "$@"
