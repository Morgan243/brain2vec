
#RESULTS_DIR="../results/results_2pair"
#SHADOW_0='20230215_1737_ff27ac5d-20a3-4c3b-9ba7-9e5ed3ab705b.json'
#SHADOW_1='20230215_0907_39966c4e-b105-4cf3-ac5a-ec75984dfa51.json'
#TARGET='20230215_0022_d15fa8a5-77f0-441c-a60a-1cc1768bc6ce.json'

RESULTS_DIR="../../results/results_2pair_50epoch"
SHADOW_0='20230217_1439_0fe41c4c-9ca4-4d76-ade1-b0fab925550d.json'
SHADOW_1='20230216_2135_063426a2-dc66-40d6-8193-28aef0012cad.json'
SHADOW_2='20230217_2055_88b1e454-3f3e-4cdc-af38-7d85aec23c08.json'
SHADOW_3='20230216_1058_ea8acc09-5cf1-46a1-a66f-e4390c4733b0.json'
SHADOW_4='20230217_0239_0daa3fb8-008b-47fa-a717-33efbd0c33f8.json'
SHADOW_5='20230217_1013_15512ce8-942a-4bf7-bf2c-d39d290fdf10.json'

TARGET_0='20230216_0730_6333d547-bb0b-4552-9b9a-af1c08f6196f.json'
TARGET_1='20230216_1811_a33834d5-704f-44fa-8a05-007db7dc463b.json'
TARGET_2='20230217_0709_cd95390a-deea-4e66-b6f2-e8e046499860.json'


python -m brain2vec.experiments.info_leakage_eval \
  --task='shadow_classifier_mi' \
	--pretrained_shadow_model_result_input_0.result_file="$RESULTS_DIR/$SHADOW_0" \
	--pretrained_shadow_model_result_input_1.result_file="$RESULTS_DIR/$SHADOW_1" \
	--pretrained_shadow_model_result_input_2.result_file="$RESULTS_DIR/$SHADOW_2" \
	--pretrained_shadow_model_result_input_3.result_file="$RESULTS_DIR/$SHADOW_3" \
	--pretrained_target_model_result_input_0.result_file="$RESULTS_DIR/$TARGET_0" \
	--pretrained_target_model_result_input_1.result_file="$RESULTS_DIR/$TARGET_1" \
	--pretrained_target_model_result_input_2.result_file="$RESULTS_DIR/$TARGET_2" \
  "$@"
