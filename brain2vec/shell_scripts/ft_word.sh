# Placeholder - which parts to parameterize?
#--input_results_dir ../results_local/results_pretrain_1pair_230507/ \
#--result_dir=../results_local/ft_speech_230530 \
#--input_results_query='quant_num_vars == 10 and n_encoder_layers== 4 and (train_sets != "UCSD-4")' \
#--input_results_query='quant_num_vars == 10 and n_encoder_layers== 4 and ("UCSD-4" in train_sets)' \

python -m brain2vec.grid.grid_on_results \
--input_results_query='quant_num_vars == 20 and n_encoder_layers== 8' \
--experiment_component_grids_str='{}' \
--result_file='dummy' \
--task=word_detection \
--sample_tuples_for=train_sets \
--sample_to_skip='UCSD-4' \
--sample_choose_n=1 \
--extra_output_keys='sensor_ras_coord_arr' \
--n_dl_workers=0 \
--n_dl_eval_workers=0 \
--dropout=0.75 \
--n_layers=2 \
--batch_size=1024 \
--batch_size_eval=2048 \
--lr_adjust_patience=10 \
--early_stopping_patience=15 \
"${@}"


