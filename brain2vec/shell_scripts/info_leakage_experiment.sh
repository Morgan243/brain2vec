CMD=./test_info_leakage_experiment.sh
time $CMD --task=membership_inference --task.device='cuda:0' --task.n_epochs=150 \
  --task.early_stopping_patience=30 \
  --task.lr_adjust_patience=15 \
  --batch_size=1028 \
  --batch_norm=False  \
  --linear_hidden_n=512 \
  --n_layers=2 \
 --task.learning_rate=0.001 \
 --dropout=0.5 \
 --fine_tuning_method='1d_linear' \
 --flatten_sensors_to_samples=True \
 --task.method='1d_linear' \
 --result_dir=$RESULT_PATH

time $CMD --task=membership_inference --task.device='cuda:0' --task.n_epochs=150 \
  --task.early_stopping_patience=30 \
  --task.lr_adjust_patience=15 \
  --batch_size=1028 \
  --batch_norm=False  \
  --linear_hidden_n=256 \
  --n_layers=2 \
 --task.learning_rate=0.001 \
 --dropout=0.5 \
 --fine_tuning_method='1d_linear' \
 --flatten_sensors_to_samples=True \
 --task.method='1d_linear' \
 --result_dir=$RESULT_PATH


time $CMD --task=membership_inference --task.device='cuda:0' --task.n_epochs=150 \
  --task.early_stopping_patience=30 \
  --task.lr_adjust_patience=15 \
  --batch_size=1028 \
  --batch_norm=True  \
  --linear_hidden_n=512 \
  --n_layers=2 \
 --task.learning_rate=0.001 \
 --dropout=0.5 \
 --fine_tuning_method='1d_linear' \
 --flatten_sensors_to_samples=True \
 --task.method='1d_linear' \
 --result_dir=$RESULT_PATH

time $CMD --task=membership_inference --task.device='cuda:0' --task.n_epochs=150 \
  --task.early_stopping_patience=30 \
  --task.lr_adjust_patience=15 \
  --batch_size=1028 \
  --batch_norm=True  \
  --linear_hidden_n=256 \
  --n_layers=2 \
 --task.learning_rate=0.001 \
 --dropout=0.5 \
 --fine_tuning_method='1d_linear' \
 --flatten_sensors_to_samples=True \
 --task.method='1d_linear' \
 --result_dir=$RESULT_PATH




time $CMD --task=membership_inference --task.device='cuda:0' --task.n_epochs=150 \
  --task.early_stopping_patience=30 \
  --task.lr_adjust_patience=15 \
  --batch_size=1028 \
  --batch_norm=True  \
  --linear_hidden_n=512 \
  --n_layers=3 \
 --task.learning_rate=0.001 \
 --dropout=0.5 \
 --fine_tuning_method='1d_linear' \
 --flatten_sensors_to_samples=True \
 --task.method='1d_linear' \
 --result_dir=$RESULT_PATH

time $CMD --task=membership_inference --task.device='cuda:0' --task.n_epochs=150 \
  --task.early_stopping_patience=30 \
  --task.lr_adjust_patience=15 \
  --batch_size=1028 \
  --batch_norm=True  \
  --linear_hidden_n=256 \
  --n_layers=3 \
 --task.learning_rate=0.001 \
 --dropout=0.75 \
 --fine_tuning_method='1d_linear' \
 --flatten_sensors_to_samples=True \
 --task.method='1d_linear' \
 --result_dir=$RESULT_PATH




