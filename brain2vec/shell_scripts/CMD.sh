export LD_LIBRARY_PATH=/opt/python3.9/lib:/opt/jdk17/lib:/usr/local/cuda-11.6/lib64:/opt/gcc11/lib64
python -m  brain2vec.debias_pretrain_experiment --n_splits=$N_SPLITS --this_split=$THIS_SPLIT \
  --choose_n_for_pretrain=$CHOOSE_N_FOR_PRETRAIN \
  --device=$DEVICE