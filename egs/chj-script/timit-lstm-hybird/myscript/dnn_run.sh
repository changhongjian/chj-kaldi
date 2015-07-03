#!/bin/bash
. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e 
#set -o errexit  只要某个语句返回值不为0则退出
# Acoustic model parameters

stage=0 
[ -n "$1"  ] && stage=$1  #可以调用从哪里开始

feats_nj=10
train_nj=30
decode_nj=5

# Config:
datause=data/data_and_ali
mfcc39_index=data/mfcc39

if [ $stage -le 0 ]; then
	dir=$mfcc39_index/train
	utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
#目前着一块还搞不明白为什么要分开
fi


if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  dir=exp/dnn4_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --hid-dim 1024 --rbm-iter 20 $mfcc39_index/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4_pretrain-dbn_dnn
  ali=${datause}
  feature_transform=exp/dnn4_pretrain-dbn/final.feature_transform
  dbn=exp/dnn4_pretrain-dbn/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $mfcc39_index/train_tr90 $mfcc39_index/train_cv10 data/lang $ali $ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
    $datause/graph $mfcc39_index/test $dir/decode_test || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
    $datause/graph $mfcc39_index/dev $dir/decode_dev || exit 1;
fi
echo "运行结束"
exit 0
# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
dir=exp/dnn4_pretrain-dbn_dnn_smbr
srcdir=exp/dnn4_pretrain-dbn_dnn
acwt=0.2

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    $mfcc39_index/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --acwt $acwt \
    --lattice-beam 10.0 --beam 18.0 \
    $mfcc39_index/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 6 iterations of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt \
    --do-smbr true --use-silphones true \
    $mfcc39_index/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 6; do
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $datause/graph $mfcc39_index/test $dir/decode_test_it${ITER} || exit 1
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $datause/graph $mfcc39_index/dev $dir/decode_dev_it${ITER} || exit 1
  done 
fi

echo Success
exit 0
