#!/bin/bash
. ./cmd.sh 
. ./path.sh ## Source the tools/utils (import the queue.pl)
set -e
stage=0 # resume training with --stage=N
[ -n "$1"  ] && stage=$1
#echo $stage
# Config:
myexp=exp
myscript=myscript
datause=data/data_and_ali
nnet_proto=data/nnet_proto #存放网络结构的目录
mfcc13_index=data/mfcc13_index
mfcc39_index=data/mfcc39
timit_mfcc_org="/data/zyou/wpr/data/TIMIT/mfcc" # 我固定了mfcc的位置

#-------------与程序有关的
runname=lstm_d2
dir=$myexp/$runname 
splice=0
splice_step=0

echo ============================================================================
echo "Beginning on" `date`
echo ============================================================================
cnt=-4
#文件配置
mkdir -p $dir/log
if [ ! -f $dir/nnet.proto ];then 
  echo "复制nnet.proto文件"
  tmp=$nnet_proto/${runname}.nnet.proto 
  if [ ! -f $tmp ];then
     tmp=$nnet_proto/nnet.proto
  fi
  cp $tmp $dir/nnet.proto
fi
if [ ! -f $dir/final.mdl ];then  
  echo "获得变化后的  final.mdl"
  copy-transition-model --binary=false $datause/final.mdl  $dir/final.mdl
fi
#分阶段处理
if [ $stage -le $cnt ]; then
  echo "处理原始 mfcc 路径 DES：$timit_mfcc_org"
  $myscript/handle_mfcc_path.sh $mfcc13_index  $timit_mfcc_org || exit 1
fi
cnt=$[$cnt+1]
if [ $stage -le $cnt ]; then
  echo " 将原始mfcc 转为39维特征  "
  for d in train dev test; do 
     $myscript/mfcc_index_13to39.sh --nj 10 --cmd "$train_cmd"  \
         $mfcc39_index/$d $mfcc13_index/$d $mfcc39_index/$d/log $mfcc39_index/$d/data
     echo "finish mfcc39  $d" 
  done
fi
cnt=$[$cnt+1]
if [ $stage -le $cnt ];then
  echo " 分割数据集 "
  tmp=$mfcc39_index/train 
  utils/subset_data_dir_tr_cv.sh  $tmp ${tmp}_tr90 ${tmp}_cv10 || exit 1
fi
cnt=$[$cnt+1]
if [ $stage -le $cnt ];then
  echo " 生成ali_train_pdf.counts  "
  labels_tr_pdf="ark:ali-to-pdf $datause/final.mdl \"ark:gunzip -c $datause/ali.*.gz |\" ark:- |"
  analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $datause/ali_train_pdf.counts 2>$datause/log/analyze_counts_pdf.log || exit 1
fi
#到这里应该为0
cnt=$[$cnt+1]
if [ $stage -le $cnt ];then
  echo " 处理 feature_transform  "
 # mkdir -p $dir/log
 # (tail --pid=$$ -F $dir/log/pretrain_feat_trans.log 2>/dev/null)& # forward log
 #我这个很长不能一直监视
  $cuda_cmd $dir/log/pretrain_feat_trans.log \
     $myscript/pretrain_feat_trans.sh --copy_feats true --apply_cmvn true \
	--splice $splice --splice_step $splice_step \
       $mfcc39_index/train $dir || exit 1;
fi
cnt=$[$cnt+1]
if [ $stage -le $cnt ]; then
  echo "开始训练 $dir "
  #feature_transform=exp/lstm/final.feature_transform
  # Train
  $cuda_cmd $dir/log/train_lstm.log \
   $myscript/train_lstm_streams.sh  \
    $mfcc39_index/train_tr90 $mfcc39_index/train_cv10 $datause $dir || exit 1;
fi

cnt=$[$cnt+1]
# need 4 files final.nnet final.mdl  feature_transform.nnet.txt ali_train_pdf.counts 
if [ $stage -le $cnt ]; then
  echo "开始解码 $dir"
  nnet-copy --binary=false $dir/final.nnet - | $myscript/hd_lstm_nnet.pl -out $dir/final.nnet.txt  || exit 1
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
    --feature_transform $dir/final.feature_transform --nnet $dir/final.nnet.txt  \
    --model $datause/final.mdl --class_frame_counts $datause/ali_train_pdf.counts \
    $datause/graph $mfcc39_index/test $dir/decode_test || exit 1;
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
    --feature_transform $dir/final.feature_transform --nnet $dir/final.nnet.txt  \
    --model $datause/final.mdl --class_frame_counts $datause/ali_train_pdf.counts \
    $datause/graph $mfcc39_index/dev $dir/decode_dev || exit 1;
fi
cnt=$[$cnt+1]
if [ $stage -le $cnt  ]; then
   $myscript/results.sh  
fi

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0
