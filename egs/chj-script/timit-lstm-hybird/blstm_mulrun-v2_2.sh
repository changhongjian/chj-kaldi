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
datause=/data/zyou/wpr/software/kaldi-trunk/egs/timit/mfcc_exp/data/data_and_ali  #data/data_and_ali
datause=data/data_and_ali

nnet_proto=data/nnet_proto #存放网络结构的目录
mfcc13_index=data/mfcc13_index
mfcc39_index=data/mfcc39
timit_mfcc_org="/data/zyou/wpr/data/TIMIT/mfcc" # 我固定了mfcc的位置
train_tool="chj_lstm_2"  #训练的程序

#-------------与程序有关的
runname="blstm_s12_v2-2_2015-7-3-8_d1cec400r256sp0tg0b60lr0.001hf0.3"
f_nnet_proto=""

dir=$myexp/$runname 
mlp_init=$dir/nnet.init #初始化的网络
feature_transform=$dir/final.feature_transform #特征的传递网络
#拼帧
splice="0_0"
splice_step=0
targets_delay=0

nn_depth=1 #多层lstm
num_fea=$((39*1))  #这个是经过特征变换后的
num_cells=$(( 800/2 ))
num_streams=12
num_recurrent_neurons=$(( 256 * 2 ))  #用了新的blstm

num_stream=$num_streams
batch_size=60

learn_rate=0.001
momentum=0 #冲量
halving_factor=0.3
max_iters=15
min_iters=0

pretrain_max_iters=2 #预训练最大迭代多少

newbob_acc=true #退火的时候是否按照 帧准确率
	
start_halving_inc=0.5
end_halving_inc=0.1
start_halving_impr=0.01
end_halving_impr=0.001

echo ============================================================================
echo "Beginning on" `date`
echo ============================================================================
cnt=-4
#文件配置
mkdir -p $dir/log
if [ ! -f $dir/nnet.proto ];then 
  echo "复制nnet.proto文件"
  tmp=$nnet_proto/${f_nnet_proto}.nnet.proto 
  if [ ! -f $tmp ];then
     tmp=$nnet_proto/${runname}.nnet.proto
     if [ ! -f $tmp ];then
        tmp=$nnet_proto/nnet.proto
     fi
  fi
  #cp $tmp $dir/nnet.proto
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
  #首先要获得训的特征和标签
  tr=$mfcc39_index/train_tr90
  cv=$mfcc39_index/train_cv10
  ali=$datause
  #这种虽然不是直白的数据，但是传给kaldi程序后，它会自动处理
  tr_feats="scp:$tr/feats.scp"
  tr_labels="ark:gunzip -c $ali/ali.*.gz | ali-to-pdf $ali/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |"

  cv_feats="scp:$cv/feats.scp"
  cv_labels="ark:gunzip -c $ali/ali.*.gz | ali-to-pdf $ali/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |"

  #初始化网络 本应该  对于单层和多层有不同的处理方式
  if ((nn_depth>0)); then
	$cuda_cmd $dir/log/pretrain_blstm.log  $myscript/pretrain_blstm_mullayer.sh  \
	  --train_tool "$train_tool"  --feature_transform $feature_transform  \
	  --learn_rate $learn_rate  --momentum $momentum \
	  --max_iters $pretrain_max_iters --min_iters $min_iters --num_stream $num_stream  \
	  --batch_size $batch_size --targets_delay $targets_delay  --newbob_acc $newbob_acc \
	  --halving_factor $halving_factor --start_halving_inc $start_halving_inc  --end_halving_inc $end_halving_inc  \
	  --start_halving_impr $start_halving_impr --end_halving_impr $end_halving_impr \
	  --nn_depth $nn_depth --num_fea $num_fea --num_cells $num_cells --num_streams $num_streams --num_recurrent_neurons $num_recurrent_neurons \
		"$mlp_init" "$tr_feats" "$tr_labels" "$cv_feats" "$cv_labels" "$dir" \
		|| exit 1;
  else
        echo "使用nnet-initialize"
	nnet-initialize --binary=true $dir/nnet.proto $mlp_init || exit 1
          
  fi


  $cuda_cmd $dir/log/train_lstm.log  $myscript/train_lstm_streams.sh  \
  --train_tool "$train_tool"  --feature_transform $feature_transform  \
  --learn_rate $learn_rate  --momentum $momentum  \
  --max_iters $max_iters --min_iters $min_iters --num_stream $num_stream  \
  --batch_size $batch_size --targets_delay $targets_delay  --newbob_acc $newbob_acc \
  --halving_factor $halving_factor --start_halving_inc $start_halving_inc  --end_halving_inc $end_halving_inc  \
  --start_halving_impr $start_halving_impr --end_halving_impr $end_halving_impr \
    "$mlp_init" "$tr_feats" "$tr_labels" "$cv_feats" "$cv_labels" "$dir" \
	|| exit 1;
	
fi

cnt=$[$cnt+1]
# need 4 files final.nnet final.mdl  feature_transform.nnet.txt ali_train_pdf.counts 
if [ $stage -le $cnt ]; then
  # Decode (reuse HCLG graph)
  $myscript/decode.sh --nj 20 --cmd "$decode_cmd" --nnet_forward "nnet-lstm-forward" --targets_delay $targets_delay --acwt 0.2 \
    --feature_transform $feature_transform --nnet $dir/final.nnet  \
    --model $datause/final.mdl --class_frame_counts $datause/ali_train_pdf.counts \
    $datause/graph $mfcc39_index/test $dir/decode_test || exit 1;
  $myscript/decode.sh --nj 20 --cmd "$decode_cmd" --nnet_forward "nnet-lstm-forward" --targets_delay $targets_delay --acwt 0.2 \
    --feature_transform $feature_transform --nnet $dir/final.nnet  \
    --model $datause/final.mdl --class_frame_counts $datause/ali_train_pdf.counts \
    $datause/graph $mfcc39_index/dev $dir/decode_dev || exit 1;
fi
cnt=$[$cnt+1]
if [ $stage -le $cnt  ]; then
   echo " 查看运行结果 "
   $myscript/results.sh | grep $runname
fi

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0
