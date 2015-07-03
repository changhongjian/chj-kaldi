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
train_tool="chj-ctc-lstm-stream1"  #训练的程序

#-------------与程序有关的
runname="ctc-d1-lr0.0008hf0.5_cuda_stream_4"

traintype="phone"
num_tgt=49
dir=$myexp/$runname 
mlp_init=$dir/nnet.init #初始化的网络
feature_transform=$dir/final.feature_transform #特征的传递网络
#拼帧
splice="0_5"
splice_step=1
targets_delay=0

nn_depth=1 #多层lstm
num_cells=800
num_streams=4
num_recurrent_neurons=500

num_stream=$num_streams
batch_size=20 #没用

learn_rate=0.0008
momentum=0 #冲量
halving_factor=0.5
max_iters=15
min_iters=0

pretrain_max_iters=2 #预训练最大迭代多少

newbob_acc=false #退火的时候是否按照 帧准确率 false-loss
	
start_halving_inc=0.5
end_halving_inc=0.1
start_halving_impr=0.01
end_halving_impr=0.001

nn_depths=(1)
streams=(4 8 12 16)
learn_rates=(0.0005 0.0001 0.00008 0.00005 0.00001)

for nn_depth in ${nn_depths[@]}; do
	
for num_streams in ${streams[@]}; do
num_stream=$num_streams

for learn_rate in ${learn_rates[@]}; do 

runname="manyexps-d${nn_depth}-s${num_streams}-lr${learn_rate}"
dir=$myexp/$runname
mlp_init=$dir/nnet.init #初始化的网络
feature_transform=$dir/final.feature_transform #特征的传递网络



echo ============================================================================
echo "Beginning on" `date`
echo ============================================================================
cnt=-1
#文件配置
mkdir -p $dir/log
#到这里应该为0
cnt=$[$cnt+1]
if [ $stage -le $cnt ];then
  echo " 处理 feature_transform  "
  $cuda_cmd $dir/log/pretrain_feat_trans.log \
     $myscript/pretrain_feat_trans.sh --copy_feats true --apply_cmvn true \
	--splice $splice --splice_step $splice_step \
       $mfcc39_index/train $dir || exit 1;
fi
cnt=$[$cnt+1]
if [ $stage -le $cnt ]; then
  echo "开始训练 $dir "
  #首先要获得训的特征和标签
  tr=$mfcc39_index/train
  cv=$mfcc39_index/dev
  ali=$mfcc39_index/targets
  #这种虽然不是直白的数据，但是传给kaldi程序后，它会自动处理
  tr_feats="scp:$tr/feats.scp"
  tr_labels="ark:$ali/train."$traintype".ark"

  cv_feats="scp:$cv/feats.scp"
  cv_labels="ark:$ali/dev."$traintype".ark"

  #  注意，预训练的时候强制改为1了 ****
  if ((nn_depth>0)); then
	$cuda_cmd $dir/log/pretrain_lstm.log  $myscript/pretrain_lstm_ctc_mullayer.sh  \
	  --train_tool "$train_tool"  --feature_transform $feature_transform  \
	  --learn_rate $learn_rate  --momentum $momentum \
	  --max_iters $pretrain_max_iters --min_iters $min_iters --num_stream $num_stream  \
	  --batch_size $batch_size --targets_delay $targets_delay  --newbob_acc $newbob_acc \
	  --halving_factor $halving_factor --start_halving_inc $start_halving_inc  --end_halving_inc $end_halving_inc  \
	  --start_halving_impr $start_halving_impr --end_halving_impr $end_halving_impr \
	  --nn_depth $nn_depth --num_cells $num_cells --num_streams $num_streams --num_recurrent_neurons $num_recurrent_neurons \
	  --num_tgt $num_tgt \
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

if [ $stage -le $cnt ]; then
	echo "开始解码 $dir "
	test_feats="$mfcc39_index/test/feats.scp"  
	test_labels="$mfcc39_index/targets/test_48_min.ark"
	arpa_lmfile="$mfcc39_index/targets/train.phone.lm"
	$myscript/decode-run-manyexps.sh --nj 20 --cmd "$decode_cmd" --nnet_forward "nnet-lstm-forward" --targets_delay $targets_delay --acwt 0.2 \
    --feature_transform $feature_transform --nnet $dir/final.nnet  \
	--class_frame_counts $num_tgt --decode_tool "chj-decode-ctc-by-pdf"   \
    $test_feats $test_labels $arpa_lmfile $dir/decode_test || exit 1;
	
fi

cnt=$[$cnt+1]
if [ $stage -le $cnt  ]; then
   echo " 查看运行结果 "
   #$myscript/results.sh | grep $runname
fi

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

stage=0
#for the iter
done
done
done

exit 0
