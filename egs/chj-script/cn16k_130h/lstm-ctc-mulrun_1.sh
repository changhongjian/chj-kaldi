#!/bin/bash
. ./cmd.sh 
. ./path.sh ## Source the tools/utils (import the queue.pl)
set -e
stage=0 
[ -n "$1"  ] && stage=$1
#echo $stage
# Config:
myexp=exp
myscript=myscript
feats_index=data/datafeats
traintype="char"
num_tgt=6725    #网络最后一层节点的数量   *** 记住要加1
# --  关于scp的索引地址的修改这里不再提供 , tr_90和 cv_10也不管了

train_tool="chj-ctc-lstm-stream1"  #训练的程序

#-------------与程序有关的
runname="ctc-d1-lr0.0001hf0.5_cuda_stream_4"

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
batch_size=20  # not use

learn_rate=0.0001
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

echo ============================================================================
echo "Beginning on" `date`
echo ============================================================================

cnt=-1
#文件配置
mkdir -p $dir/log
#分阶段处理

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
       $feats_index/train $dir || exit 1;
fi
cnt=$[$cnt+1]
if [ $stage -le $cnt ]; then
  echo "开始训练 $dir "
  #首先要获得训的特征和标签
  tr=$feats_index/train
  cv=$feats_index/dev     #记住这里于test是共用的
  ali=$feats_index/targets
  #这种虽然不是直白的数据，但是传给kaldi程序后，它会自动处理
  tr_feats="scp:$tr/feats.scp"
  tr_labels="ark:$ali/train."$traintype".ark"

  cv_feats="scp:$cv/feats.scp"
  cv_labels="ark:$ali/test."$traintype".ark"

  #初始化网络 本应该  对于单层和多层有不同的处理方式
  if ((nn_depth>0)); then
	$cuda_cmd $dir/log/pretrain_lstm.log  $myscript/pretrain_lstm_ctc_mullayer.sh  \
	  --train_tool "$train_tool"  --feature_transform $feature_transform  \
	  --learn_rate $learn_rate  --momentum $momentum \
	  --max_iters $pretrain_max_iters --min_iters $min_iters --num_stream $num_stream  \
	  --batch_size $batch_size --targets_delay $targets_delay  --newbob_acc $newbob_acc \
	  --halving_factor $halving_factor --start_halving_inc $start_halving_inc  --end_halving_inc $end_halving_inc  \
	  --start_halving_impr $start_halving_impr --end_halving_impr $end_halving_impr \
	  --nn_depth $nn_depth  --num_cells $num_cells --num_streams $num_streams --num_recurrent_neurons $num_recurrent_neurons \
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
	test_feats="$feats_index/test/feats.scp"  
	test_labels="$feats_index/targets/test.$traintype.ark"
	$myscript/decode-run.sh --nj 20 --cmd "$decode_cmd" --nnet_forward "nnet-lstm-forward" --targets_delay $targets_delay --acwt 0.2 \
    --feature_transform $feature_transform --nnet $dir/final.nnet  \
	 --decode_tool "chj-decode-ctc-by-pdf"   \
    $test_feats $test_labels $dir/decode_test || exit 1;
	
fi

cnt=$[$cnt+1]
if [ $stage -le $cnt ]; then
    #decode_script="$myscript/decode-run.sh"
    decode_script="$myscript/decode-run-beamsearch.sh"
    test_feats="$feats_index/test/feats.scp"
    test_labels="$feats_index/targets/test.$traintype.ark"

	char_int_map="$feats_index/targets/$traintype-int.map"
	arpa_lmfile="$feats_index/targets/train.$traintype.lm2"
    $decode_script --nj 20 --cmd "$decode_cmd" --nnet_forward "nnet-lstm-forward" --targets_delay $targets_delay --acwt 0.2 \
    --feature_transform $feature_transform --nnet $dir/final.nnet  \
     --decode_tool "chj-ctc-decode-beam-search-use-int-LM"  \
    $test_feats $test_labels  $char_int_map $arpa_lmfile $dir/decode_test  || exit 1;


fi


cnt=$[$cnt+1]
if [ $stage -le $cnt  ]; then
   echo " 查看运行结果 "
   #$myscript/results.sh | grep $runname
fi

echo ============================================================================
echo "Finished successfully on" `date`
echo ============================================================================

exit 0
