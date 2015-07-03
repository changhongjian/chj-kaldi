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
train_tool="chj-ctc-lstm-stream"  #训练的程序

#-------------与程序有关的
runname="ctc-blstm-d1-lr0.0008hf0.5_cuda_stream_1-4"

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
batch_size=20

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

echo ============================================================================
echo "Beginning on" `date`
echo ============================================================================
cnt=-4
#文件配置
mkdir -p $dir/log
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
  "" 
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
  ali=$mfcc39_index/targets
  #这种虽然不是直白的数据，但是传给kaldi程序后，它会自动处理
  tr_feats="scp:$tr/feats.scp"
  tr_labels="ark:$ali/train_tr90."$traintype".ark"

  cv_feats="scp:$cv/feats.scp"
  cv_labels="ark:$ali/train_cv10."$traintype".ark"

  #  注意，预训练的时候强制改为1了 ****
  if ((nn_depth>0)); then
	$cuda_cmd $dir/log/pretrain_blstm.log  $myscript/pretrain_blstm_ctc_mullayer.sh  \
	  --train_tool "$train_tool"  --feature_transform $feature_transform  \
	  --learn_rate $learn_rate  --momentum $momentum \
	  --max_iters $pretrain_max_iters --min_iters $min_iters --num_stream 1  \
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
	test_feats="$mfcc39_index/test/feats.scp"  
	test_labels="$mfcc39_index/targets/test.$traintype.ark"
	$myscript/decode-run.sh --nj 20 --cmd "$decode_cmd" --nnet_forward "nnet-lstm-forward" --targets_delay $targets_delay --acwt 0.2 \
    --feature_transform $feature_transform --nnet $dir/final.nnet  \
	--class_frame_counts $num_tgt --decode_tool "chj-decode-ctc-by-pdf"   \
    $test_feats $test_labels $dir/decode_test || exit 1;
	
#	dev_feats="$mfcc39_index/dev/feats.scp"
#    dev_labels="$mfcc39_index/targets/dev.$traintype.ark"
#    $myscript/decode-run.sh --nj 20 --cmd "$decode_cmd" --nnet_forward "nnet-lstm-forward" --targets_delay $targets_delay --acwt 0.2 \
#    --feature_transform $feature_transform --nnet $dir/final.nnet  \
#    --class_frame_counts $num_tgt --decode_tool "chj-decode-ctc-by-pdf"   \
#    $dev_feats $dev_labels $dir/decode_dev || exit 1;
fi

cnt=$[$cnt+1]
if [ $stage -le $cnt ]; then
    #decode_script="$myscript/decode-run.sh"
    decode_script="$myscript/decode-run-beamsearch.sh"
    test_feats="$mfcc39_index/test/feats.scp"
    test_labels="$mfcc39_index/targets/test.$traintype.ark"

	char_int_map="$mfcc39_index/targets/phone-int.map"
	arpa_lmfile="$mfcc39_index/targets/train.phone.lm3"
    lm_order=3
    $decode_script --nj 20 --cmd "$decode_cmd" --nnet_forward "nnet-lstm-forward" --targets_delay $targets_delay --acwt 0.2 \
    --feature_transform $feature_transform --nnet $dir/final.nnet  \
    --class_frame_counts $num_tgt --decode_tool "chj-ctc-decode-beam-search-use-int-LM" --lm_order $lm_order  \
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
