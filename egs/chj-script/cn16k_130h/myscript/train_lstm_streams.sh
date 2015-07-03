#!/bin/bash
#2015-3-11 修改李杰师兄的

# training options
learn_rate=0.008  #学习速率
momentum=0 #冲量
l1_penalty=0
l2_penalty=0

# data processing
num_stream=4  #几句话一起训练
batch_size=20 #每句话的batch
targets_delay=5 #label延时
dump_interval=10000

# learn rate scheduling
max_iters=30
min_iters=
start_halving_inc=0.5
end_halving_inc=0.1
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5

## add by JLi @ 20140524  主要是关于如何处理退火的
newbob_acc=true    # learn rate schedule, when true, annealing according to frame accuracy, else according to average loss. 
## end add 
# misc.
verbose=1 #不懂  adj. 冗长的；啰嗦的
train_tool=bd-nnet-train-lstm-streams #训练工具
feature_transform=  #特征网络的位置
echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 

. utils/parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 <mlp_init> <tr_feats> <tr_labels> <cv_feats> <cv_labels> <exp-dir>"
   echo " e.g.: $0 "
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi
#原版本中的nnet_init改为mlp_init 
mlp_init=$1
tr_feats=$2
tr_labels=$3
cv_feats=$4
cv_labels=$5
dir=$6 #训练的文件夹

 # for mean-var normalisation(AddShift & Rescale component)
#nnet_init=$dir/nnet.init

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

# Skip training  如果存在就不重新训练
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

# choose mlp to start with

mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*} #最后一个/之后 .之前
# optionally resume training from the best epoch
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

#cp $mlp_best $dir/nnet/${mlp_base}_iter0 #也就是给nnet_init换了一个名字
#$mlp_best=$dir/nnet/${mlp_base}_iter0
echo "train lstm iter00"
log=$dir/log/iter00.initial.log; hostname>$log
$train_tool \
	--cross-validate=true \
	--learn-rate=$learn_rate \
	--momentum=$momentum \
	--feature-transform=$feature_transform \
	--num-stream=$num_stream \
	--batch-size=$batch_size \
	--targets-delay=$targets_delay \
	--dump-interval=$dump_interval \
	--verbose=$verbose \
	$cv_feats "$cv_labels" $mlp_best  \
	2>> $log || exit 1;
#应该是预训练
	
loss=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
loss_type=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"


# resume lr-halving
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)

echo "进入训练循环中..."
#training
for iter in $(seq -w  $max_iters); do
	echo -n "ITERATION $iter: "
	mlp_next=$dir/nnet/${mlp_base}_iter${iter}  #要传给的下一个网络
    # skip iteration if already done
    [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
    
	log=$dir/log/iter${iter}.tr.log; hostname>$log
    # train
    $train_tool \
        --feature-transform=$feature_transform \
        --learn-rate=$learn_rate \
        --momentum=$momentum \
        --num-stream=$num_stream \
        --batch-size=$batch_size \
        --targets-delay=$targets_delay \
        --dump-interval=$dump_interval \
        --verbose=$verbose \
        $tr_feats "$tr_labels" $mlp_best $mlp_next \
		2>> $log || exit 1;
        
	tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
	echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "
	  
	## add by JLi @ 20140524
	tr_acc=$(cat $dir/log/iter${iter}.tr.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); exit; } END{print acc}')
	echo -n "TRAIN ACCURACY $(printf "%.2f" $tr_acc), (lrate $(printf "%.6g" $learn_rate)), "
	## end add. 
	
	 # cross-validation
    log=$dir/log/iter${iter}.cv.log; hostname>$log
    # validate
    $train_tool \
        --cross-validate=true \
        --learn-rate=$learn_rate \
        --momentum=$momentum \
        --feature-transform=$feature_transform \
        --num-stream=$num_stream \
        --batch-size=$batch_size \
        --targets-delay=$targets_delay \
        --dump-interval=$dump_interval \
        --verbose=$verbose \
        $cv_feats "$cv_labels" $mlp_next  \
        2>> $log || exit 1;

	loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
    echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

    ## add by JLi @ 20140524
    acc_new=$(cat $dir/log/iter${iter}.cv.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); exit; } END{print acc}')
    echo -n "CROSSVAL ACCURACY $(printf "%.2f" $acc_new), "
    ## end add. 
	
#分别是针对准确率 和 loss 来做的
  if [ $newbob_acc == "true" ]; then
    echo "Learn rate shceduling: newbob according to frame accuracy."
    acc_prev=$acc
    if [ "1" == "$(awk "BEGIN{print($acc_new>$acc);}")" ]; then
      acc=$acc_new
      mlp_best=$dir/nnet/${mlp_base}_iter${iter}_lr${learn_rate}_tr$(printf "%.2f" $tr_acc)_cv$(printf "%.2f" $acc_new)
      mv $mlp_next $mlp_best
      echo "nnet accepted ($(basename $mlp_best))"
      echo $mlp_best > $dir/.mlp_best 
    else
      mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_lr${learn_rate}_tr$(printf "%.2f" $tr_acc)_cv$(printf "%.2f" $acc_new)_rejected
      mv $mlp_next $mlp_reject
      echo "nnet rejected ($(basename $mlp_reject))"
    fi

    # create .done file as a mark that iteration is over
    touch $dir/.done_iter$iter

    # stopping criterion
    if [[ "1" == "$halving" && "1" == "$(awk "BEGIN{print($acc < $acc_prev+$end_halving_inc)}")" ]]; then
      if [[ "$min_iters" != "" ]]; then
        if [ $min_iters -gt $iter ]; then
          echo we were supposed to finish, but we continue, min_iters : $min_iters
          continue
        fi
      fi
      echo finished, too small rel. improvement $(awk "BEGIN{print($acc-$acc_prev)}")
      break
    fi

    # start annealing when improvement is low
    if [ "1" == "$(awk "BEGIN{print($acc < $acc_prev+$start_halving_inc)}")" ]; then
      halving=1
      echo $halving >$dir/.halving
    fi

    # do annealing
    if [ "1" == "$halving" ]; then
      learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
      echo $learn_rate >$dir/.learn_rate
    fi
  else
    # accept or reject new parameters (based on objective function)
    echo "Learn rate shceduling: newbob according to objective function."
    loss_prev=$loss
    if [ "1" == "$(awk "BEGIN{print($loss_new<$loss);}")" ]; then
      loss=$loss_new
      mlp_best=$dir/nnet/${mlp_base}_iter${iter}_lr${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
      mv $mlp_next $mlp_best
      echo "nnet accepted ($(basename $mlp_best))"
      echo $mlp_best > $dir/.mlp_best 
    else
      mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_lr${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
      mv $mlp_next $mlp_reject
      echo "nnet rejected ($(basename $mlp_reject))"
    fi

    # create .done file as a mark that iteration is over
    touch $dir/.done_iter$iter

    # stopping criterion
    if [[ "1" == "$halving" && "1" == "$(awk "BEGIN{print(($loss_prev - $loss)/$loss_prev < $end_halving_impr)}")" ]]; then
      if [[ "$min_iters" != "" ]]; then
        if [ $min_iters -gt $iter ]; then
          echo we were supposed to finish, but we continue, min_iters : $min_iters
          continue
        fi
      fi
      echo finished, too small rel. improvement $(awk "BEGIN{print(($loss_prev-$loss)/$loss_prev)}")
      break
    fi

    # start annealing when improvement is low
    if [ "1" == "$(awk "BEGIN{print(($loss_prev-$loss)/$loss_prev < $start_halving_impr)}")" ]; then
      halving=1
      echo $halving >$dir/.halving
    fi
    
    # do annealing
    if [ "1" == "$halving" ]; then
      learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
      echo $learn_rate >$dir/.learn_rate
    fi

  fi ## newbob=true
	
done

# select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  echo "Error training neural network..."
  exit 1
fi

sleep 3
exit 0
