#!/bin/bash
#2015-3-11 修改李杰师兄的
#2015-3-13 仿照李杰师兄，增加多层预训练
#思路 1）预训练 2）拼层
# training options
learn_rate=0.008  #学习速率
momentum=0 #冲量
l1_penalty=0
l2_penalty=0

# data processing
num_stream=12  #几句话一起训练 和下面的num_streams 一样，但是为了统一格式就这样了
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


nn_depth=1 #---加入这个深度
num_fea=39 # 应该是不用提供了我程序里自动判断了
script_pos="myscript"
proto_opts=   #这个是给python文件的参数，可以没有
#就是经过feature_transform后的维数，本来可用nnet_forward算出来的，
#但是我这里需要调用程序直接给出
num_cells=800
num_streams=12
num_recurrent_neurons=512
num_tgt=49	

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
mlp_init=$1 #现在这个只是个目录+名字 程序里面生成的为了区别加了_
tr_feats=$2
tr_labels=$3
cv_feats=$4
cv_labels=$5
dir=$6 #训练的文件夹

parentdir=$dir #把真正程序的dir存下来
dir="$dir/pretrain" #扩展一下
work_dir=$PWD
echo "work_dir:"$work_dir

[ ! -d $dir ] && mkdir -p $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

#cp  $parentdir/final.mdl $dir/final.mdl  not use in ctc  ****

###### PERFORM THE PRE-TRAINING ###### 开始预训练

for depth in $(seq 1 $nn_depth); do
	echo -e "\n# 开始预训练的层数 $depth"
	dbn=$dir/$depth.dbn
	[ -f $dbn ] && echo "DBN '$dbn' 已经训练了，跳过" && continue
	mlp_init_=$dir/$depth.nnet.init
	#input-dim 下面这个可能有问题
        num_fea=$(nnet-lstm-forward --do_once=true  $feature_transform "$tr_feats" ark:- | feat-to-dim ark:- - ) 
        #output-dim  算了一次后面就不用再算了
	mlp_proto=$dir/$depth.nnet.proto
	#下面不一定要判断 我是为了保持队形
	if [ "$depth" == "1" ];then
		echo "生成 prototype $mlp_proto"
		lstm_layers=1
		$script_pos/make_lstm_proto.py $proto_opts  \
			$num_fea $num_tgt $lstm_layers $num_cells $num_streams $num_recurrent_neurons > $mlp_proto || exit 1
		# initialize
		mlp_init_tmp=$dir/$depth.nnet.init.tmp
		log=$dir/log/$depth.nnet_initialize.log
		echo "Initializing $mlp_proto -> $mlp_init_tmp"
		nnet-initialize $mlp_proto $mlp_init_tmp 2>$log || { cat $log; exit 1; }

                mv $mlp_init_tmp $mlp_init_
	else
		[ -f $dir/$((depth-1)).dbn ] && num_fea=$(nnet-lstm-forward --do_once=true "nnet-concat $feature_transform $dir/$((depth-1)).dbn -|"  "$tr_feats" ark:- | feat-to-dim ark:- -)
		[ -z "$num_fea" ] && echo "Getting nnet input dimension failed!!" && exit 1
		# make network prototype
		echo "Genrating network prototype $mlp_proto"
		lstm_layers=1  ## to generate the last hidden layer and softmax layer.
		$script_pos/make_lstm_proto.py $proto_opts \
		  $num_fea $num_tgt $lstm_layers $num_cells $num_streams $num_recurrent_neurons > $mlp_proto || exit 1
		# initialize
		mlp_init_tmp=$dir/$depth.nnet.init.tmp
		log=$dir/log/$depth.nnet_initialize.log
		echo "Initializing $mlp_proto -> $mlp_init_tmp"
		nnet-initialize $mlp_proto $mlp_init_tmp 2>$log || { cat $log; exit 1; }
		#nnet-initialize --binary=false $mlp_proto $mlp_init_tmp 2>$log || { cat $log; exit 1; }
		#这个就是合并  上面那个也是先初始化一个网络。然后拼起来
		nnet-concat $dir/$((depth-1)).dbn $mlp_init_tmp $mlp_init_ || exit 1 
		#nnet-concat --binary=false $mlp_init_ 就是这时拼接起来的网络
                rm  $mlp_init_tmp
	fi
        echo "深度  $depth ----训练"	
	# Run several iterations of the MPE/sMBR training
	cur_mdl=$mlp_init_
	if [ -f $dir/$depth.nnet ]; then
		echo "Skipped, file $dir/$depth.nnet exists"
	else
		dir_bk=$dir #备份一下 为了统一格式
		
		dir=$dir/$depth.train
		mkdir -p $dir/{log,nnet}

		$cuda_cmd $dir/log/train_lstm.log  $script_pos/train_lstm_streams.sh  \
		  --train_tool "$train_tool"  --feature_transform $feature_transform  \
		  --learn_rate $learn_rate  --momentum $momentum  \
		  --max_iters $max_iters --min_iters $min_iters --num_stream $num_stream  \
		  --batch_size $batch_size --targets_delay $targets_delay  --newbob_acc $newbob_acc \
		  --halving_factor $halving_factor --start_halving_inc $start_halving_inc  --end_halving_inc $end_halving_inc  \
		  --start_halving_impr $start_halving_impr --end_halving_impr $end_halving_impr \
			"$cur_mdl" "$tr_feats" "$tr_labels" "$cv_feats" "$cv_labels" "$dir" \
			|| exit 1;

		( cd $dir_bk; ln -s $depth.train/final.nnet $depth.nnet; )
		cd $work_dir 
		
		dir=$dir_bk  #还原
		cur_mdl=$dir_bk/$depth.nnet
	fi

	echo "结束 training finished for depth=$depth"
	nnet-copy --remove-last-layers=2 $dir/$depth.nnet $dir/$depth.dbn 2>$dir/log/nn-copy.$depth.log || { cat $dir/log/nn-copy.$depth.log; exit 1; }
	#nnet-copy --binary=false 
done
cur_mdl=$dir/$depth.nnet
echo "cp $cur_mdl ---->- $mlp_init "
cp $cur_mdl $mlp_init  #把预训练的结果装进去
echo "预训练结束---------------"

sleep 3
exit 0
