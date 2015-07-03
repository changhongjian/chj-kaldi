#!/bin/bash
#. ./path.sh  # 那些类似 $() 样的进程，必须要加这个环境变量

#  **** this is changed at 2015-5-2 ,when I train cn , it become simple but is not compatible with the org file 

#data=data/train
#dir=exp/my_pre_dbn
copy_feats=true
copy_feats_tmproot=   #后面那些X的数目是固定的 tmp_train.scp.XXXXXX 不写就在tmp下面
apply_cmvn=true
splice=0
splice_step=0 #对于splice的处理没有修改 那个python文件直接被我跳过了

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;
data=$1
dir=$2

for f in $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log
###### PREPARE FEATURES ######
echo "# PREPARING FEATURES"
# shuffle the list
echo "Preparing train/cv lists"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
# print the list size
wc -l $dir/train.scp

# re-save the shuffled features, so they are stored sequentially on the disk in /tmp/
if [ "$copy_feats" == "true" ]; then
  tmpdir=$(mktemp -d $copy_feats_tmproot); mv $dir/train.scp{,_non_local}
#保存一下，因为后面train.scp就会被改变了
  copy-feats scp:$dir/train.scp_non_local ark,scp:$tmpdir/train.ark,$dir/train.scp || exit 1
  trap "echo \"Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT
#出现错误该如何处理 也就是程序意外退出后，执行这个命令
#如果不使用trap 以后可以用 copy-feats ark:$tmpdir/train.ark ark,t:- | less 看到
fi

# create a 10k utt subset for global cmvn estimates  只取前面一些作CMVN
head -n 10000 $dir/train.scp > $dir/train.scp.10k

###### PREPARE FEATURE PIPELINE ######

# read the features
feats="ark:copy-feats scp:$dir/train.scp ark:- |"

#feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
# optionally add per-speaker CMVN
if [ $apply_cmvn == "true" ]; then
  echo "Will use CMVN statistics : $data/cmvn.scp"
  [ ! -r $data/cmvn.scp ] && echo "Cannot find cmvn stats $data/cmvn.scp" && exit 1;
  cmvn="scp:$data/cmvn.scp"
  feats="$feats apply-cmvn  --utt2spk=ark:$data/utt2spk $cmvn ark:- ark:- |"
  #feats="$feats add-deltas ark:- ark:- |"
else
  echo "apply_cmvn disabled (per speaker norm. on input features)"
fi

#echo $feats
# get feature dim
echo -n "Getting feature dim : "
feat_dim=$(feat-to-dim "$feats" -) #--print-args=false
echo $feat_dim

# Now we will start building feature_transform which will 
# be applied in CUDA to gain more speed.
#
# We will use 1GPU for both feature_transform and MLP training in one binary tool. 
# It is necessary, because we need to run it as a single process, using single GPU
# and avoiding I/O overheads.

if [  -f "$dir/final.feature_transform" ]; then
  echo Warning: Using already prepared file  "--->" "$dir/final.feature_transform"
else
  echo "Using splice +/- $splice , step $splice_step"
  feature_transform=$dir/tr_splice$splice-$splice_step.nnet
  if (( splice_step==0 )) ; then  #被我大换血了
	echo "happy -- 不用splice"
	compute-cmvn-stats "$(echo $feats | sed 's|train.scp|train.scp.10k|')" - | cmvn-to-nnet - $feature_transform
  else
    echo "Generate the splice transform "
#以下是我自己组装成的格式
    splice=(${splice//_/ })
    bg=${splice[0]}
    ed=${splice[1]}
    cnt=0;
    echo -n "<Splice> ">$feature_transform
    str="["
    for (( i=$bg;i<=$ed;i+=$splice_step )) ; do
       str=$str" $i"
       ((cnt++))
    done
    str=$str" ]"
    i=$(( cnt * feat_dim  ))
    echo -n $i" " >>$feature_transform
    echo $feat_dim >>$feature_transform
    echo $str >>$feature_transform
    # utils/nnet/gen_splice.py --fea-dim=$feat_dim --splice=$splice --splice-step=$splice_step > $feature_transform
	# Renormalize the MLP input to zero mean and unit variance
	  feature_transform_old=$feature_transform
	  feature_transform=${feature_transform%.nnet}_cmvn-g.nnet
	  echo "Renormalizing MLP input features into $feature_transform"
	  nnet-forward --use-gpu=yes \
		$feature_transform_old "$(echo $feats | sed 's|train.scp|train.scp.10k|')" \
		ark:- 2>$dir/log/cmvn_glob_fwd.log |\
	  compute-cmvn-stats ark:- - | cmvn-to-nnet - - |\
	  nnet-concat --binary=false $feature_transform_old - $feature_transform
  fi
  [ -f $dir/final.feature_transform ] && unlink $dir/final.feature_transform #有删除功能
  (cd $dir; ln -s $(basename $feature_transform) final.feature_transform )
fi

echo -n "最终的维度:"
echo $(feat-to-dim --print-args=false "$feats nnet-forward --use-gpu=no $feature_transform ark:- ark:- |" - 2>/dev/null)
echo "特征处理运行成功"
exit 0

