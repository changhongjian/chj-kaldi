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
rundir=`pwd`
src_index=$rundir/data/featsorg
des_index=$rundir/data/featschg

dir=$myexp/$runname
#feat-to-dim ark:feats.ark - 查看最后生成的维度 
  for d in train dev test; do
	 mkdir -p $des_index/$d 
	 copy-feats "ark,s,cs:apply-cmvn --utt2spk=ark:$src_index/$d/utt2spk scp:$src_index/$d/cmvn.scp scp:$src_index/$d/feats.scp ark:- | add-deltas ark:- ark:- |" \
		 ark,scp:$des_index/$d/feats.ark,$des_index/$d/feats.scp || exit 1;
#注意 格式必须为 ark,scp 不能错  下面这个就是给出特征计算 cmvn 
     compute-cmvn-stats scp:$des_index/$d/feats.scp ark,scp:$des_index/$d/cmvn.ark,$des_index/$d/cmvn.scp
     echo "finish mlp42  $d" 
  done
exit 0
