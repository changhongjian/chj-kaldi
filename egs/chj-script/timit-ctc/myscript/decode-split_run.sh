#!/bin/bash

nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
class_frame_counts= # non-default location of PDF counts (optional)

stage=0 # stage=1 skips lattice generation
nj=4
cmd=run.pl

nnet_forward="nnet-lstm-forward"  #长虹剑自己添加
targets_delay=0
apply_log=false

num_threads=1 # if >1, will use latgen-faster-parallel
parallel_opts="-pe smp $((num_threads+1))" # use 2 CPUs (1 DNN-forward, 1 decoder)
use_gpu="no" # yes|no|optionaly
acwt=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "parameter error"
   exit 1;
fi

feats=$1
labels=$2
dir=$3

data=$dir/hd_dir
mkdir -p $dir/log
mkdir -p $hd_dir
#data=$dir/splithandle
cp $feats $data/feats.scp  #change to my dir 
sdata=$data/split$nj;
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads" 

feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"


if [ $stage -le 0 ]; then
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode.JOB.log \
    $nnet_forward --feature-transform=$feature_transform --targets-delay=$targets_delay --use-gpu=$use_gpu \
	 --apply-log=$apply_log $feats $labels "ark,t:$sdata/output.JOB.ark"  \
	  || exit 1;
fi

# Run the scoring
if ! $skip_scoring ; then

fi

exit 0;
