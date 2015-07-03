#!/bin/bash

nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
class_frame_counts= # non-default location of PDF counts (optional)

stage=0 # stage=1 skips lattice generation

nnet_forward="nnet-lstm-forward"  #长虹剑自己添加
decode_tool=""
targets_delay=0
apply_log=false

beam_num=30
lm_weight=1.0
lm_order=2


num_threads=1 # if >1, will use latgen-faster-parallel
parallel_opts="-pe smp $((num_threads+1))" # use 2 CPUs (1 DNN-forward, 1 decoder)
use_gpu="no" # yes|no|optionaly
acwt=
#not use
nj=4
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "parameter error"
   exit 1;
fi

feats=$1
labels=$2
char_int_map=$3
arpa_lmfile=$4
dir=$5

data=$dir/hd_dir
mkdir -p $dir/log
mkdir -p $data
#data=$dir/splithandle
if [ ! -f "$data/feats.scp" ]; then	
  cp $feats $data/feats.scp  #change to my dir 
fi

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads" 

feats="ark,s,cs:copy-feats scp:$data/feats.scp ark:- |"


if [ $stage -le 0 ] && [ ! -f "$data/output.ark" ] ; then
	log=$dir/log/nnet_forward.log
    $nnet_forward --feature-transform=$feature_transform --targets-delay=$targets_delay --use-gpu=$use_gpu \
	 --apply-log=$apply_log $nnet  "$feats"  "ark:$data/output.ark"  \
	 2>$log || exit 1;
fi

if [ $stage -le 1 ]; then
	#hddir=$data/decode_method_v2
    #mkdir -p $hddir
	log=$dir/log/decode_use_beamsearch.log
    echo `date` > $log
	
    $decode_tool --apply_log=true --beam=$beam_num --lm_order=$lm_order  "ark:$data/output.ark" "--notuse--"  "$arpa_lmfile" "ark,t:$data/bestpath_48.ark" 2>$log || exit 1;

	myscript/ark_vec-ctc-shrink.rb "$labels" > "$data/target_48_min.ark" 2>>$log || exit 1; #仅是为了去掉0
	compute-wer --text --mode=present "ark:$data/target_48_min.ark" "ark:$data/bestpath_48.ark"  - | grep "WER" >"$dir/result48_2.txt"  2>>$log || exit 1;

fi

# Run the scoring
if ! $skip_scoring ; then
	:
fi

exit 0;
