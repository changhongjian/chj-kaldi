#!/bin/bash

nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
class_frame_counts= # non-default location of PDF counts (optional)

stage=0 # stage=1 skips lattice generation

nnet_forward="nnet-lstm-forward"  #长虹剑自己添加
decode_tool=""
targets_delay=0
apply_log=false

num_threads=1 # if >1, will use latgen-faster-parallel
parallel_opts="-pe smp $((num_threads+1))" # use 2 CPUs (1 DNN-forward, 1 decoder)
use_gpu="no" # yes|no|optionaly
acwt=
nj=4
cmd=run.pl
# End configuration section.
beam_num=30

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "parameter error"
   exit 1;
fi

feats=$1
labels=$2
arpa_lmfile=$3
dir=$4

data=$dir/hd_dir
mkdir -p $dir/log
mkdir -p $data
#data=$dir/splithandle
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads" 

feats="ark,s,cs:copy-feats scp:$feats ark:- |"


if [ $stage -le 0 ] && [ ! -f "$data/output.ark" ] ; then
	log=$dir/log/nnet_forward.log
    $nnet_forward --feature-transform=$feature_transform --targets-delay=$targets_delay --use-gpu=$use_gpu \
	 --apply-log=$apply_log $nnet  "$feats"  "ark:$data/output.ark"  \
	 2>$log || exit 1;
fi

if [ $stage -le 1 ]; then
    reslog="$dir/result48.txt" #内含各种组合
	log=$dir/log/decode_use_max_id.log
    echo `date` > $log
	echo "decode_use_max_id" > $reslog
	chj-matrix-max-pdf-to-id --ctc-shrink=true "ark:$data/output.ark" "ark,t:$data/output.maxid.ark" 2>>$log || exit 1;
	compute-wer --text --mode=present "ark:$labels" "ark:$data/output.maxid.ark"  - | grep "WER" >>"$reslog"  2>>$log || exit 1;
   
    log=$dir/log/decode_use_lm.log
    echo `date` > $log
    echo "decode_use_lm" >> $reslog
    for x in $(seq 2 5) ;do
		outark="ark,t:$data/bestpath_48.ark" #会被覆盖
		lx-beam-search --apply_log=true --beam=$beam_num --lm_order=$x "ark:$data/output.ark" "$arpa_lmfile$x"	"$outark" 2>>$log || exit 1;
		compute-wer --text --mode=present "ark:$labels" "$outark"  - | grep "WER" >>"$reslog"  2>>$log || exit 1;        
    done   
 
fi

# Run the scoring
if ! $skip_scoring ; then
	a=1
#    myscript/decode_by_pdf.rb $data/output.ark  $labels $data/output.shrink $data/label.shrink
fi

exit 0;
