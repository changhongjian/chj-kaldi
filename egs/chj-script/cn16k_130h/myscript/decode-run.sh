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
	log=$dir/log/decode_use_max_id.log
    echo `date` > $log
	##the follow one can do all the thing by itself
    #$decode_tool "ark:$data/output.ark" "ark:$labels" "$dir/result.txt" 2>>$log || exit 1;	
	
	chj-matrix-max-pdf-to-id --ctc-shrink=true "ark:$data/output.ark" "ark,t:$data/output.maxid.ark" 2>>$log || exit 1;

	myscript/ark_vec-ctc-shrink.rb "$labels" > "$data/target_min.ark" 2>>$log || exit 1; #仅是为了去掉0
	chj-decode-ctc-by-shrink-id "ark:$data/output.maxid.ark" "ark:$data/target_min.ark" "$dir/result.txt-notuse" 2>>$log || exit 1;

#chj-decode-ctc-by-shrink is equal to computer-wer
	compute-wer --text --mode=present "ark:$data/target_min.ark" "ark:$data/output.maxid.ark"  - | grep "WER" >"$dir/result48_1.txt"  2>>$log || exit 1;

fi

# Run the scoring
if ! $skip_scoring ; then
	a=1
#    myscript/decode_by_pdf.rb $data/output.ark  $labels $data/output.shrink $data/label.shrink
fi

exit 0;
