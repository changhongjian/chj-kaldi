#!/bin/bash

nnet=final.nnet               # non-default location of DNN (optional)
feature_transform=final.feature_transform  # non-default location of feature_transform (optional)

stage=0 # stage=1 skips lattice generation
[ -n "$1"  ] && stage=$1
nnet_forward="nnet-lstm-forward"  #长虹剑自己添加
decode_tool="chj-ctc-decode-beam-search-use-int-LM"
targets_delay=0
apply_log=false
beam_num=100

num_threads=1 # if >1, will use latgen-faster-parallel
parallel_opts="-pe smp $((num_threads+1))" # use 2 CPUs (1 DNN-forward, 1 decoder)
use_gpu="no" # yes|no|optionaly
# End configuration section.
myscript=/data/zyou/wpr/software/kaldi-trunk/egs/cn16k_130h/first/myscript
datadir=/data/zyou/wpr/software/kaldi-trunk/egs/cn16k_130h/first/data/data_feats_delta
arpa_lmfile=$datadir/targets/train.char.lm2
feats=$datadir/test/feats.scp
labels=$datadir/targets/test.char.ark
dir=one

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
   	if [ ! -f $data/output.maxid.ark ] ;then
	chj-matrix-max-pdf-to-id --ctc-shrink=true "ark:$data/output.ark" "ark,t:$data/output.maxid.ark" 2>>$log || exit 1;
    fi
	if [ ! -f $data/target_min.ark ] ;then
	$myscript/ark_vec-ctc-shrink.rb "$labels" > "$data/target_min.ark" 2>>$log || exit 1; #仅是为了去掉0
	fi
	#chj-decode-ctc-by-shrink-id "ark:$data/output.maxid.ark" "ark:$data/target_min.ark" "$dir/result.txt-notuse" 2>>$log || exit 1;
    #chj-decode-ctc-by-shrink is equal to computer-wer

	compute-wer --text --mode=present "ark:$data/target_min.ark" "ark:$data/output.maxid.ark"  - | grep "WER" >"$dir/result.txt"  2>>$log || exit 1;

fi

if [ $stage -le 2 ]; then
	log=$dir/log/decode_use_beamsearch.log
    echo `date` > $log

    $decode_tool --apply_log=true --beam=$beam_num  "ark:$data/output.ark" "--notuse--"  "$arpa_lmfile" "ark,t:$data/bestpath.ark" 2>$log || exit 1;

    #myscript/ark_vec-ctc-shrink.rb "$labels" > "$data/target_min.ark" 2>>$log || exit 1; #仅是为了去掉0
    compute-wer --text --mode=present "ark:$data/target_min.ark" "ark:$data/bestpath.ark"  - | grep "WER" >"$dir/result_2.txt"  2>>$log || exit 1;
	

fi

# Run the scoring
if ! $skip_scoring ; then
	a=1
#    myscript/decode_by_pdf.rb $data/output.ark  $labels $data/output.shrink $data/label.shrink
fi

exit 0;
