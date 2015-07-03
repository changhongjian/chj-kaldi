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
lm_order=3
arpa_lmfile=$datadir/targets/train.char.lm$lm_order
fst_lmfile=$datadir/targets/arpa_cn_${lm_order}.fst
feats=$datadir/test/feats.scp
labels=$datadir/targets/test.char.ark
dir=three_3

data=$dir/hd_dir
mkdir -p $dir/log
mkdir -p $data
ln -s three/hd_dir/target_min.ark $data/target_min.ark
#data=$dir/splithandle

if [ ! -f "$data/feats.scp" ]; then
  cp $feats $data/feats.scp  #change to my dir 
fi

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads" 

feats="ark,s,cs:copy-feats scp:$data/feats.scp ark:- |"


if [ $stage -le 0 ] && [ ! -f "$data/output.ark" ] ; then
    ln -s  "../../one/hd_dir/output-top1000.ark" $data/output.ark
fi

reslog=$dir/comp-result.txt
echo -n ""  >$reslog
if [ $stage -le 1 ]; then
	log=$dir/log/decode_use_max_id.log
#    echo `date` > $log
	##the follow one can do all the thing by itself
    #$decode_tool "ark:$data/output.ark" "ark:$labels" "$dir/result.txt" 2>>$log || exit 1;	
   	if [ ! -f $data/output.maxid.ark ] ;then
:
#	   chj-matrix-max-pdf-to-id --ctc-shrink=true "ark:$data/output.ark" "ark,t:$data/output.maxid.ark" 2>>$log || exit 1;
    fi
	if [ ! -f $data/target_min.ark ] ;then
:
#	   $myscript/ark_vec-ctc-shrink.rb "$labels" > "$data/target_min.ark" 2>>$log || exit 1; #仅是为了去掉0
	fi
#    echo "decode_use_max_id" >$reslog
#	compute-wer --text --mode=present "ark:$data/target_min.ark" "ark:$data/output.maxid.ark"  - | grep "WER" >>"$reslog"  2>>$log || exit 1;

fi


if [ $stage -le 2 ]; then
  beams=(30 50) 
  names=(beam-search-with-fst-adv)
  for beam_num in ${beams[@]}; do
    for name in ${names[@]};do
      log="$dir/log/$name.$beam_num.log"
   	  echo "$name $beam_num" >> $reslog
      outark="ark,t:$data/d_${name}_${beam_num}.ark"
      
      if [[ -f $outark  ]]; then
:
#		continue
      fi
      lmfile=
      if [[ ${name/fst//} == $name ]]
      then
  		lmfile=$arpa_lmfile
  	  else
  		lmfile=$fst_lmfile
      fi
      $name --apply_log=true --lm_order=$lm_order  --beam=$beam_num  "ark:$data/output.ark" $lmfile $outark >$log || exit 1;
      compute-wer --text --mode=present "ark:$data/target_min.ark" $outark  - | grep "WER" >>"$reslog"  2>>$log || exit 1;
  	done
  done
fi

# Run the scoring
if ! $skip_scoring ; then
	a=1
#    myscript/decode_by_pdf.rb $data/output.ark  $labels $data/output.shrink $data/label.shrink
fi

exit 0;
