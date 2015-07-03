#!/bin/bash
. ./cmd.sh 
. ./path.sh ## Source the tools/utils (import the queue.pl)
set -e
stage=0 # resume training with --stage=N
[ -n "$1"  ] && stage=$1
#echo $stage
# Config:
myscript=myscript
src_index=data/featsorg
des_index=data/featschg

dir=$myexp/$runname 
for d in train dev test; do
	 utils/utt2spk_to_spk2utt.pl $src_index/$d/utt2spk > $src_index/$d/spk2utt
     $myscript/mfcc_index_13to39.sh --nj 10 --cmd "$train_cmd"  \
         $des_index/$d $src_index/$d $des_index/$d/log $des_index/$d/data
     echo "finish mfcc39  $d" 
done
exit 0
