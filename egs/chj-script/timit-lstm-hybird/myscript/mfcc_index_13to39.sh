#!/bin/bash
#删减 make_fmllr_feats.sh
#cmvn 使用了 compute_cmvn_stats.sh 里面的
nj=4
cmd=run.pl

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

data=$1 #data-fmllr-tri3/test 主目录
srcdata=$2 #data/test 最原始的mfcc特征目录
logdir=$3 #data-fmllr-tri3/test/log
feadir=$4 #data-fmllr-tri3/test/data feats的 ark文件保存的地方

sdata=$srcdata/split$nj; #分成多个进程处理
#本来有几个_opts 我不要了

mkdir -p $data $logdir $feadir
[[ -d $sdata && $srcdata/feats.scp -ot $sdata ]] || split_data.sh $srcdata $nj || exit 1;

# Check files exist,
for f in $sdata/1/feats.scp $sdata/1/cmvn.scp; do
  [ ! -f $f ] && echo "$0: Missing file $f" && exit 1;
done

#feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";
feats="ark,s,cs:apply-cmvn --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";
#--delta-order=$delta_order  其实应该有这个参数的 这个很深，经过很多个类，发现默认是2

# Prepare the output dir,
cp $srcdata/* $data 2>/dev/null; rm $data/{feats,cmvn}.scp;
# Make $bnfeadir an absolute pathname,
feadir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $feadir ${PWD}`
#这样就可以索引到本目录下了
# Store the output-features,
name=`basename $data`
$cmd JOB=1:$nj $logdir/make_mfccc13to39.JOB.log \
  copy-feats "$feats" \
  ark,scp:$feadir/mfccc13to39_$name.JOB.ark,$feadir/mfccc13to39_$name.JOB.scp || exit 1;
#所以最后我保存的特征就是已经做过delta还有cmvn的   JOB 的替换都是在run.pl这个程序里
# Merge the scp,
for n in $(seq 1 $nj); do
  cat $feadir/mfccc13to39_$name.$n.scp 
done > $data/feats.scp
#如果要求出cmvn则如下 其实我觉的没用，应为cvnm_g 直接就用了下面的方式
cmvndir=$feadir
# make $cmvndir an absolute pathname.
cmvndir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $cmvndir ${PWD}`
! compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp ark,scp:$cmvndir/cmvn_$name.ark,$cmvndir/cmvn_$name.scp \
    2> $logdir/cmvn_$name.log && echo "Error computing CMVN stats" && exit 1;
cp $cmvndir/cmvn_$name.scp $data/cmvn.scp
echo "$0: Done!,  $srcdata --> $data,"

exit 0;
