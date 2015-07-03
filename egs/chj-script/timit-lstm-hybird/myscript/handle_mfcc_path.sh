#!/bin/bash
data="data"
mfccdir=~/my_dir/data/mfcc #/home/wpr
[ -n "$1" ] && data=$1
[ -n "$2" ] && mfccdir=$2
[ ! -d $data ] && echo " $data 目录不存在" && exit 1;
for d in train dev test; do 
  for f in $data/$d/{feats,cmvn}.scp; do
	#echo $f
	perl -e ' 
		my($file,$dir)=@ARGV; #print $file;
		open (FILE,"<$file") or die "error open $file";
		open (FILETMP,">${file}_tmp") or die "error open $file";
		while(<FILE>){
			#print $_;
			my($id,$path)=split /\s+/,$_;
			$path=$dir."/".(split /\//,$path)[-1];
			#print  $id."\n".$path."\n";break;
			print FILETMP	$id." ".$path."\n";
		}
		close FILE;
		close FILETMP;	
		system("mv ${file}_tmp $file");
		system("unlink ${file}_tmp");
	' $f $mfccdir
#
  done
done
