#!/usr/bin/perl -w
#解码的时候把nstream改成1
#这个脚本是一开始解码只能为1而些的，现在不需要了
#
my $usage = "Usage: $0  -out final.nnet.txt \n
eg. nnet-copy --binary=false final.nnet - | ./hd_lstm_nnet.pl -out final.nnet.txt
just have a change for decode about lstm stream. 需要使用管道,作为输入
\n";

use strict;
use Getopt::Long;
die "$usage" unless(@ARGV >=1 );
my ($out_nnet);
GetOptions ("out=s" => \$out_nnet); # Output nnet  多个则用都好隔开
die $usage unless(defined($out_nnet));
open (OUT,">$out_nnet") or die "Cannot open mappings file '$out_nnet': $!";
while(<>){
	s/<NumStream>\s*(\d*)\s*\[\s*$/<NumStream> 1  \[/;
	print OUT $_;
}
close OUT;

