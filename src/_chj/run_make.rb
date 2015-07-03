#!/usr/bin/ruby

editfile="beam-search-with-fst-adv.cc"
makefile="makefile-cs"
runcmd="beam-search-with-fst-adv --apply_log=true --lm_order=4 --beam=10 ark:/data/zyou/wpr/software/kaldi-trunk/egs/cn16k_130h/first/exp/lx-decode-beam/four/hd_dir/output.ark /data/zyou/wpr/software/kaldi-trunk/egs/cn16k_130h/first/data/data_feats_delta/targets/arpa_cn_4.fst ark,t:-"
if ARGV.size==0 || ARGV[0]=="mr"
	`make -f #{makefile}`
	exec runcmd
elsif ARGV[0]=="m"
	exec("make -f #{makefile}")
elsif ARGV[0]=="e"
	exec("vim #{editfile}")
elsif ARGV[0]=="r"
	exec runcmd
elsif ARGV[0]=="s"
	exec "vim #{__FILE__}"
end

puts "--- use ./#{__FILE__} s to see  ---"

