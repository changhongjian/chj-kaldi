#!/usr/bin/ruby
#输入 mfcc dir及 conf dir  default use mfcc's  targets dir 
##eg.  ./handle_label.rb ../data/mfcc39/ ../data/conf/ 
=begin
	
=end

conf_dir="/data/zyou/wpr/software/kaldi-trunk/egs/timit/ctc/data/conf/"
fin=ARGV[0]
fin="/tmp/chj.txt"

	fp=File.new conf_dir+"/phone-int-48-39.map","r"
    ps_d={}
	
    while line=fp.gets
		next if line.strip==""
        sid,did=line.split /\s+/
        ps_d[ sid ] = did 
    end
    fp.close
	
	fp=File.new fin,"r"
    while line=fp.gets
		next if line.strip==""
		dt=line.split /\s+/
		print dt[0]+" "
		print dt[1]+" "
		2.upto(dt.size-1).each do |i|
		
			print ps_d[dt[i]]+" " unless ps_d[dt[i]].nil? || ps_d[dt[i]]=="0"
		end
		puts dt[-1]
	end
	fp.close

exit 
