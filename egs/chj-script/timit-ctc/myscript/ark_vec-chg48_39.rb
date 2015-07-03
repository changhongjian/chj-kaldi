#!/usr/bin/ruby
=begin
最后的输出利用重定向	
=end

conf_dir="/data/zyou/wpr/software/kaldi-trunk/egs/timit/ctc/data/conf/"
fin=ARGV[0]
#fin="/tmp/chj.txt"

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
		#print dt[1]+" "
		1.upto(dt.size-1).each do |i|	
			print ps_d[dt[i]]+" " unless ps_d[dt[i]].nil? || ps_d[dt[i]]=="0"
			#print "0 " if ps_d[dt[i]]=="0"   #平时这个是需要注释的
		end
		puts 
		#puts dt[-1]
	end
	fp.close

