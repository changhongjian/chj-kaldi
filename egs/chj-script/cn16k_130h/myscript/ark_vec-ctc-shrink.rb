#!/usr/bin/ruby
#输入 mfcc dir及 conf dir  default use mfcc's  targets dir 
##eg.  ./handle_label.rb ../data/mfcc39/ ../data/conf/ 
=begin
	
=end

fin=ARGV[0]
#fin="/tmp/chj.txt"
	
	fp=File.new fin,"r"
    while line=fp.gets
		next if line.strip==""
		dt=line.split /\s+/
		print dt[0]+" "
		#print dt[1]+" "
		1.upto(dt.size-1).each do |i|	
			print dt[i]+" " unless dt[i]=="0"
		end
		puts 
		#puts dt[-1]
	end
	fp.close

