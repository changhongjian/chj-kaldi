#!/usr/bin/ruby 

ARGV.each do |x|
	fname=x+"/decode_test/result_1.txt"
	File.open(fname,"r") do |f|
		puts f.gets
	end
end
