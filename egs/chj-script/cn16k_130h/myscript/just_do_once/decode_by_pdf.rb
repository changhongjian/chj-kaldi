#!/usr/bin/ruby
=begin
=end


load "myscript/parse_option.rb"

if $argv.size != 4
	puts  "argv not fits"
	exit 1
end

pdf_f=$argv[0]
label_f=$argv[1]
out_r_f=$argv[2]
out_l_f=$argv[3]

out_r_fp=File.open(out_r_f,"w");

File.open(pdf_f,"r") do |fp|
	while line=fp.gets
		sz=line.split /\s+/
		n=sz.size
		n-=1
		[2...n].each do |x|
			
		end
	end
end


close out_r_fp

out_l_fp=File.open(out_l_f,"w");
close out_l_fp







