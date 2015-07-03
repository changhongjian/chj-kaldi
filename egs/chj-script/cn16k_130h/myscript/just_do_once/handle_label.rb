#!/usr/bin/ruby
=begin
就是在一个ark文件中插入 0 
=end
fnin=ARGV[0]
fnout=ARGV[1]

File.open(fnout,"w") do |fout|
	File.open(fnin,"r") do |fin|
		while line=fin.gets
			sp=line.split /\s+/
		    out=sp[0]+" 0"
			(1...sp.size).each do |x|
				out+=" "+sp[x].to_s+" 0"
			end
			fout.puts out
		end
	end
end

