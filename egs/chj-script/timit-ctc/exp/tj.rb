#!/usr/bin/ruby 

fout=File.new("rs.txt","w")
Dir.glob("manyexps-*").sort do |a,b|
#manyexps-d1-s4-lr0.00005
   a=~/manyexps-d(\d+)-s(\d+)-lr(.*)/
   a=[$1.to_i,$2.to_i,$3.to_f]
   b=~/manyexps-d(\d+)-s(\d+)-lr(.*)/
   b=[$1.to_i,$2.to_i,$3.to_f]
   n=0;
   while(a[n]==b[n] && n<2)
     n+=1
   end
   a[n]<=>b[n]

end.each do |f|
    next unless File.exist? f+"/decode_test/result48.txt"
    File.open(f+"/decode_test/result48.txt") do |fp|
        fout.puts f
        while line=fp.gets
             fout.puts line
		end
		fout.puts 
	end
	#puts f
end

fout.close
