#!/usr/bin/ruby
dev =false
dev =true , ARGV.shift if ARGV[0]=="-d"
ARGV.each do |x|
	Dir.glob("exp/#{x}") do |fname|
		puts fname
		puts %x[
			grep Sum #{fname}/decode_test/score_*/*.sys 2>/dev/null | utils/best_wer.sh
		]   
		puts %x[
            grep Sum #{fname}/decode_dev/score_*/*.sys 2>/dev/null | utils/best_wer.sh
        ] if dev
		puts
	end
end

