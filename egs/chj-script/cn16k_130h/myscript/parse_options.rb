#!/usr/bin/ruby
=begin
作者：虹猫大侠
时间：2015-4-19
=end
#######
$argv=[]
$config=""
0.step(ARGV.size-1,1){|x|
	#p ARGV[x]
	if ARGV[x]== "--config" then
		load ARGV[x+1] if File.exist? ARGV[x+1]
	end
}
while ARGV.size>0
	case ARGV[0]
	when /\-\-(.*)=(.+)/
		ARGV.shift
		val=$2
		eval %Q(
				if ! defined? $#{$1} then
					puts \"Error: invalid option #{$1}\"
					exit(2)
				else
					val=val.to_i if $#{$1}.kind_of? Integer 
					$#{$1}=val
				end
			)
	when /\-\-(.*)/
		ARGV.shift #默认后面为 1
		val=ARGV.shift
		if val.nil? then
			puts "Error: option #{$1} has no value"
			exit(1)
		else
			# %Q 代表双引号
			eval %Q(
					if ! defined? $#{$1} then
						puts \"Error: invalid option #{$1}\"
						exit(2)
					else
						val=val.to_i if $#{$1}.kind_of? Integer 
						$#{$1}=val
					end
				)
		end
	when /(?:\-\-h)|(?:\-h)/
		ARGV.shift
		if  defined? $help_message then
			puts $help_message
		else
			puts "Warning: no help message"  
		end
	else
		$argv.push ARGV.shift
	end
end

#######



