#!/usr/bin/ruby
#require 'optparse'
#输入 mfcc dir及 conf dir  default use mfcc's  targets dir 
##eg.  ./handle_label.rb ../data/mfcc39/ ../data/conf/ 
=begin
	conf 里面有那个map文件 这里用到就是 48
    targets 需要用自己之前的那个脚本生成 
=end
if ARGV.size != 2 then
	puts "you must provide mfcc dir,conf dir"
	exit(1)
end
mfcc_dir,conf_dir=ARGV
need_dirs=%w[train dev test train_cv10 train_tr90 targets3]
suffix=%w[word phone]
useindex=[1]
def hd_word(fpin,fpout)
	while line=fpin.gets
		next if line !~/^(.*?)\s(.*?)\s?$/
		#out=$1+" "+"[ "+$2+" ]"
		out=$1+" "+$2
		fpout.puts out
	end
end
def hd_phone(fpin,fpout,p60_48)
	while line=fpin.gets
		next if line !~ /^(.*?)\s(.*?)\s?$/
		#out=$1+" [ 0 "
		out=$1+" 0 "
		phones=$2.split(" ")
		for p in phones
			next if p.nil? || p60_48[p].nil?
			out+=p60_48[p].to_s+" "+"0 "
		end
		#out+="]"
		fpout.puts out
	end
end
hd_methods=[]
hd_methods[0]=self.method :hd_word
hd_methods[1]=self.method :hd_phone
need_dirs.each do |dir|
	if !File.directory?(mfcc_dir+"/"+dir) then
		puts "can't find dir--> #{dir} from #{mfcc_dir}"
		exit(2)
	end
end
targets_dir=mfcc_dir+need_dirs[-1]
if File.exist?(targets_dir) then
	fp=File.new conf_dir+"/phones.60-48-39.map","r"
	p48,p60_48={},{}
	cnt=1
	while line=fp.gets
		ps=line.split /\s+/
		next if ps[0].nil?
		if !ps[1].nil? && p48[ps[1]].nil?
			p48[ps[1]]=cnt; cnt+=1;
		end
		p60_48[ps[0]]=p48[ps[1]] unless ps[1].nil?
	end
	fp.close
	fp=File.new targets_dir+"/phone-int.map","w"
	p48=p48.invert
	p48=p48.sort {|a,b|
		a[0]<=>b[0]
	}
	for p in p48
		fp.puts p[0].to_s+" "+p[1]
	end
	fp.close
end
n=need_dirs.size - 1
need_dirs[0...n].each do |dir|
	for i in useindex
		fnm=mfcc_dir+"/"+need_dirs[-1]+"/"+dir+"."+suffix[i]
		fpin=File.new(fnm,"r")
                fpout=File.new(fnm+".ark","w");
		hd_methods[i].call(fpin,fpout,p60_48) if i==1
		fpin.close
		fpout.close
        end

end



