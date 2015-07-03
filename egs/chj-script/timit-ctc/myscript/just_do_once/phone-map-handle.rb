#!/usr/bin/ruby
#输入 mfcc dir及 conf dir  default use mfcc's  targets dir 
##eg.  ./handle_label.rb ../data/mfcc39/ ../data/conf/ 
=begin
	
=end

conf_dir="/data/zyou/wpr/software/kaldi-trunk/egs/timit/ctc/data/conf/"

	fp=File.new conf_dir+"/phones.60-48-39.map","r"
    ps,pd,ps_d={},{},{}
    s_cnt,d_cnt=1,1
	s,d=1,2
    while line=fp.gets
        ptmp=line.split /\s+/
        next if ptmp[s].nil? || ptmp[d].nil?
        if  ps[ptmp[s]].nil?
            ps[ptmp[s]]=s_cnt; s_cnt+=1;
        end
		if  pd[ptmp[d]].nil?
			pd[ptmp[d]]=d_cnt; d_cnt+=1;
		end
        ps_d[ ps[ ptmp[s] ] ]=pd[ ptmp[d] ] 
    end
    fp.close
	fp=File.new conf_dir+"/phone-int-48.map","w"
	ps=ps.sort do |a,b|
		a[1]<=>b[1]
	end
	for p in ps
		fp.puts p[1].to_s+" "+p[0]
	end
	fp.close
	
	fp=File.new conf_dir+"/phone-int-39.map","w"
    pd=pd.sort do |a,b|
        a[1]<=>b[1]
    end
    for p in pd
        fp.puts p[1].to_s+" "+p[0]
    end
    fp.close

    fp=File.new conf_dir+"/phone-int-48-39.map","w"
	for p in ps
		fp.puts p[1].to_s+" "+ps_d[p[1]].to_s unless ps_d[p[1]].nil? 
	end
    fp.close

exit # 下面的不用了，上面就随便改成自己所需要的就行 

in_f=ARGV[0]
#out_f=ARGV[1]
def hd_phone(fpin,fpout,p60_48)
	while line=fpin.gets
		next if line !~ /^(.*?)\s(.*?)\s?$/
		out=$1+" [ 0 "
		phones=$2.split(" ")
		for p in phones
			next if p.nil? || p60_48[p].nil?
			out+=p60_48[p].to_s+" "+"0 "
		end
		out+="]"
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



