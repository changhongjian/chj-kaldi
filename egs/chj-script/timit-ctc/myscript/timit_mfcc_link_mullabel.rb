#!/usr/bin/ruby
#require 'optparse'
#输入Timit的目录，mfcc所在的目录  default use mfcc's dev test train targets dir
##eg. ./timit_mfcc_link_mullabel.rb /data/zyou/wpr/data/TIMIT/  ../data/mfcc39/ | less
if ARGV.size != 2 then
	puts "you must provide timit dir,mfcc dir"
	exit(1)
end
timit_dir,mfcc_dir=ARGV
need_dirs=%w[train dev test train_cv10 train_tr90 targets]
orgsuffix=%w[txt wrd phn]
suffix=%w[sentence word phone]
useindex=[0]  #[0,1,2]
need_dirs=%w[train targets]
def hd_sentence(fpin,fpout,key)
	 line=fpin.gets
         next if line !~  /^\d+\s\d+\s(.*?)\s?$/
         fpout.puts key+" "+$1
end
def hd_word(fpin,fpout,key)
	words=key
	while line=fpin.gets
		next if line !~/^\d+\s\d+\s(.*?)\s?$/
		words+=" "+$1
	end
	fpout.puts words
end
def hd_phone(fpin,fpout,key)
	phones=key
	while line=fpin.gets
		next if line !~/^\d+\s\d+\s(.*?)\s?$/
		phones+=" "+$1
	end
	fpout.puts phones
end
hd_methods=[]#[self.method :hd_sentence,self.method :hd_word,self.method :hd_phone]
hd_methods[0]=self.method :hd_sentence
hd_methods[1]=self.method :hd_word
hd_methods[2]=self.method :hd_phone
need_dirs.each do |dir|
	if !File.directory?(mfcc_dir+"/"+dir) then
		puts "can't find dir--> #{dir} from #{mfcc_dir}"
		exit(2)
	end
end
n=need_dirs.size - 1
need_dirs[0...n].each do |dir|
	fpin=File.new(mfcc_dir+"/"+dir+"/feats.scp","r");
	fpout=[]
	for i in useindex
                fpout[i]=File.new("#{mfcc_dir}/#{need_dirs[-1]}/#{dir}.#{suffix[i]}","w");
        end

	#type=dir=="test"?"test/":"train/"    it is not like want I thought before
        while line=fpin.gets
		next if line !~/(.*?)\s/
		line=$1
		keys=$1.split(/_/)	
		fname=timit_dir+"/"+'{test,train}/dr*/'+keys[0]+"/"+keys[1]+".wav" #***
		fname=Dir.glob(fname)
		if fname.size!=1 then
			puts "in "+mfcc_dir+"/"+dir+"/feats.scp,"+$1+" not found"
			exit 3
		end
		for i in useindex
			fnm=fname[0].sub(/wav$/,orgsuffix[i])
			fp=File.new(fnm,"r");
			hd_methods[i].call(fp,fpout[i],line)
			fp.close
		end
	end
	fpin.close
        for i in useindex
                fpout[i].close
        end

end



