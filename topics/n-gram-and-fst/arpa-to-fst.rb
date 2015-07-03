#!/usr/bin/ruby
=begin
作者：长虹剑 time:from 2015-5-9 to 2015-5-11
根据kaldi脚本改写的 arpa生成fst的程序

程序创建的文件为临时文件，会自动删除。

使用方法：
需要自己改写目录  （ruby的shell在使用用户目录配置的环境变量有问题，目前解决不了，所以才这么麻烦）

create_words : 因为后来生成文法文件就直接利用了数字，如果是这样的话，words.txt就不用准备了，程序会自动生成，同时加上必要的东西。
natural_base : arpa 文件是以10为底的对数，如果转成以为底的就需要用这个，kaldi里面就是这个，同时有时不用这个还会报错（这个也需要解决）。
=end

require "tempfile"  

farpa=ARGV[0] 
words_num=6724 #通过这个自动,生成文件words.txt
create_words=true
fout="arpa_cn_2.fst"
isymbols="words.txt" #打算用临时文件
osymbols="words.txt"
kaldidir="/data/zyou/wpr/software/kaldi-trunk/"
kaldi_openfst_dir="tools/openfst-1.3.4/bin/"
arpa2fst=kaldidir+"/src/bin/arpa2fst"
natural_base=true
fstprint=kaldidir+kaldi_openfst_dir+"fstprint"
fstcompile=kaldidir+kaldi_openfst_dir+"fstcompile"
fstrmepsilon=kaldidir+kaldi_openfst_dir+"fstrmepsilon"
fstarcsort=kaldidir+kaldi_openfst_dir+"fstarcsort"

#system("cat #{farpa} | egrep -v '<s> <s>|</s> <s>|</s> </s>'  | arpa2fst - ")
#cmd=`cat #{farpa} | egrep -v '<s> <s>|</s> <s>|</s> </s>'  | #{arpa2fst} - | #{fstprint}`
cmd=%x{
cat #{farpa} |
egrep -v '<s> <s>|</s> <s>|</s> </s>' |
#{arpa2fst} --natural-base=#{natural_base} - |
#{fstprint}
}
#io=IO.popen(cmd,"r+")
sz=cmd.split /\n/
cmd=nil
temp = Tempfile.new("stuff")
#temp = File.new("/tmp/stuff","w")
sz.each do |x|
	x.gsub!(/^(\d+\s+\d+\s+)\<eps\>(\s+)/,"\\1#0\\2");
	a= x.split("\t"); # must be this
	#p x
	#p a
	if a.size>=4
		if a[2]=="<s>" || a[2]=="</s>"
			a[2]="<eps>"
		end
		if a[3]=="<s>" || a[3]=="</s>"
			a[3]="<eps>"
		end
	end
	x=a.join("\t")
	temp.puts x
end
sz=nil
name=temp.path
temp.close 
if create_words==true
	temp=Tempfile.new("words.txt")
	temp.puts "<eps> 0"
	1.upto(words_num) do |i|
		temp.puts "#{i} #{i}"
	end
	temp.puts "#0 #{words_num+1}"
	temp.puts "<s> #{words_num+2}"
	temp.puts "</s> #{words_num+3}"
	osymbols=isymbols=temp.path
	temp.close
end

#不用反点 是为了好看
%x{
#{fstcompile} --isymbols=#{isymbols} \
     --osymbols=#{osymbols}  --keep_isymbols=false --keep_osymbols=false #{name} | 
#{fstrmepsilon} |
#{fstarcsort} --sort_type=ilabel > #{fout}
}

=begin
`fstcompile --isymbols=#{isymbols} \
     --osymbols=#{osymbols}  --keep_isymbols=false --keep_osymbols=false #{name} | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > #{fout} `
=end



