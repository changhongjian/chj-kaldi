#!/usr/bin/ruby

config=%w[
/data/zyou/wpr/software/kaldi-trunk/egs/cn16k_130h/first/data/datafeats/targets/
train.char.ark
train.char.forlm
int.dict
train.char.lm
4
]

userdir=config[0]

forgphone=userdir+config[1]

fsentence=userdir+config[2]

File.open(fsentence , "w")  { |fpout|
	File.open(forgphone,"r") { |f|
		while line=f.gets
			next if line.strip==""
			sp=line.split /\s+/
			sp.shift  #去掉第一个
			rs=""
			sp.each do |x|
				next if x=="0"  #这个很通用
				rs+=x+" "
			end
			fpout.puts rs
		end
	}
}

order=config[-1]
flmvocab=userdir+config[3]
flm=userdir+config[4]+order

#`ngram-count -text #{fsentence} -order #{order} -write #{flmcount} `
`ngram-count -text #{fsentence} -vocab #{flmvocab}  -order #{order} -lm #{flm} -interpolate -kndiscount  `
# -interpolate -kndiscount



