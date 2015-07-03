#include<fst/fstlib.h>
#include<fst/fst-decl.h>
#include<stdlib.h>
#include<sstream>
#include <cmath>
#include <limits>
using namespace fst;

const float kLogZeroFloat = -std::numeric_limits<float>::infinity();
template<typename Real>
inline Real LogAdd(Real x, Real y) {
  Real diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= -15.9424) {
    Real res;
    res = x + log1pf(expf(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

void func1(){
	// A vector FST is a general mutable FST 
	 StdVectorFst fst;  //需要命名空间
	//
	// // Adds state 0 to the initially empty FST and make it the start state. 
	 fst.AddState();   // 1st state will be state 0 (returned by AddState) 
	 fst.SetStart(0);  // arg is state ID
	//
	// // Adds two arcs exiting state 0.
	// // Arc constructor args: ilabel, olabel, weight, dest state ID. 
	 fst.AddArc(0, StdArc(1, 1, 0.5, 1));  // 1st arg is src state ID 
	 fst.AddArc(0, StdArc(2, 2, 1.5, 1)); 
	//
	// // Adds state 1 and its arc. 
	 fst.AddState();
	 fst.AddArc(1, StdArc(3, 3, 2.5, 2));
	//
	// // Adds state 2 and set its final weight. 
	 fst.AddState();
	 fst.SetFinal(2, 3.5);  // 1st arg is state ID, 2nd arg weight 
	fst.Write("binary.fst");
// 读取  StdFst *fst = StdFst::Read("binary.fst");

/*
 * 还有一种通过文件产生方式
 # arc format: src dest ilabel olabel [weight]
# final state format: state [weight]
# lines may occur in any order except initial state must be first line
# unspecified weights default to 0.0 (for the library-default Weight type)
text.fst 
0 1 a x .5
0 1 b y 1.5
1 2 c z 2.5
2 3.5  
然后提供字符到数字的映射，上面那种方式不用
isyms.txt
<eps> 0
a 1
b 2
c 3
osyms.txt
<eps> 0
x 1
y 2
z 3

然后哦如下产生
# Creates binary Fst from text file. 
# The symbolic labels will be converted into integers using the symbol table files. 
fstcompile --isymbols=isyms.txt --osymbols=osyms.txt text.fst binary.fst
通上，但是保留符号
fstcompile --isymbols=isyms.txt --osymbols=osyms.txt --keep_isymbols --keep_osymbols text.fst binary.fst
 * */
}

void func2(){
	StdFst *pfst = StdFst::Read("G.fst");
	StdFst & fst=*pfst;
	typedef StdArc::StateId StateId;
	StateId initial_state = fst.Start();
	int n=0;
	int i=2;
    for (ArcIterator<StdFst> aiter(fst, i); !aiter.Done(); aiter.Next()){
		n++;
		const StdArc &arc = aiter.Value();
		cout<<arc.nextstate<<" "<<arc.ilabel<<" "<<arc.olabel<<" "<<arc.weight<<endl;
	}
	cout<<n<<endl;
	delete pfst;
/*
 * 这里面有一些详细的使用方法
 # Gets the initial state; if == kNoState => empty FST. 
StateId initial_state = fst.Start();

# Get state i's final weight; if == Weight::Zero() => non-final. 
Weight weight = fst.Final(i);
 # Iterates over the FSTs states. 
for (StateIterator<StdFst> siter(fst); !siter.Done(); siter.Next()) 
  StateId state_id = siter.Value();
 # Iterates over state i's arcs. 
for (ArcIterator<StdFst> aiter(fst, i); !aiter.Done(); aiter.Next())
  const StdArc &arc = aiter.Value();
 # Iterates over state i's arcs that have input label l (FST must support this -
# in the simplest cases,  true when the input labels are sorted). 
Matcher<StdFst> matcher(fst, MATCH_INPUT);
matcher.SetState(i);
if (matcher.Find(l)) 
  for (; !matcher.Done(); matcher.Next())
     const StdArc &arc = matcher.Value();
 * */	
	
}

//这个就是我要做的实验

void func3(){
	//StdFst *pfst = StdFst::Read("G.fst");
	StdFst *pfst = StdFst::Read("arpa.fst");
    StdFst & fst=*pfst;
    typedef StdArc::StateId StateId;
    StateId initial_state = fst.Start();
    int n=0;
    int forword=initial_state;
    int words[2]={0};
	int len=2;
	float w=0;
	forword=1;
	while(cin>>words[0]>>words[1]){
		float rs=0;
		forword=1;
		for(int i=0;i<len;i++){
			for (ArcIterator<StdFst> aiter(fst, forword); !aiter.Done(); aiter.Next()){
				const StdArc &arc = aiter.Value();
				//cout<<arc.olabel<<endl;
				if(words[i]==arc.ilabel && words[i]==arc.olabel){
					stringstream ss;
					ss<<arc.weight;
					ss>>w;
					cout<<i<<" "<<w<<" "<<(w/-2.302585)<<endl;
					forword=arc.nextstate;
					rs=rs+w/-2.302585;
					break;
				}
			}
			cout<<"-----"<<endl;
		}
		cout<<"###-> "<<rs<<endl;
	}
/*


int words[3]={0};
	int len=2;
	float w=0;
	forword=1;
	while(cin>>words[1]>>words[2]){
		float rs=0;
		forword=1;
		words[0]=words[1];
		for(int i=1;i<=len;i++){
			for (ArcIterator<StdFst> aiter(fst, forword); !aiter.Done(); aiter.Next()){
				const StdArc &arc = aiter.Value();
				//cout<<arc.olabel<<endl;
				if(words[i-1]==arc.ilabel && words[i]==arc.olabel){
					stringstream ss;
					ss<<arc.weight;
					ss>>w;
					cout<<i<<" "<<w<<" "<<(w/-2.302585)<<endl;
					forword=arc.nextstate;
					rs=rs+w/-2.302585;
					break;
				}
			}
			cout<<"-----"<<endl;
		}
		cout<<"###-> "<<rs<<endl;
	}

*/
    delete pfst;

}
int main(){
	//func1();
	func3();
	return 0;
}
