
#include <limits>
#include<stdlib.h>
#include<fst/fstlib.h>
#include<fst/fst-decl.h>
#include <algorithm>
#include <iterator>
#include <list>
#include <map>
#include <fstream>
#include <sstream>
#include "base/kaldi-math.h"
#include <string>
using namespace kaldi;
using namespace fst;
using namespace std;
typedef kaldi::int32 int32;
string num2str(int i)
{
    stringstream ss;
    ss<<i;
    return ss.str();
}
const int MAX_ORDER=6; // for LM
float LM_log_prob(const StdFst & fst,const  int &  k ,const std::vector<int>& path,const float & lm_wght,const int &  order){
    // 根据语言模型，给定 path ，求k的概率
  int stateid=1; //看图的结论
  float w=kLogZeroFloat,a=0;
  static unsigned int history[MAX_ORDER]={0};
  memset(history,0,MAX_ORDER*sizeof(unsigned int));
  int wrd_num = path.size();
  int hismax=order-1;
  if(wrd_num<hismax) hismax=wrd_num;
  for(int i=0;i<hismax;i++){
      history[i]=path[ wrd_num-hismax+i ] ; 
  }
  history[hismax]=k;

  for(int i=0;i<=hismax;i++){
    ArcIterator<StdFst> aiter(fst, stateid);
	const StdArc * chj_arc=aiter.CHJ_Find_for_lm(history[i],false);
//	cout<<stateid<<endl;
    if(chj_arc!=NULL){
		if(i==hismax){
	        w=chj_arc->weight.Value();
		}else{
			a=0;
			stateid=chj_arc->nextstate;
		}
    }else{
//		 for (ArcIterator<StdFst> aiter(fst, stateid); !aiter.Done(); aiter.Next()){
//			const StdArc &arc = aiter.Value();
	//	    cerr<<arc.olabel<<" ---"<<endl;
//		}
		chj_arc=aiter.CHJ_Find_for_lm(0,true); // 看图的结论
		if(chj_arc!=NULL){
			a+=chj_arc->weight.Value();
			stateid=chj_arc->nextstate;
			i--; // back 
		}else{
			std::cerr<<" there must be something wrong ,in you fst lm "<<endl;
			exit(-1);
		}
	}
  } 
  return  -1 * (w + a) * lm_wght;
}

int main(int argn,char *argv[]){
	StdFst * pfst =  StdFst::Read(argv[1]);
	StdFst &  fst = * pfst;
	int order=4;
	float lm_wght=1;
	std::vector<int> path(order-1);
	int k;
	do{
		for(int i=0;i<order-1;i++){
			cin>>path[i];
		}
		cin>>k;
		cout<<LM_log_prob(fst,k,path,lm_wght,order)<<endl;
	}while('e'!=getchar());
	delete pfst;
	return 0;
}
