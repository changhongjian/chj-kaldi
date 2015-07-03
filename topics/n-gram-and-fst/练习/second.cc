#include<fst/fstlib.h>
#include<fst/fst-decl.h>
#include "lm/model.hh"
#include <stdlib.h> // itoa
#include <time.h>
#include <sys/time.h>
#include<sstream>
using namespace fst;
using namespace lm::ngram;
int ngram=2;
int totalnum=48; //6724
int csnum=100000000;
int getSrandInt(){
	return rand()%totalnum+1;
}
string num2str(int32 i)
{
    stringstream ss;
    ss<<i;
    return ss.str();
}
void func2(const Model& lm_mdl){
  const Vocabulary &vocab = lm_mdl.GetVocabulary();
  lm::FullScoreReturn ret;
  State out_state(lm_mdl.NullContextState());
  static unsigned int history[6]={0};
  int last=ngram-1;
  int n=csnum;
  while(n--){
	  for(int i=0;i<ngram;i++){
		  history[i]=vocab.Index( num2str( getSrandInt() ) ) ; //我就是这里错了
	  }
	  //history[0]=vocab.Index( num2str( 1 ) );
	  //history[1]=vocab.Index( num2str( 1 ) );
	  ret=lm_mdl.FullScoreForgotState(history, history+last, vocab.Index( num2str(last) ), out_state);
      //cout<<ret.prob<<endl;
  } 

}

void func3(const StdFst & fst){
    typedef StdArc::StateId StateId;
    StateId initial_state = fst.Start();
    int words[6]={0};
	int n=csnum;
	int rd=0;
	stringstream ss;
	while(n--){
		for(int i=0;i<ngram;i++){
			words[i]=getSrandInt();
		}
		//words[0]=1;
		//words[1]=1;
		int stateid=1;
		for(int i=0;i<ngram;i++){
		   float w=-111111;
		   ArcIterator<StdFst> aiter(fst, stateid);
		   const StdArc * chj_arc=aiter.CHJ_Find(words[i]);
		   if(chj_arc!=NULL){
				w=chj_arc->weight.Value();
				//cout<<chj_arc->weight<<endl;
		   }
/*
		   for (ArcIterator<StdFst> aiter(fst, stateid); !aiter.Done(); aiter.Next()){
			    const StdArc &arc = aiter.Value();
				if(words[i]==arc.ilabel && words[i]==arc.olabel){
					//cout<<arc.weight<<endl;
					stringstream  ss;
					ss<<arc.weight;
					ss>>w;
					stateid=arc.nextstate;
					break;
				}
		  }
*/
		  if(w==-111111){
				//cout<<"++++++"<<n<<endl;
				rd+=ngram-i;
				break;
		  }
	   }
	}
	cout<<"$$$"<<rd<<endl;
}
int main(){
	srand((unsigned)time(NULL));
	char f_arpa[30]={"data/train.phone.lm2"},f_fst[30]={"arpa.fst"}; 
	//cin>>f_arpa;
	//cin>>f_fst;
	//cin>>ngram;
	ngram=2;
	Model lm_mdl(f_arpa);
    StdFst *  pfst = StdFst::Read(f_fst);

	long int lTime;//默认微秒数  
    struct timeval start,end;

	gettimeofday(&start,NULL);
	func2(lm_mdl);
	gettimeofday(&end,NULL); 
	lTime = 1000000*( end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec; 
	lTime /= 1000000;
	cout<<"###---> time used "<<lTime<<endl;
	gettimeofday(&start,NULL);
	func3(*pfst);
	gettimeofday(&end,NULL);
    lTime = 1000000*( end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
    lTime /= 1000000;
    cout<<"###---> time used "<<lTime<<endl;
	return 0;
}

