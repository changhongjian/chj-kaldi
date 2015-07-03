/*
 *长虹剑 改 杰哥 的 
 * */
#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "base/kaldi-math.h"

#include "lm/model.hh"

#include <algorithm>
#include <iterator>
#include <list>
#include <map>
#include <fstream>
#include <sstream>


using namespace lm::ngram;
using namespace std;

namespace kaldi{
namespace nnet1{

string num2str(int32 i)
{
    stringstream ss;
    ss<<i;
    return ss.str();
}
const int MAX_ORDER=6; // for LM
class Prefix{
public:
    Prefix(){
        Rb=Rnb=kLogZeroFloat;
        score=kLogZeroFloat;
    }
    Prefix(const std::vector<int32>& path_,const float & Rb_=kLogZeroFloat,const  float &  Rnb_=kLogZeroFloat){
        path=path_;
        Rb=Rb_;
        Rnb=Rnb_;
        score=kLogZeroFloat;
		//makeScore();
    }
    std::vector<int> path;
    float Rb;
    float Rnb;
    float score;
    void makeScore(){
		score=LogAdd(Rb,Rnb);
		cmpscore_=score/(1+path.size()); //这样处理不知是否合适
    } //先执行这个，才能执行ScoreForCompare
    float ScoreForCompare() const {return cmpscore_;}
    
    bool operator == (Prefix const & cls) const{
        return path==cls.path;
    }
    bool operator < (Prefix const & cls) const{
        return ScoreForCompare() < cls.ScoreForCompare();
    }
private:
	float cmpscore_;
};
typedef std::vector<Prefix> VECTOR;
float LM_log_prob(const Model& lm_mdl,const  int &  k ,const std::vector<int>& path,float & lm_wght,int &  order){
    // 根据语言模型，给定 path ，求k的概率 
  //return 0;
  const Vocabulary &vocab = lm_mdl.GetVocabulary();
  lm::FullScoreReturn ret;
  State out_state(lm_mdl.NullContextState());
  static unsigned int history[MAX_ORDER]={0};
  memset(history,0,MAX_ORDER*sizeof(unsigned int));
  history[0] = vocab.Index("<s>"); // 这个这里面是没有用的
  int wrd_num = path.size();
  int uselen=(order-1)<wrd_num?(order-1):wrd_num;
  for(int i=0;i<wrd_num && i<order-1;i++){
      history[i]=vocab.Index( num2str( path[ wrd_num-i-1 ] ) ) ; //我就是这里错了
  }
  // history 是需要要相反的
  //ret = lm_mdl.FullScoreForgotState(history, history+order-1, vocab.Index(num2str(k)), out_state);
  ret = lm_mdl.FullScoreForgotState(history, history+uselen, vocab.Index(num2str(k)), out_state);
  float lm_score = ret.prob;
  //lm_score = log(pow(10, lm_score));
  lm_score *=2.302585;

  return lm_wght * lm_score;
}



void AddToList(const Prefix &  prf,int & beam, VECTOR & B,Prefix *  & prf_min){ 
    bool done=false;
    VECTOR::iterator it = find(B.begin(),B.end(),prf);
    if(it!=B.end()){ //存在的处理
        if(prf.ScoreForCompare() > (*it).ScoreForCompare()){
            Prefix &p=*it;
            p.Rnb=prf.Rnb;
            p.Rb=prf.Rb;
            p.makeScore();
            done=true;
        }
    }else{
        if(B.size()<beam){
            B.push_back(prf);
            done=true;
        }else{
            if(prf.ScoreForCompare() > prf_min->ScoreForCompare()){
                prf_min->path=prf.path;
                prf_min->Rnb=prf.Rnb;
                prf_min->Rb=prf.Rb;
                prf_min->makeScore();
				done=true;
            }
        }
    }
    if(done){
		if(B.size()==1) prf_min=&B[0];
		else{
			prf_min =&(* min_element(B.begin(),B.end()) );
		}
    }
}

// pf_list is std::list<Prefix>
std::vector<int32> BeamSearch(const Model& lm_mdl, const Matrix<BaseFloat>& nnet_out, int32 & beam, float & lm_wght, int & order)
{
  const int32 T = nnet_out.NumRows();
  const int32 pdfs = nnet_out.NumCols();
  // Initialize
  Prefix pf0, * prf_min=NULL;
  pf0.Rb=0;
  pf0.Rnb=kLogZeroFloat;
  pf0.makeScore();
  VECTOR BB[2]; // 0 表示B 1 表示Bw

  BB[0].push_back(pf0);

  const int turn=1; // 1 & -- 0 |
  float score_ext_lm=-1;
  float score_ext_nolm=-1;
  // Here we begin ...
  Timer tm;
  for (int32 t = 0; t < T; ++t) {
    //杜绝复制
    VECTOR& Bw=BB[turn&t];  //BW是已经有的
    VECTOR& B=BB[turn&(t-1)]; // B是即将要加入的
    B.clear(); //后面维护 B 小于 beam
	int Bw_size=Bw.size();
    for (int i=0; i<Bw_size; i++) {

      const Prefix &  prf = Bw[i]; //
      float Rb=prf.score + nnet_out(t, 0)
			,Rnb=kLogZeroFloat;

      const std::vector<int32>& y = prf.path;	 
      int y_size=y.size();
      int32 y_e = y_size==0?-1:y.back();
      bool push=true;
      if (y_size!=0){
        Rnb = prf.Rnb + nnet_out(t, y_e);

        Prefix prf_tmp(std::vector<int32>(y.begin(), y.end()-1));	// the prefix of y with y_e removed
        
        VECTOR::iterator iter =find(Bw.begin(),Bw.end(),prf_tmp);
        if (iter != Bw.end()){ //如果剩余部分能找到  找不到就算了
            score_ext_nolm = nnet_out(t, y_e) + 
				( (y_size>=2 && y[y_size-1]==y[y_size-2])?(*iter).Rb : LogAdd((*iter).Rb,(*iter).Rnb) );
            if(B.size()==beam){
				float tmp_Rnb=LogAdd(score_ext_nolm,Rnb);
				if(prf_min->ScoreForCompare() > LogAdd(Rb,tmp_Rnb)/(y_size+1)){
					push=false;
				}
			}else{
                score_ext_lm=LM_log_prob(lm_mdl,y_e,prf_tmp.path,lm_wght,order); 
                Rnb=LogAdd(score_ext_lm+score_ext_nolm,Rnb);
			}
        
		  }// if (iter != ...)
      } // if (y_size!=0)
	  if(push){
		Prefix new_prf(y,Rb,Rnb);
		new_prf.makeScore();
		AddToList(new_prf, beam, B,prf_min);
	  }
            //
      for (int32 k = 1; k != pdfs; ++k){
        score_ext_nolm = nnet_out(t, k) + (y_e==k?prf.Rb:LogAdd(prf.Rb,prf.Rnb));
	    if(B.size()==beam && prf_min->ScoreForCompare() > score_ext_nolm/(y_size+2) ){
        
		}else{
			score_ext_lm=LM_log_prob(lm_mdl,k,y,lm_wght,order);

			std::vector<int32> y_k = y;
		    y_k.push_back(k);
	        Prefix prf_tmp(y_k,kLogZeroFloat,0);
			prf_tmp.Rnb=score_ext_lm+score_ext_nolm;
		    prf_tmp.makeScore();
	        AddToList(prf_tmp, beam, B,prf_min);
        }

      } // for (int32 k = 1; ... )
    } // for (size_t B = 0; ... )
  } // for (int32 t = 0; ... )
  std::cout << "real time: " << tm.Elapsed()/(T/100) << std::endl;
  std::cout << "Time overall: " << tm.Elapsed() << std::endl;
  VECTOR & B=BB[turn&T];
  int size_B=B.size();
  sort(B.begin(),B.end());
  return B[size_B-1].path;
}

}  // namespace kaldi
}  // namespace nnet1


int main(int argc, char *argv[]) {

  
//  chj_logname="debug-beam1.log";
//  chj_setfile();

  using namespace kaldi;
  using namespace kaldi::nnet1;
  using namespace lm::ngram;
  typedef kaldi::int32 int32;  
// cerr<<(kLogZeroFloat==kLogZeroFloat)<<endl;
//cerr<<kLogZeroBaseFloat<<" "<<kLogZeroFloat<<endl;

  try {
    const char *usage =
        "Perform LSTM-CTC beam search  decoding.\n"
        " 不需要 char-int 的 map文件   \n"
        "Usage: progname  [options] \n"
        "e.g.: \n"
        " progname  ark:features.pdf.ark  arpa_lmfile   ark,t:decode-result.ark\n";


    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

	bool apply_log =false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    int32 beam=10;
    po.Register("beam", &beam, "beam for search");

    BaseFloat lm_wght=1.0;
    po.Register("lm-weight", &lm_wght, "beam for search");

    int32 order=2;
    po.Register("lm-order", &order, "order of language model");
    
    po.Read(argc, argv);

    if ( po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(beam > 0);
	std::string feature_rspecifier=po.GetArg(1);
    std::string   arpa_lmfile = po.GetArg(2);
    std::string result_wspecifier = po.GetArg(3);
        
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorWriter result_writer(result_wspecifier);

    Timer time;
    double time_now = 0;
    int32 num_done = 0;

    // Load ARPA LM
    Model lm_mdl(arpa_lmfile.c_str());
    
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> nnet_out_host = feature_reader.Value(); //一句话的信息。 
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << feature_reader.Key() 
                    << ", " << nnet_out_host.NumRows() << "frm";

      if(apply_log){
			nnet_out_host.ApplyLog();
	  }
      //check for NaN/inf
      for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
        for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r,c);
          if (val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
          if (val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
        }
      }

      // beam search
      std::vector<int32> best_path;
      best_path = BeamSearch(lm_mdl, nnet_out_host, beam, lm_wght, order);

      // write
      result_writer.Write(feature_reader.Key(), best_path);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += nnet_out_host.NumRows();
    }//feature finished!
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

