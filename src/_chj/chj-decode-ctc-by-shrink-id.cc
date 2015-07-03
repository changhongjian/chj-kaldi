// nnetbin/bd-nnet-train-lstm-streams.cc

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
///////----------------
#include "_chj/chj-nnet-loss.h"
#include "_chj/chj.h"

int editDistance(int * src,int m,int * des,int n){
//dp and roll array
	//chj_ss<<endl<<"res: "<<m<<" / "<<n<<endl;chj_pt();
	int a,b;
	int dp[2][n+1];
	for(int j=0;j<=n;j++){
		dp[0][j]=j;
	}
	dp[0][0]=0;
	for(int i=1;i<=m;i++){
		dp[1&i][0]=i;
		for(int j=1;j<=n;j++){
			a=dp[1&(i-1)][j-1];
		//	chj_ss<<a<<" ";
			if(src[i-1]!=des[j-1]) a++;
			b=dp[1&(i-1)][j]+1;
		//	chj_ss<<b<<" ";
			if(a>b)a=b;
			b=dp[1&i][j-1]+1;
		//	chj_ss<<b<<" ";
			if(a>b)a=b;
			dp[1&i][j]=a;
			//chj_ss<<a<<" "<<(1&i)<<" -- "<<j<<" ";
//			chj_ss<<a<<" ";
		}
//		chj_ss<<endl;chj_pt();
	}
//	chj_ss<<endl;chj_pt();
	int rt=dp[1&m][n];
	//KALDI_LOG <<rt<<" "<<m<<" "<<n;
	return rt;
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "CTC 只用后验概率 解码"
        "\n"
        "Usage:  chj-decode-ctc-by-shrink-id [options] \n"
        "e.g.: \n"
        "  chj-decode-ctc-by-shrink-id ark:feat.id.ark ark:targets.id.ark result.txt  \n";

    ParseOptions po(usage);


    bool binary = true; 
    po.Register("binary", &binary, "Write output in binary mode");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
    
//	std::string hd_type="";
//    po.Register("hd_type", &hd_type, "I don't want to create another cc file");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      result  = po.GetArg(3);
        
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif


    SequentialBaseFloatVectorReader feature_reader(feature_rspecifier);
    //SequentialBaseFloatVectorReader targets_reader(targets_rspecifier);
	SequentialInt32VectorReader targets_reader(targets_rspecifier);
	

    Timer time;

    int32 num_done = 0, total_frames=0;
    int32 all_phones=0,edit_dist=0;
	
	for ( ; !feature_reader.Done(); feature_reader.Next(),targets_reader.Next()) {
		Vector<BaseFloat> feat = feature_reader.Value();
		//Vector<BaseFloat> target= targets_reader.Value();
	//	Vector<int32> target= targets_reader.Value();
		std::vector<int32> t=targets_reader.Value();
		Vector<BaseFloat> target;
		//处理vector
		int n=target.Dim();
		int m=feat.Dim();
		//KALDI_ASSERT( m>=n && n>0  );
		int32 * des=new int[n];
		int32 * src=new int[m];
		for(int32 i=0;i<n;i++){
			des[i]=(int32)target(i);
		}
		for(int32 i=0;i<m;i++){
			src[i]=(int32)feat(i);
		}
		all_phones+=n;
		//edit_dist+=editDistanceWithPath(src,mm,des,n);
        edit_dist+=editDistance(src,m,des,n);
		delete [] des;
		delete [] src;
	
		num_done+=1;
//		break;
    }

    // after last minibatch : show what happens in network 
    KALDI_LOG << "### After " << num_done << " sentences,PER is >>"<<(edit_dist*100.0/all_phones)<<"% <<"<<endl;
	

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
