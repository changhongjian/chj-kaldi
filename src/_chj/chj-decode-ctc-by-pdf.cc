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

int editDistanceWithPath(int * src,int m,int * des,int n){
	int a,b,c,t;
	int i,j,rt;
	int dp[m+1][n+1];
	for(j=0;j<=n;j++) dp[0][j]=j;
	dp[0][0]=0;
	for(i=1;i<=m;i++){
		dp[i][0]=i;
		for(j=1;j<=n;j++){
			a=dp[i-1][j-1];
			if(src[i-1]!=des[j-1])a++;
			b=dp[i][j-1]+1;
			if(a>b)a=b;
			b=dp[i-1][j]+1;
			if(a>b)a=b;
			dp[i][j]=a;
//			chj_ss<<a<<" ";
		}
//		chj_ss<<endl;chj_pt();
	}
    rt=dp[m][n];
//	KALDI_LOG <<rt<<" "<<m<<" "<<n;
	return rt;
//	exit(0);
	rt=0;
	i=m; j=n;
	while(i!=0 && j!=0){
		t=dp[i][j];
		a=dp[i-1][j-1];
		if(t==a){
			chj_ss<<endl<<"right: "<<i<<" / "<<j<<endl;chj_pt();
			rt++;
			i--; j--; continue;
		}
		b=dp[i][j-1];
		if(t==b){
			j--; continue;
		}
		c=dp[i-1][j];
		if(t==c){
            i--; continue;
        }
		if(a==t-1){
			i--; j--; continue;
		}
		if(b==t-1){
			j--; continue;
		}
		
		if(c==t-1){
			i--; continue;
		}
//		chj_ss<<endl<<a<<" "<<b<<" "<<c<<" "<<t<<" "<<i<<" "<<j<<endl;chj_pt();

	} 
//	KALDI_LOG <<rt<<" "<<m<<" "<<n;
	return m-rt;
}
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
        "Usage:  chj-decode-ctc-by-pdf  [options] \n"
        "e.g.: \n"
        "  chj-decode-ctc-by-pdf ark:pdf.ark ark:targets.ark result.txt  \n";

    ParseOptions po(usage);


    bool binary = true; 
    po.Register("binary", &binary, "Write output in binary mode");

    std::string use_gpu="yes";
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


    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatVectorReader targets_reader(targets_rspecifier);
	

    Timer time;

    int32 num_done = 0, total_frames=0;
    int32 all_phones=0,edit_dist=0;
	
	int aaa=0,bbb=0;

	for ( ; !feature_reader.Done(); feature_reader.Next(),targets_reader.Next()) {
		Matrix<BaseFloat> feat = feature_reader.Value();
		Vector<BaseFloat> target_tmp= targets_reader.Value();
	//	std::vector<int32> target_tmp=targets_reader.Value();
		//处理vector
		int n=target_tmp.Dim()/2;
		int m=feat.NumRows();
		int c=feat.NumCols();
		KALDI_ASSERT( m>=n && n>0  );
		int32 * des=new int[n];
		int32 * src=new int[m];
		int32 mm=0,pp=-111;
		for(int32 i=0;i<n;i++){
			des[i]=(int32)target_tmp(1+2*i);
		}
		for(int32 i=0;i<m;i++){
			BaseFloat a=feat(i,0);
			int32 p=0;
			for(int32 j=1;j<c;j++){
				if(feat(i,j)>a){
					a=feat(i,j);
					p=j;
				}
			}
			if(pp!=p && p!=0){
				src[mm++]=p;
			}
			pp=p;
		}
/*
		for(int i=0;i<mm;i++){
			KALDI_LOG<<src[i]<<" ";
		}
		KALDI_LOG<<endl;
		for(int i=0;i<n;i++){
			KALDI_LOG<<des[i]<<" ";
		}
*/
		all_phones+=n;
		aaa+=n;
		//bbb+=editDistanceWithPath(src,mm,des,n);
		//edit_dist+=editDistanceWithPath(src,mm,des,n);
        edit_dist+=editDistance(src,mm,des,n);
		delete [] des;
		delete [] src;
	
		num_done+=1;
//		break;
    }
	
    // after last minibatch : show what happens in network 
    KALDI_LOG << "### After " << num_done << " sentences,PER is >>"<<(edit_dist*100.0/all_phones)<<"% <<"<<endl;
//	KALDI_LOG << "### After " << num_done << " sentences,PER is >>"<<(bbb*1.0/aaa)<<"% <<"<<endl;


#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
