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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "CTC 只用后验概率 解码"
        "\n"
        "Usage:  chj-matrix-max-pdf-to-id  [options] \n"
        "e.g.: \n"
        "  chj-matrix-max-pdf-to-id ark:pdf.ark ark:max.id.ark  \n";

    ParseOptions po(usage);


    bool binary = true; 
    po.Register("binary", &binary, "Write output in binary mode");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
    
	bool ctc_shrink=false;
    po.Register("ctc-shrink", &ctc_shrink, "是否使用ctc的压缩方式");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      result_wspecifier = po.GetArg(2);
        
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif


    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    Int32VectorWriter result_writer(result_wspecifier);

	for ( ; !feature_reader.Done(); feature_reader.Next()) {
		CuMatrix<BaseFloat> feat =CuMatrix<BaseFloat>( feature_reader.Value() );
        CuArray<int32> max_id;
        std::vector<int32> max_id_host;
		int m=feat.NumRows();
        feat.FindRowMaxId(&max_id);
        max_id_host.resize(m);
        max_id.CopyToVec(&max_id_host);
        if(ctc_shrink){
            std::vector<int32>  shrink;
            int32 pp=-111;
            for(int32 i=0;i<m;i++){
                int32 p=max_id_host[i];
                if(pp!=p && p!=0){
                    shrink.push_back(p);
                }
                pp=p;
            }
            result_writer.Write(feature_reader.Key(), shrink);
        }else{
            result_writer.Write(feature_reader.Key(), max_id_host);
        }
		
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
