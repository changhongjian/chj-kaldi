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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "计算矩阵特征文件的列数"
        "\n"
        "Usage:  \n"
        "e.g.: \n"
        "feats-rows ark:pdf.ark  \n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1);
    kaldi::int64 m=0,n=0;
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

	for ( ; !feature_reader.Done(); feature_reader.Next()) {
		n+= feature_reader.Value().NumRows();
		m++;
    }
    std::cout<<n<<" / "<<m<<endl;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
