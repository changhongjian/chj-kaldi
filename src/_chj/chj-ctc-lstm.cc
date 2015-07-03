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
        "长虹剑的 CTC 训练脚本"
        "\n"
        "Usage:  chj-ctc-lstm [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " chj-ctc-lstm scp:feature.scp ark:targets.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");
    std::string objective_function = "CTC";
    po.Register("objective-function", &objective_function, "Objective function : CTC");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
    
    //<jiayu>
    int32 targets_delay=5;
    po.Register("targets-delay", &targets_delay, "---LSTM--- BPTT targets delay"); 

    int32 batch_size=20;
    po.Register("batch-size", &batch_size, "---LSTM--- BPTT batch size"); 

    int32 num_stream=4;
    po.Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training"); 

    int32 dump_interval=50000;
    po.Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping"); 
    //</jiayu>

    // Add dummy randomizer options, to make the tool compatible with standard scripts
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    bool randomize = false;
    po.Register("randomize", &randomize, "Dummy option, for compatibility...");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialInt32VectorReader targets_reader(targets_rspecifier);


    CTC ctc;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;

    //  book-keeping for multi-streams
    std::vector<std::string> keys(num_stream);
    std::vector<Matrix<BaseFloat> > feats(num_stream);

    // bptt batch buffer
    //int32 feat_dim = nnet.InputDim();
    CuMatrix<BaseFloat> feat_transf, nnet_out, obj_diff;

    num_stream = 1;
    nnet.SetStream(num_stream); 
	std::vector<int> new_utt_flags(num_stream, 1);
	
//    chj_logname="chj-cudacs.log" ;  

	for ( ; !feature_reader.Done(); feature_reader.Next(),targets_reader.Next()) {
		CuMatrix<BaseFloat> feat = CuMatrix<BaseFloat>(feature_reader.Value());
		std::vector<int32> target=targets_reader.Value();
		//处理matrix
		nnet_transf.Feedforward(feat, &feat_transf);
		nnet.Reset(new_utt_flags);
        int len=feat_transf.NumRows();
        for(int t=0;t<len;t++){
            if(t+targets_delay<len){
                feat_transf.Row(t).CopyFromVec(feat_transf.Row(t+targets_delay));
            }else{
                feat_transf.Row(t).CopyFromVec(feat_transf.Row(len));
            }
        }
        nnet.Propagate(feat_transf, &nnet_out);
        if (objective_function == "CTC") {
			//ctc.Run(nnet_out,target,&diff_cs);
			ctc.RunSpeedUp(nnet_out,target,&obj_diff);	

			//Matrix<BaseFloat> df1(diff_cs);
			//Matrix<BaseFloat> df2(obj_diff);
			
        } else {
            KALDI_ERR << "Unknown objective function code : " << objective_function;
        }
        // backward pass
        if (!crossvalidate) { // not use just for compatibility
            nnet.Backpropagate(obj_diff, NULL);
        }

        // 1st minibatch : show what happens in network 
        if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
            KALDI_VLOG(1) << "### After " << total_frames << " frames,";
            KALDI_VLOG(1) << nnet.InfoPropagate();
            if (!crossvalidate) {
                KALDI_VLOG(1) << nnet.InfoBackPropagate();
                KALDI_VLOG(1) << nnet.InfoGradient();
            }
        }

        int frame_progress = feat_transf.NumRows();
        total_frames += frame_progress;

        int num_done_progress = 1;  // I change
        num_done += num_done_progress;
         
        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
            if ((total_frames-frame_progress)/25000 != (total_frames/25000)) { // print every 25k frames
                KALDI_VLOG(2) << "### After " << total_frames << " frames,";
                KALDI_VLOG(2) << nnet.InfoPropagate();
                if (!crossvalidate) {
                    KALDI_VLOG(2) << nnet.InfoBackPropagate();
                    KALDI_VLOG(2) << nnet.InfoGradient();
                }
            }
        }

        // report the speed
        if ((num_done-num_done_progress)/1000 != (num_done/1000)) {
            double time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
            
#if HAVE_CUDA==1
            // check the GPU is not overheated
            CuDevice::Instantiate().CheckGpuHealth();
#endif
        }
		if ((num_done-num_done_progress)/dump_interval != (num_done/dump_interval)) {
            char nnet_name[512];
            if (!crossvalidate) {
                sprintf(nnet_name, "%s_utt%d", target_model_filename.c_str(), num_done);
                nnet.Write(nnet_name, binary);
            }
        }
	}
    
    // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  

    if (objective_function == "CTC") {
      KALDI_LOG << ctc.Report();
    } else {
      KALDI_ERR << "Unknown objective function code : " << objective_function;
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
