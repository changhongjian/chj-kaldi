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
#include "_chj/chj-nnet-loss.h"  需要换成自己的 ctc loss 头文件



int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "长虹剑的 CTC 训练脚本,LSTM多流"
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
    std::vector<int32> targets[num_stream];
    std::vector<int32> lens(num_stream);
    
    // bptt batch buffer
    int32 feat_dim = nnet.InputDim();
	int32 netout_dim=nnet.OutputDim();
    CuMatrix<BaseFloat> feat_transf, nnet_out, obj_diff;
    CuMatrix<BaseFloat> stream_feats;
	//chj_setfile();
    //chj_logname="chj-streamcs.log"; 
	//chj_setfile();
	std::vector<int> new_utt_flags(num_stream, 1); //每次都更新，因为这里就是一句话
	int32 maxlen=0;
    int32 use_num_stream=0;
// change stream
	nnet.SetStream(num_stream);

	while(!feature_reader.Done()){ //不知道会不会少一句话
		int s=0;
        int t=0;
		CuMatrix<BaseFloat>  feats[num_stream];
        CuMatrix<BaseFloat>  diffs;
        CuMatrix<BaseFloat> cufeat;
        for(s=0;s<num_stream && !feature_reader.Done();s++,feature_reader.Next(),targets_reader.Next() ){
            //cufeat[s]=CuMatrix<BaseFloat>(feature_reader.Value());
            cufeat=CuMatrix<BaseFloat>(feature_reader.Value());
            nnet_transf.Feedforward( cufeat , &feat_transf); 
            feats[s]=feat_transf;
            targets[s]=targets_reader.Value();
            lens[s]=feats[s].NumRows(); // it means T
        }
        use_num_stream=s; //是多大就是多大

		if(use_num_stream != num_stream) {
			nnet.SetStream(use_num_stream);
			new_utt_flags.resize(use_num_stream,1);
			// 小心写入的时候还要还原回去 
		}
        maxlen=*std::max_element(lens.begin(), lens.begin()+s);
        int32 all_rows=maxlen*use_num_stream;
        stream_feats.Resize(all_rows,feat_dim,kSetZero);
        s=0;
        t=0;
        for(int i=0;i<all_rows;i++){
            if(t<lens[s]){
                if(t+targets_delay<lens[s]){
                    stream_feats.Row(i).CopyFromVec(feats[s].Row(t+targets_delay));
                }else{
                    stream_feats.Row(i).CopyFromVec(feats[s].Row(lens[s]-1)); // i=t * use_num_stream + s
                }
            }// 也就是不够的话其余全为默认的值0
            ++s;
            if(s==use_num_stream){
                s=0;
                ++t;
            }
        }
        nnet.Reset(new_utt_flags);
        nnet.Propagate(stream_feats, &nnet_out);
		if (objective_function == "CTC") {
            // feats  这里又代表单个nnet_out了

            for(s=0;s<use_num_stream;s++){
                feats[s].Resize(maxlen,netout_dim,kSetZero);
            }
            s=t=0;
            for(int i=0;i<all_rows;i++){
                if(t<lens[s]){
                    feats[s].Row(t).CopyFromVec(nnet_out.Row(i));
                }// 也就是不够的话其余全为默认的值0
                ++s;
                if(s==use_num_stream){
                    s=0;
                    ++t;
                }
            }
			obj_diff.Resize(all_rows,netout_dim,kSetZero);

			for(s=0;s<use_num_stream;s++){
			/* 需要换成自己的*/
                ctc.RunSpeedUp(feats[s],targets[s],&diffs,lens[s]);   // 也就是列数的缩小
			//	ctc.Run(feats[s],targets[s],&diffs,lens[s]);
				for(int i=0;i<lens[s];i++){
					obj_diff.Row(i*use_num_stream+s).CopyFromVec(diffs.Row(i));
				}
            }
        } else {
            KALDI_ERR << "Unknown objective function code : " << objective_function;
        }
        // backward pass
        if (!crossvalidate) {
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

        int frame_progress = feat_transf.NumRows(); ///????????????
        total_frames += frame_progress;

        int num_done_progress = 0; ///????
        num_done += use_num_stream;
        
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
				//if(use_num_stream != num_stream) {
		        //    nnet.SetStream(num_stream); //小心
                //}
                    
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
		if(use_num_stream != num_stream) {
            nnet.SetStream(num_stream); //小心
        }
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
