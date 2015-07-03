// nnetbin/bd-nnet-train-lstm-streams.cc

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of LSTM training by Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "The updates are done per-utternace, shuffling options are dummy for compatibility reason.\n"
        "\n"
        "Usage:  bd-nnet-train-lstm-streams [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " bd-nnet-train-lstm-streams scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");
    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

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
    RandomAccessPosteriorReader target_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Xent xent;
    Mse mse;
    
    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;

    //  book-keeping for multi-streams
    std::vector<std::string> keys(num_stream);
    std::vector<Matrix<BaseFloat> > feats(num_stream);
    std::vector<Posterior> targets(num_stream);
    std::vector<int> curt(num_stream, 0);
    std::vector<int> lent(num_stream, 0);
    std::vector<int> new_utt_flags(num_stream, 0);
    // bptt batch buffer
    int32 feat_dim = nnet.InputDim();
    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
    Matrix<BaseFloat> feat(batch_size * num_stream, batch_size==0?0:feat_dim, kSetZero);
    Posterior target(batch_size * num_stream);
    CuMatrix<BaseFloat> feat_transf, nnet_out, obj_diff;
    
    int use_batch_size=batch_size;
    while (1) {
        for (int s = 0; s < num_stream; s++) {
            if (curt[s] < lent[s]) {
                new_utt_flags[s] = 0;
                continue;
            }
            while (!feature_reader.Done()) {
                keys[s]  = feature_reader.Key();
                nnet_transf.Feedforward( CuMatrix<BaseFloat>(feature_reader.Value()), &feat_transf);
                feats[s]=Matrix<BaseFloat>(feat_transf);
                if (!target_reader.HasKey(keys[s])) {
                    KALDI_WARN << keys[s] << ", missing targets";
                    num_no_tgt_mat++;
                    feature_reader.Next();
                    continue;
                }
                targets[s] = target_reader.Value(keys[s]);
                if ( batch_size!=0 && feats[s].NumRows() != targets[s].size()) {
                    KALDI_WARN << keys[s] << ", length miss-match between feats and targets, skip";
                    feature_reader.Next();
                    continue;
                }
                curt[s] = 0;
                lent[s] = feats[s].NumRows();
                new_utt_flags[s] = 1;  // a new utterance feeded to this stream
                feature_reader.Next();
                break;
            }
        }
        int done = 1;
        for (int s = 0; s < num_stream; s++) {
            if (curt[s] < lent[s]) done = 0;  // this stream still contains valid data, not exhausted
        }
        if (done) break;
		if(batch_size==0){
	        use_batch_size = *std::max_element(lent.begin(), lent.begin()+num_stream);
			frame_mask.Resize(use_batch_size * num_stream,kSetZero);
			feat.Resize(use_batch_size * num_stream, feat_dim, kSetZero);
			target=Posterior(use_batch_size * num_stream);
		}
        for (int t = 0; t < use_batch_size; t++) {
            for (int s = 0; s < num_stream; s++) {
                // frame_mask & targets padding
                if (curt[s] < lent[s]) {
                    frame_mask(t * num_stream + s) = 1;
                    target[t * num_stream + s] = targets[s][curt[s]];

                } else {
                    frame_mask(t * num_stream + s) = 0;
                    target[t * num_stream + s] = targets[s][lent[s]-1];
                }

//std::cout<< feat.Row(t * num_stream + s).Dim()<<" "  <<feats[s].Row(curt[s]+targets_delay).Dim()  <<endl;
                // feat shifting & padding
                if (curt[s] + targets_delay < lent[s]) {
                    feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]+targets_delay));
                } else {
                    feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(lent[s]-1));
                }
                curt[s]++;
            }
        }
/*
		if(batch_size==0){
			for (int s = 0; s < num_stream; s++) {
                curt[s]=lent[s];
            }
		}
*/
        // for streams with new utterance, history states need to be reset
        nnet.Reset(new_utt_flags);

        // apply optional feature transform
       // nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feat_transf);

        // forward pass
        nnet.Propagate(CuMatrix<BaseFloat>(feat), &nnet_out);
    
        // evaluate objective function we've chosen
        if (objective_function == "xent") {
            xent.EvalMasked(frame_mask, nnet_out, target, &obj_diff);
        //} else if (objective_function == "mse") {     // not supported yet
        //    mse.Eval(nnet_out, targets_batch, &obj_diff);
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

        int frame_progress = frame_mask.Sum();
        total_frames += frame_progress;

        int num_done_progress = 0;
        for (int i =0; i < new_utt_flags.size(); i++) {
            num_done_progress += new_utt_flags[i];
        }
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

    if (objective_function == "xent") {
      KALDI_LOG << xent.Report();
    } else if (objective_function == "mse") {
      KALDI_LOG << mse.Report();
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
