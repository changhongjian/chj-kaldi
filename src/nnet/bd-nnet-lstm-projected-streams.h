// nnet/bd-nnet-lstm-projected-streams.h

#ifndef BD_KALDI_NNET_LSTM_PROJECTED_STREAMS_H_
#define BD_KALDI_NNET_LSTM_PROJECTED_STREAMS_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"
#include "_chj/chj.h"
/*************************************
 * x: input neuron
 * g: squashing neuron near input
 * i: Input gate
 * f: Forget gate
 * o: Output gate
 * c: memory Cell (CEC)
 * h: squashing neuron near output
 * m: output neuron of Memory block
 * r: recurrent projection neuron
 * y: output neuron of LSTMP
 *************************************/

namespace kaldi {
namespace nnet1 {
class LstmProjectedStreams : public UpdatableComponent {
public:
    LstmProjectedStreams(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        ncell_(0),
        nrecur_(output_dim),
        //dropout_rate_(0.0),
        nstream_(0)
    { }

    ~LstmProjectedStreams()
    { }

    Component* Copy() const { return new LstmProjectedStreams(*this); }
    ComponentType GetType() const { return myLstmProjectedStreams; }
    
    void SetStream(int32 num_stream){
	nstream_=num_stream;

        prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);
    }
    static void InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
//chj_pt("222    in  initamt\n");
        m.SetRandUniform();  // uniform in [0, 1]
        m.Add(-0.5);         // uniform in [-0.5, 0.5]
        m.Scale(2 * scale);  // uniform in [-scale, +scale]
    }

    static void InitVecParam(CuVector<BaseFloat> &v, float scale) {
//chj_pt("333    in initvec\n");
        Vector<BaseFloat> tmp(v.Dim());
        for (int i=0; i < tmp.Dim(); i++) {
            tmp(i) = (RandUniform() - 0.5) * 2 * scale;
        }
        v = tmp;
    }

    void InitData(std::istream &is) {
//chj_pt("444    in initdata\n");
        // define options
        float param_scale = 0.02;
        // parse config
        std::string token;
        while (!is.eof()) {
            ReadToken(is, false, &token); 
            if (token == "<CellDim>") 
                ReadBasicType(is, false, &ncell_);
            else if (token == "<NumStream>") 
                ReadBasicType(is, false, &nstream_);
            //else if (token == "<DropoutRate>") 
            //    ReadBasicType(is, false, &dropout_rate_);
            else if (token == "<ParamScale>") 
                ReadBasicType(is, false, &param_scale);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                           << " (CellDim|NumStream|ParamScale)";
                           //<< " (CellDim|NumStream|DropoutRate|ParamScale)";
            is >> std::ws;
        }

        prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);

        // init weight and bias (Uniform)
        w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);  InitMatParam(w_gifo_x_, param_scale);
        w_gifo_r_.Resize(4*ncell_, nrecur_, kUndefined);     InitMatParam(w_gifo_r_, param_scale);
        w_r_m_.Resize(nrecur_, ncell_, kUndefined);          InitMatParam(w_r_m_, param_scale);

        bias_.Resize(4*ncell_, kUndefined);        InitVecParam(bias_, param_scale);
        peephole_i_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_i_c_, param_scale);
        peephole_f_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_f_c_, param_scale);
        peephole_o_c_.Resize(ncell_, kUndefined);  InitVecParam(peephole_o_c_, param_scale);

        // init delta buffers
        w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero); 
        w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);    
        bias_corr_.Resize(4*ncell_, kSetZero);     

        peephole_i_c_corr_.Resize(ncell_, kSetZero);
        peephole_f_c_corr_.Resize(ncell_, kSetZero);
        peephole_o_c_corr_.Resize(ncell_, kSetZero);

        w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero); 

    }

    void ReadData(std::istream &is, bool binary) {
//chj_pt("555    in readdata\n");
        ExpectToken(is, binary, "<CellDim>");
        ReadBasicType(is, binary, &ncell_);
        ExpectToken(is, binary, "<NumStream>");
        ReadBasicType(is, binary, &nstream_);
        //ExpectToken(is, binary, "<DropoutRate>");
        //ReadBasicType(is, binary, &dropout_rate_);



        w_gifo_x_.Read(is, binary);
        w_gifo_r_.Read(is, binary);
        bias_.Read(is, binary);

        peephole_i_c_.Read(is, binary);
        peephole_f_c_.Read(is, binary);
        peephole_o_c_.Read(is, binary);

        w_r_m_.Read(is, binary);

        prev_nnet_state_.Resize(nstream_, 7*ncell_ + 1*nrecur_, kSetZero);

        // init delta buffers
        w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero); 
        w_gifo_r_corr_.Resize(4*ncell_, nrecur_, kSetZero);    
        bias_corr_.Resize(4*ncell_, kSetZero);     

        peephole_i_c_corr_.Resize(ncell_, kSetZero);
        peephole_f_c_corr_.Resize(ncell_, kSetZero);
        peephole_o_c_corr_.Resize(ncell_, kSetZero);

        w_r_m_corr_.Resize(nrecur_, ncell_, kSetZero); 
    }

    void WriteData(std::ostream &os, bool binary) const {
        WriteToken(os, binary, "<CellDim>");
        WriteBasicType(os, binary, ncell_);
        WriteToken(os, binary, "<NumStream>");
        WriteBasicType(os, binary, nstream_);
        //WriteToken(os, binary, "<DropoutRate>");
        //WriteBasicType(os, binary, dropout_rate_);

        w_gifo_x_.Write(os, binary);
        w_gifo_r_.Write(os, binary);
        bias_.Write(os, binary);

        peephole_i_c_.Write(os, binary);
        peephole_f_c_.Write(os, binary);
        peephole_o_c_.Write(os, binary);

        w_r_m_.Write(os, binary);
    }

    int32 NumParams() const { 
//chj_pt("666   in numpara\n");
        return ( w_gifo_x_.NumRows() * w_gifo_x_.NumCols() +
                 w_gifo_r_.NumRows() * w_gifo_r_.NumCols() +
                 bias_.Dim() +
                 peephole_i_c_.Dim() +
                 peephole_f_c_.Dim() +
                 peephole_o_c_.Dim() +
                 w_r_m_.NumRows() * w_r_m_.NumCols() );
    }

    void GetParams(Vector<BaseFloat>* wei_copy) const {
//chj_pt("777    in getpara\n");
        wei_copy->Resize(NumParams());

        int32 offset, len;

        offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_gifo_x_);

        offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_gifo_r_);

        offset += len; len = bias_.Dim();
        wei_copy->Range(offset, len).CopyFromVec(bias_);

        offset += len; len = peephole_i_c_.Dim();
        wei_copy->Range(offset, len).CopyFromVec(peephole_i_c_);

        offset += len; len = peephole_f_c_.Dim();
        wei_copy->Range(offset, len).CopyFromVec(peephole_f_c_);

        offset += len; len = peephole_o_c_.Dim();
        wei_copy->Range(offset, len).CopyFromVec(peephole_o_c_);

        offset += len; len = w_r_m_.NumRows() * w_r_m_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_r_m_);

        return;
    }
    std::string Info() const {
        return std::string("    ") + 
            "\n  w_gifo_x_  "     + MomentStatistics(w_gifo_x_) + 
            "\n  w_gifo_r_  "     + MomentStatistics(w_gifo_r_) +
            "\n  bias_  "         + MomentStatistics(bias_) +
            "\n  peephole_i_c_  " + MomentStatistics(peephole_i_c_) +
            "\n  peephole_f_c_  " + MomentStatistics(peephole_f_c_) +
            "\n  peephole_o_c_  " + MomentStatistics(peephole_o_c_) +
            "\n  w_r_m_  "        + MomentStatistics(w_r_m_);
    }
  
    std::string InfoGradient() const {
        return std::string("    ") + 
            "\n  w_gifo_x_corr_  "     + MomentStatistics(w_gifo_x_corr_) + 
            "\n  w_gifo_r_corr_  "     + MomentStatistics(w_gifo_r_corr_) +
            "\n  bias_corr_  "         + MomentStatistics(bias_corr_) +
            "\n  peephole_i_c_corr_  " + MomentStatistics(peephole_i_c_corr_) +
            "\n  peephole_f_c_corr_  " + MomentStatistics(peephole_f_c_corr_) +
            "\n  peephole_o_c_corr_  " + MomentStatistics(peephole_o_c_corr_) +
            "\n  w_r_m_corr_  "        + MomentStatistics(w_r_m_corr_);
    }

    void Reset(std::vector<int> &stream_reset_flag) {
//chj_pt("777    in reset\n");
        // reset flag: 1 - reset stream network state
        KALDI_ASSERT(prev_nnet_state_.NumRows() == stream_reset_flag.size());
        for (int s = 0; s < stream_reset_flag.size(); s++) {
            if (stream_reset_flag[s] == 1) {
                prev_nnet_state_.Row(s).SetZero();
            }
        }
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
//chj_pt("888    in prog\n");
        int DEBUG = 0;
//sprintf(chj_str,"numrows: %d nstream: %d   +++_+++ \n ",in.NumRows(),nstream_);
//chj_pt();
        KALDI_ASSERT(in.NumRows() % nstream_ == 0);
        int32 T = in.NumRows() / nstream_;
        int32 S = nstream_;

        // 0:forward pass history, [1, T]:current sequence, T+1:dummy
        propagate_buf_.Resize((T+2)*S, 7 * ncell_ + nrecur_, kSetZero);  
        propagate_buf_.RowRange(0*S,S).CopyFromMat(prev_nnet_state_);

        // disassemble entire neuron activation buffer into different neurons
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_)); CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_)); CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_)); CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*ncell_, nrecur_));

        CuSubMatrix<BaseFloat> YGIFO(propagate_buf_.ColRange(0, 4*ncell_));

        // x -> g, i, f, o, not recurrent, do it all in once
        YGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, w_gifo_x_, kTrans, 0.0);
        //// LSTM forward dropout 
        //// Google paper 2014: Recurrent Neural Network Regularization
        //// by Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals
        //if (dropout_rate_ != 0.0) {
        //    dropout_mask_.Resize(in.NumRows(), 4*ncell_, kUndefined);
        //    dropout_mask_.SetRandUniform();     // [0,1]
        //    dropout_mask_.Add(-dropout_rate_);  // [-dropout_rate, 1-dropout_rate_],
        //    dropout_mask_.ApplyHeaviside();     // -tive -> 0.0, +tive -> 1.0
        //    YGIFO.RowRange(1*S,T*S).MulElements(dropout_mask_);
        //}

        // bias -> g, i, f, o
        YGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_);

        for (int t = 1; t <= T; t++) {
            // multistream buffers for current time-step
            CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S,S));  

            CuSubMatrix<BaseFloat> y_gifo(YGIFO.RowRange(t*S,S));
    
            // r(t-1) -> g, i, f, o
            y_gifo.AddMatMat(1.0, YR.RowRange((t-1)*S,S), kNoTrans, w_gifo_r_, kTrans,  1.0);

            // c(t-1) -> i(t) via peephole
            y_i.AddMatDiagVec(1.0, YC.RowRange((t-1)*S,S), kNoTrans, peephole_i_c_, 1.0);

            // c(t-1) -> f(t) via peephole
            y_f.AddMatDiagVec(1.0, YC.RowRange((t-1)*S,S), kNoTrans, peephole_f_c_, 1.0);

            // i, f sigmoid squashing
            y_i.Sigmoid(y_i);
            y_f.Sigmoid(y_f);
    
            // g tanh squashing
            y_g.Tanh(y_g);
    
            // g -> c
            y_c.AddMatDotMat(1.0, y_g, kNoTrans, y_i, kNoTrans, 0.0);

            // c(t-1) -> c(t) via forget-gate
            y_c.AddMatDotMat(1.0, YC.RowRange((t-1)*S,S), kNoTrans, y_f, kNoTrans, 1.0);

            y_c.ApplyFloor(-50);   // optional clipping of cell activation
            y_c.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR
    
            // h tanh squashing
            y_h.Tanh(y_c);
    
            // c(t) -> o(t) via peephole (non-recurrent) & o squashing
            y_o.AddMatDiagVec(1.0, y_c, kNoTrans, peephole_o_c_, 1.0);

            // o sigmoid squashing
            y_o.Sigmoid(y_o);
    
            // h -> m via output gate
            y_m.AddMatDotMat(1.0, y_h, kNoTrans, y_o, kNoTrans, 0.0);
            
            // m -> r
            y_r.AddMatMat(1.0, y_m, kNoTrans, w_r_m_, kTrans, 0.0);

            if (DEBUG) {
                std::cerr << "forward-pass frame " << t << "\n";
                std::cerr << "activation of g: " << y_g;
                std::cerr << "activation of i: " << y_i;
                std::cerr << "activation of f: " << y_f;
                std::cerr << "activation of o: " << y_o;
                std::cerr << "activation of c: " << y_c;
                std::cerr << "activation of h: " << y_h;
                std::cerr << "activation of m: " << y_m;
                std::cerr << "activation of r: " << y_r;
            }
        }

        // recurrent projection layer is also feed-forward as LSTM output
        out->CopyFromMat(YR.RowRange(1*S,T*S));

        // now the last frame state becomes previous network state for next batch
        prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S,S));
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        int DEBUG = 0;

        int32 T = in.NumRows() / nstream_;
        int32 S = nstream_;

        // disassemble propagated buffer into neurons
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*ncell_, nrecur_));
    
        // 0:dummy, [1,T] frames, T+1 backward pass history
        backpropagate_buf_.Resize((T+2)*S, 7 * ncell_ + nrecur_, kSetZero);

        // disassemble backpropagate buffer into neurons
        CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*ncell_, ncell_));
        CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(7*ncell_, nrecur_));

        CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_.ColRange(0, 4*ncell_));

        // projection layer to LSTM output is not recurrent, so backprop it all in once
        DR.RowRange(1*S,T*S).CopyFromMat(out_diff);

        for (int t = T; t >= 1; t--) {
            CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S,S));  
    
            CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_i(DI.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_f(DF.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_o(DO.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_c(DC.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_r(DR.RowRange(t*S,S));
    
            // r
            //   Version 1 (precise gradients): 
            //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
            d_r.AddMatMat(1.0, DGIFO.RowRange((t+1)*S,S), kNoTrans, w_gifo_r_, kNoTrans, 1.0);

            /*
            //   Version 2 (Alex Graves' PhD dissertation): 
            //   only backprop g(t+1) to r(t) 
            CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, ncell_));
            d_r.AddMatMat(1.0, DG.RowRange((t+1)*S,S), kNoTrans, w_g_r_, kNoTrans, 1.0);
            */

            /*
            //   Version 3 (Felix Gers' PhD dissertation): 
            //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
            //   CEC(with forget connection) is the only "error-bridge" through time
            ;
            */
    
            // r -> m
            d_m.AddMatMat(1.0, d_r, kNoTrans, w_r_m_, kNoTrans, 0.0);
    
            // m -> h via output gate
            d_h.AddMatDotMat(1.0, d_m, kNoTrans, y_o, kNoTrans, 0.0);
            d_h.DiffTanh(y_h, d_h);
    
            // o
            d_o.AddMatDotMat(1.0, d_m, kNoTrans, y_h, kNoTrans, 0.0);
            d_o.DiffSigmoid(y_o, d_o);
    
            // c
            //   1. diff from h(t)
            //   2. diff from c(t+1) (via forget-gate between CEC)
            //   3. diff from i(t+1) (via peephole)
            //   4. diff from f(t+1) (via peephole)
            //   5. diff from o(t)   (via peephole, not recurrent)
            d_c.AddMat(1.0, d_h);  
            d_c.AddMatDotMat(1.0, DC.RowRange((t+1)*S,S), kNoTrans, YF.RowRange((t+1)*S,S), kNoTrans, 1.0);
            d_c.AddMatDiagVec(1.0, DI.RowRange((t+1)*S,S), kNoTrans, peephole_i_c_, 1.0);
            d_c.AddMatDiagVec(1.0, DF.RowRange((t+1)*S,S), kNoTrans, peephole_f_c_, 1.0);
            d_c.AddMatDiagVec(1.0, d_o                   , kNoTrans, peephole_o_c_, 1.0);
    
            // f
            d_f.AddMatDotMat(1.0, d_c, kNoTrans, YC.RowRange((t-1)*S,S), kNoTrans, 0.0);
            d_f.DiffSigmoid(y_f, d_f);
    
            // i
            d_i.AddMatDotMat(1.0, d_c, kNoTrans, y_g, kNoTrans, 0.0);
            d_i.DiffSigmoid(y_i, d_i);
    
            // c -> g via input gate
            d_g.AddMatDotMat(1.0, d_c, kNoTrans, y_i, kNoTrans, 0.0);
            d_g.DiffTanh(y_g, d_g);
    
            // debug info
            if (DEBUG) {
                std::cerr << "backward-pass frame " << t << "\n";
                std::cerr << "derivative wrt input r " << d_r;
                std::cerr << "derivative wrt input m " << d_m;
                std::cerr << "derivative wrt input h " << d_h;
                std::cerr << "derivative wrt input o " << d_o;
                std::cerr << "derivative wrt input c " << d_c;
                std::cerr << "derivative wrt input f " << d_f;
                std::cerr << "derivative wrt input i " << d_i;
                std::cerr << "derivative wrt input g " << d_g;
            }
        }

        // g,i,f,o -> x, do it all in once
        in_diff->AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kNoTrans, w_gifo_x_, kNoTrans, 0.0);

        //// backward pass dropout
        //if (dropout_rate_ != 0.0) {
        //    in_diff->MulElements(dropout_mask_);
        //}
    
        // calculate delta
        const BaseFloat mmt = opts_.momentum;
    
        // weight x -> g, i, f, o
        w_gifo_x_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kTrans, 
                                      in                     , kNoTrans, mmt);
        // recurrent weight r -> g, i, f, o
        w_gifo_r_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kTrans, 
                                      YR.RowRange(0*S,T*S)   , kNoTrans, mmt);
        // bias of g, i, f, o
        bias_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1*S,T*S), mmt);
    
        // recurrent peephole c -> i
        peephole_i_c_corr_.AddDiagMatMat(1.0, DI.RowRange(1*S,T*S), kTrans, 
                                              YC.RowRange(0*S,T*S), kNoTrans, mmt);
        // recurrent peephole c -> f
        peephole_f_c_corr_.AddDiagMatMat(1.0, DF.RowRange(1*S,T*S), kTrans, 
                                              YC.RowRange(0*S,T*S), kNoTrans, mmt);
        // peephole c -> o
        peephole_o_c_corr_.AddDiagMatMat(1.0, DO.RowRange(1*S,T*S), kTrans, 
                                              YC.RowRange(1*S,T*S), kNoTrans, mmt);

        w_r_m_corr_.AddMatMat(1.0, DR.RowRange(1*S,T*S), kTrans, 
                                   YM.RowRange(1*S,T*S), kNoTrans, mmt);
    
        if (DEBUG) {
            std::cerr << "gradients(with optional momentum): \n";
            std::cerr << "w_gifo_x_corr_ " << w_gifo_x_corr_;
            std::cerr << "w_gifo_r_corr_ " << w_gifo_r_corr_;
            std::cerr << "bias_corr_ " << bias_corr_;
            std::cerr << "w_r_m_corr_ " << w_r_m_corr_;
            std::cerr << "peephole_i_c_corr_ " << peephole_i_c_corr_;
            std::cerr << "peephole_f_c_corr_ " << peephole_f_c_corr_;
            std::cerr << "peephole_o_c_corr_ " << peephole_o_c_corr_;
        }
    }

    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
        const BaseFloat lr  = opts_.learn_rate;

        w_gifo_x_.AddMat(-lr, w_gifo_x_corr_);
        w_gifo_r_.AddMat(-lr, w_gifo_r_corr_);
        bias_.AddVec(-lr, bias_corr_, 1.0);
    
        peephole_i_c_.AddVec(-lr, peephole_i_c_corr_, 1.0);
        peephole_f_c_.AddVec(-lr, peephole_f_c_corr_, 1.0);
        peephole_o_c_.AddVec(-lr, peephole_o_c_corr_, 1.0);
    
        w_r_m_.AddMat(-lr, w_r_m_corr_);

//        /* 
//          Here we deal with the famous "vanishing & exploding difficulties" in RNN learning.
//
//          *For gradients vanishing*
//            LSTM architecture introduces linear CEC as the "error bridge" across long time distance
//            solving vanishing problem.
//
//          *For gradients exploding*
//            LSTM is still vulnerable to gradients explosing in BPTT(with large weight & deep time expension).
//            To prevent this, we tried L2 regularization, which didn't work well
//
//          Our approach is a *modified* version of Max Norm Regularization:
//          For each nonlinear neuron, 
//            1. fan-in weights & bias model a seperation hyper-plane: W x + b = 0
//            2. squashing function models a differentiable nonlinear slope around this hyper-plane.
//
//          Conventional max norm regularization scale W to keep its L2 norm bounded,
//          As a modification, we scale down large (W & b) *simultaneously*, this:
//            1. keeps all fan-in weights small, prevents gradients from exploding during backward-pass.
//            2. keeps the location of the hyper-plane unchanged, so we don't wipe out already learned knowledge.
//            3. shrinks the "normal" of the hyper-plane, smooths the nonlinear slope, improves generalization.
//            4. makes the network *well-conditioned* (weights are constrained in a reasonible range).
//
//          We've observed faster convergence and performance gain by doing this.
//        */
//
        int DEBUG = 0;
        BaseFloat max_norm = 1.0;   // weights with large L2 norm may cause exploding in deep BPTT expensions
                                    // TODO: move this config to opts_
        CuMatrix<BaseFloat> L2_gifo_x(w_gifo_x_);
        CuMatrix<BaseFloat> L2_gifo_r(w_gifo_r_);
        L2_gifo_x.MulElements(w_gifo_x_);
        L2_gifo_r.MulElements(w_gifo_r_);

        CuVector<BaseFloat> L2_norm_gifo(L2_gifo_x.NumRows());
        L2_norm_gifo.AddColSumMat(1.0, L2_gifo_x, 0.0);
        L2_norm_gifo.AddColSumMat(1.0, L2_gifo_r, 1.0);
        L2_norm_gifo.Range(1*ncell_, ncell_).AddVecVec(1.0, peephole_i_c_, peephole_i_c_, 1.0);
        L2_norm_gifo.Range(2*ncell_, ncell_).AddVecVec(1.0, peephole_f_c_, peephole_f_c_, 1.0);
        L2_norm_gifo.Range(3*ncell_, ncell_).AddVecVec(1.0, peephole_o_c_, peephole_o_c_, 1.0);
        L2_norm_gifo.ApplyPow(0.5);

        CuVector<BaseFloat> shrink(L2_norm_gifo);
        shrink.Scale(1.0/max_norm);
        shrink.ApplyFloor(1.0);
        shrink.InvertElements();

        w_gifo_x_.MulRowsVec(shrink);
        w_gifo_r_.MulRowsVec(shrink);
        bias_.MulElements(shrink);

        peephole_i_c_.MulElements(shrink.Range(1*ncell_, ncell_));
        peephole_f_c_.MulElements(shrink.Range(2*ncell_, ncell_));
        peephole_o_c_.MulElements(shrink.Range(3*ncell_, ncell_));

        if (DEBUG) {
            if (shrink.Min() < 0.95) {   // we dont want too many trivial logs here
                std::cerr << "gifo shrinking coefs: " << shrink;
            }
        }
        
    }

private:
    // dims
    int32 ncell_;
    int32 nrecur_;  // recurrent projection layer dim
    int32 nstream_;

    CuMatrix<BaseFloat> prev_nnet_state_;

    // non-recurrent dropout 
    //BaseFloat dropout_rate_;
    //CuMatrix<BaseFloat> dropout_mask_;

    // feed-forward connections: from x to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_x_;
    CuMatrix<BaseFloat> w_gifo_x_corr_;

    // recurrent projection connections: from r to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_r_;
    CuMatrix<BaseFloat> w_gifo_r_corr_;

    // biases of [g, i, f, o]
    CuVector<BaseFloat> bias_;
    CuVector<BaseFloat> bias_corr_;

    // peephole from c to i, f, g 
    // peephole connections are block-internal, so we use vector form
    CuVector<BaseFloat> peephole_i_c_;
    CuVector<BaseFloat> peephole_f_c_;
    CuVector<BaseFloat> peephole_o_c_;

    CuVector<BaseFloat> peephole_i_c_corr_;
    CuVector<BaseFloat> peephole_f_c_corr_;
    CuVector<BaseFloat> peephole_o_c_corr_;

    // projection layer r: from m to r
    CuMatrix<BaseFloat> w_r_m_;
    CuMatrix<BaseFloat> w_r_m_corr_;

    // propagate buffer: output of [g, i, f, o, c, h, m, r]
    CuMatrix<BaseFloat> propagate_buf_;

    // back-propagate buffer: diff-input of [g, i, f, o, c, h, m, r]
    CuMatrix<BaseFloat> backpropagate_buf_;

};
} // namespace nnet1
} // namespace kaldi

#endif
