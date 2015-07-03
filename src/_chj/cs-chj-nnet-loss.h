// nnet/nnet-loss.h

// Copyright 2011  Brno University of Technology (author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_CHJ_NNET_LOSS_H_
#define KALDI_CHJ_NNET_LOSS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"

namespace kaldi {
namespace nnet1 {

class CTC{// 基本版的
public:
	CTC():frames_(0),loss_(0),sentences_(0){}
	~CTC(){}
	//一次总的执行未做优化
	// const Vector<BaseFloat> 
	void Run(const CuMatrix<BaseFloat> &net_out, std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff,int32 shrinkT=-1);
    /* 多线程加自写CUDA的函数 */
    void RunSpeedUp(const CuMatrix<BaseFloat> &net_out, std::vector<int32> &target,
            CuMatrix<BaseFloat> *diff,int32 shrinkT=-1);
    
	std::string Report();
private:
	int32 sentences_;
	int32 frames_;
	BaseFloat loss_;

};//CTC_base


} // namespace nnet1
} // namespace kaldi

#endif

