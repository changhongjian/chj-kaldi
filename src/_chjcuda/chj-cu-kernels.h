// cudamatrix/bd-cu-kernels.h

#ifndef KALDI_CHJ_CUDAMATRIX_CU_KERNELS_H_
#define KALDI_CHJ_CUDAMATRIX_CU_KERNELS_H_

#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "_chjcuda/chj-cu-kernels-ansi.h"
#include "cudamatrix/cu-matrixdim.h"
#include "base/kaldi-error.h"

/*
 * In this file are C++ templated wrappers 
 * of the ANSI-C CUDA kernels
 */

namespace chj {
namespace cuda{
using namespace kaldi;

inline int n_blocks(int size, int block_size) { 
	return size / block_size + ((size % block_size == 0)? 0 : 1); 
}
#define CU_SAFE_CALL(fun) \
{ \
  int32 ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "cudaError_t " << ret << " : \"" << cudaGetErrorString((cudaError_t)ret) << "\" returned from '" << #fun << "'"; \
  } \
  cudaThreadSynchronize(); \
} 


//A=B+C;
inline void mat_add_mat(MatrixDim A_dim,float *A,float *B,float *C,float NOT_USE_VALUE){
	int rows=A_dim.rows;	
	int cols=A_dim.cols;
	dim3 Bl(CU2DBLOCK, CU2DBLOCK);
	dim3 Gr(n_blocks(rows, CU2DBLOCK),n_blocks(cols, CU2DBLOCK));
	chj_cudaF_mat_add_mat(Gr,Bl,A_dim,A,B,C,NOT_USE_VALUE);
	CU_SAFE_CALL(cudaGetLastError());
}
inline void ctc_loss_fun(MatrixDim A_dim,float *A,MatrixDim B_dim,float *B,MatrixDim C_dim,float *C,float p,float NOT_USE_VALUE){
	int rows=A_dim.rows;
	int cols=A_dim.cols;
	dim3 Bl(CU2DBLOCK, CU2DBLOCK);
	dim3 Gr(n_blocks(rows, CU2DBLOCK),n_blocks(cols, CU2DBLOCK));
	chj_cudaF_ctc_loss_fun(Gr,Bl,A_dim,A,B_dim,B,C_dim,C,p,NOT_USE_VALUE);
	CU_SAFE_CALL(cudaGetLastError());
}

} //cuda
} // namespace chj

#endif // HAVE_CUDA

#endif
