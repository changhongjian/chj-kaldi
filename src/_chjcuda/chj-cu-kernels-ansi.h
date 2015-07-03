// cudamatrix/bd-cu-kernels-ansi.h

#ifndef KALDI_CHJ_CUDAMATRIX_CU_KERNELS_ANSI_H_
#define KALDI_CHJ_CUDAMATRIX_CU_KERNELS_ANSI_H_
#include "cudamatrix/cu-matrixdim.h"
#include<vector>

#if HAVE_CUDA == 1
extern "C" {

void chj_cudaF_mat_add_mat(dim3 Gr,dim3 Bl,MatrixDim A_dim,float *A,float *B,float *C,float NOT_USE_VALUE);
void chj_cudaF_ctc_loss_fun(dim3 Gr,dim3 Bl,MatrixDim A_dim,float *A,MatrixDim B_dim,float *B,MatrixDim C_dim,float *C,float p,float NOT_USE_VALUE);

} // extern "C" 

#endif // HAVE_CUDA

#endif
