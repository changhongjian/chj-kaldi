// cudamatrix/bd-cu-kernels.cu

// In this file is the CUDA code of the CUDA kernels, plus the ANSI-C wrappers

#include <cfloat>
#include "chj-cu-kernels-ansi.h"
#include <cmath>

//static is not support
template<typename Real>
__device__
inline Real LogAdd(Real x, Real y) {
  Real diff;
  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= -15.9424) {
    Real res;
    res = x + log1pf(expf(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

template<typename Real>
__global__
static void _chj_cuda_mat_add_mat(MatrixDim A_dim,Real * A,Real *B,Real *C,Real NOT_USE_VALUE){
	int i = blockIdx.x * blockDim.x + threadIdx.x; // row index
	int j = blockIdx.y * blockDim.y + threadIdx.y; // column index
	int index = i * A_dim.stride + j;
	if (i < A_dim.rows && j < A_dim.cols) {
		if(B[index] == NOT_USE_VALUE || C[index] == NOT_USE_VALUE ){
			A[index] = NOT_USE_VALUE;
		}else{
			A[index] = B[index] + C[index];
		}
	}
}//_chj_cuda_mat_add_mat

template<typename Real>
__global__
static void _chj_cuda_ctc_loss_fun(MatrixDim A_dim,Real * A,MatrixDim B_dim,Real *B,MatrixDim C_dim,Real *C,Real p,Real NOT_USE_VALUE){
	int i = blockIdx.x * blockDim.x + threadIdx.x; // row index
	int j = blockIdx.y * blockDim.y + threadIdx.y; // column index
	int index = i * A_dim.stride + j;
//B_dim not A_dim 这是因为我的程序要求
	if (i < B_dim.rows && j < A_dim.cols) {
		Real a=NOT_USE_VALUE;
		int n=C[j*C_dim.stride];
		for(int ii=1;ii<=n;ii++){
            int idb=i*B_dim.stride+ C[ j*C_dim.stride + ii]; //*** 
			if(B[idb]!=NOT_USE_VALUE){
				if(a==NOT_USE_VALUE)a=B[idb];
				else a=LogAdd(a,B[idb]);
			}
		}
		if(a!=NOT_USE_VALUE){
			A[index] -= expf(a-p); //7.29			
		}
	}
}//_chj_cuda_mat_add_mat

/***********************************************************************
 * ANSI-C wrappers of CUDA kernels
 */

void chj_cudaF_mat_add_mat(dim3 Gr,dim3 Bl,MatrixDim A_dim,float *A,float *B,float *C,float NOT_USE_VALUE){
	_chj_cuda_mat_add_mat<<<Gr,Bl>>>(A_dim,A,B,C,NOT_USE_VALUE);
}

void chj_cudaF_ctc_loss_fun(dim3 Gr,dim3 Bl,MatrixDim A_dim,float *A,MatrixDim B_dim,float *B,MatrixDim C_dim,float *C,float p,float NOT_USE_VALUE){
	_chj_cuda_ctc_loss_fun<<<Gr,Bl>>>(A_dim,A,B_dim,B,C_dim,C,p,NOT_USE_VALUE);
}



