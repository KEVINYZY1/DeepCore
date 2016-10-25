#ifndef __fftconv_h__
#define __fftconv_h__

#include"../../include/fft/fft.h"
#include"../../include/blas/cgemm.h"
#include"../../include/blas/perm.h"

typedef struct fftconvOp{
	cuda_kernel_t* p_kernel;
	size_t         divpt[3];
	int            n_kernels;
} fftconvOp_t;

int  fftconv_createOp( fftconvOp_t*, size_t*, const cuda_context_t*, unsigned int, int, int, int, int, int );
int  fftconv_createOp_filter( fftconvOp_t*, size_t*, const cuda_context_t*, unsigned int, int, int, int, int, int );
void fftconv( fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, const float*, CUstream );
void fftconv_filter( fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
void fftconv_releaseOp( fftconvOp_t* );

#endif
