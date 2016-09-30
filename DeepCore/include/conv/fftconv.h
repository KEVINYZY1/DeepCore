#ifndef __fftconv_h__
#define __fftconv_h__

#include"../../include/fft/fft.h"
#include"../../include/blas/cgemm.h"
#include"../../include/blas/perm.h"

typedef struct fftconvOp{
	cuda_kernel_t kfft[3];
	cuda_kernel_t kperm[3];
	cuda_kernel_t kcgemm;
	size_t        divpt[3];
} fftconvOp_t;

size_t fftconv_createOp( fftconvOp_t*, const cuda_context_t*, unsigned int, int, int, int, int, int );
size_t fftconv_createOp_filter( fftconvOp_t*, const cuda_context_t*, unsigned int, int, int, int, int, int );
void   fftconv( fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, float*, CUstream );
void   fftconv_filter( fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif
