#ifndef __cellconv_h__
#define __cellconv_h__

#include"../../include/blas/cgemm.h"
#include"../../include/fft/fft.h"

typedef struct cellconvOp{
	cuda_kernel_t kfft[3];
	cuda_kernel_t kcgemm;
	size_t        adivpt;
	size_t        bdivpt;
} cellconvOp_t;

size_t cellconv_createOp( cellconvOp_t*, const cuda_context_t*, unsigned int, int, int, int, int, int );
size_t cellconv_createOp_filter( cellconvOp_t*, const cuda_context_t*, unsigned int, int, int, int, int, int );
void   cellconv( cellconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, const float*, CUstream );
void   cellconv_filter( cellconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif
