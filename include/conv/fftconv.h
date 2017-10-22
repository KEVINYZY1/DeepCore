#ifndef __fftconv_h__
#define __fftconv_h__

#include"../../include/fft/fft.h"
#include"../../include/blas/cgemm.h"

typedef struct idc_fftconvOp{
    cuda_kernel_t kfft[3];
    cuda_kernel_t kcgemm;
    size_t        divpt[2];
} idc_fftconvOp_t;

size_t idc_fftconv_createOp( idc_fftconvOp_t*, const cuda_context_t*, uint32_t, int, int, int, int, int, int, int, int, int, int, int, int, int );
size_t idc_fftconv_createOp_grad( idc_fftconvOp_t*, const cuda_context_t*, int, int, int, int, int, int, int, int, int, int, int, int, int, int );
void   idc_fftconv( idc_fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
void   idc_fftconv_relu( idc_fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, float, CUstream );
void   idc_fftconv_grad( idc_fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

size_t idc_cellconv_createOp( idc_fftconvOp_t*, const cuda_context_t*, uint32_t, int, int, int, int, int, int, int, int, int, int, int, int, int );
size_t idc_cellconv_createOp_grad( idc_fftconvOp_t*, const cuda_context_t*, int, int, int, int, int, int, int, int, int, int, int, int, int, int );
void   idc_cellconv( idc_fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
void   idc_cellconv_relu( idc_fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, float, CUstream );
void   idc_cellconv_grad( idc_fftconvOp_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif
