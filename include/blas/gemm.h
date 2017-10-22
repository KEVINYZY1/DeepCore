#ifndef __gemm_h__
#define __gemm_h__

#include"../idc_string.h"
#include"../../include/cuda/cuda_ctx.h"

void idc_gemm_create_kernel( cuda_kernel_t*, const cuda_context_t*, uint32_t, int, int, int, int, int, int, int, int );
void idc_gemm( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );
void idc_gemm_relu( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, float, CUstream );
void idc_gemm_grad( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, float, CUstream );

#endif