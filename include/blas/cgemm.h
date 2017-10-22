#ifndef __cgemm_h__
#define __cgemm_h__

#include"../../include/cuda/cuda_ctx.h"
#include"../../include/idc_string.h"
#include"../../include/idc_bitop.h"

void idc_cgemm_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int, int, int );
void idc_cgemv_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int, int, int );
void idc_flatcgemm_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int, int );
void idc_flatcgevv_create_kernel( cuda_kernel_t*, const cuda_context_t*, int, int, int, int, int );
void idc_cgemm( cuda_kernel_t*, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUstream );

#endif