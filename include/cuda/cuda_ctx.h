#ifndef __cuda_ctx_h__
#define __cuda_ctx_h__

#include"../idc_status.h"
#include"../../include/fft/fft_rf.h"
#include"cuda_kernel.h"

static const uint16_t g_fftRF_ofs[]={0,4,12,28,60,124};

typedef struct cuda_context{
    CUcontext   ctx;
	CUmodule    module;
    CUmodule    module_fftconv;
    CUmodule    module_conv;
	CUmodule    module_blas;
    CUmodule    module_conv_fp16;
	CUmodule    module_blas_fp16;
    CUdeviceptr d_RF[2];
    CUdevice    dev;
    int         status;
    int         arch;
    int         n_sm;
    int         cmemnb;
    int         max_nbx;
    int         max_nby;
    int         max_block_size;
    int         max_smemnb_per_block;
} cuda_context_t;

int                cuda_context_create( cuda_context_t*, char* );
int                cuda_context_get_current( const cuda_context_t*, int );
void               cuda_context_release( cuda_context_t* );
__forceinline void cuda_context_bind( const cuda_context_t* p ){ cuCtxSetCurrent(p->ctx); }
__forceinline void cuda_context_unbind(){ cuCtxPopCurrent(NULL); }
__forceinline void cuda_context_create_kernel( cuda_kernel_t* p_kernel, CUmodule module, const char* p_name )
{
    cuModuleGetFunction( &p_kernel->id, module, p_name );
    p_kernel->smemnb=0;
    p_kernel->extra[0]=(void*)CU_LAUNCH_PARAM_BUFFER_POINTER;
    p_kernel->extra[1]=(void*)p_kernel->args;
    p_kernel->extra[2]=(void*)CU_LAUNCH_PARAM_BUFFER_SIZE;
    p_kernel->extra[3]=(void*)&p_kernel->arg_size;
    p_kernel->extra[4]=(void*)CU_LAUNCH_PARAM_END;
}

#endif
