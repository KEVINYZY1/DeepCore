#pragma warning( disable:4996 )
#include"../../include/cuda/cuda_ctx.h"

static const unsigned int long long kbin_sm35[]=
{
#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
#include"../../include/dev/kbin64_sm35.h"
#else
#include"../../include/dev/kbin32_sm35.h"
#endif
};
static const unsigned int long long kbin_sm37[]=
{
#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
#include"../../include/dev/kbin64_sm37.h"
#else
#include"../../include/dev/kbin32_sm37.h"
#endif
};
static const unsigned int long long kbin_sm50[]=
{
#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
#include"../../include/dev/kbin64_sm50.h"
#else
#include"../../include/dev/kbin32_sm50.h"
#endif
};
static const unsigned int long long kbin_sm52[]=
{
#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
#include"../../include/dev/kbin64_sm52.h"
#else
#include"../../include/dev/kbin32_sm52.h"
#endif
};
static const unsigned int long long kbin_sm53[]=
{
#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
#include"../../include/dev/kbin64_sm53.h"
#else
#include"../../include/dev/kbin32_sm53.h"
#endif
};
static const unsigned int long long kbin_sm60[]=
{
#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
#include"../../include/dev/kbin64_sm60.h"
#else
#include"../../include/dev/kbin32_sm60.h"
#endif
};
static const unsigned int long long kbin_sm61[]=
{
#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
#include"../../include/dev/kbin64_sm61.h"
#else
#include"../../include/dev/kbin32_sm61.h"
#endif
};
static const unsigned int long long kbin_sm62[]=
{
#if defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
#include"../../include/dev/kbin64_sm62.h"
#else
#include"../../include/dev/kbin32_sm62.h"
#endif
};
/****************************************************************************************************************************************************************
================================================================================================================================================================
/***************************************************************************************************************************************************************/

int cuda_context_create( cuda_context_t* p_ctx, char* p_temp )
{
	void* p_devbin;
	int i, n, p, q;	
	cuDriverGetVersion(&i);
	if(i<7000) 
		return ERROR_INVALID_DRIVER;
	cuDevicePrimaryCtxRetain( &p_ctx->ctx, p_ctx->dev );
	cuDevicePrimaryCtxSetFlags( p_ctx->dev, CU_CTX_SCHED_AUTO );
	cuCtxPushCurrent( p_ctx->ctx );
	switch(p_ctx->arch)
	{
	case 35: p_devbin=(void*)kbin_sm35; break;
	case 37: p_devbin=(void*)kbin_sm37; break;
	case 50: p_devbin=(void*)kbin_sm50; break;
	case 52: p_devbin=(void*)kbin_sm52; break;
	case 53: p_devbin=(void*)kbin_sm53; break;
	case 60: p_devbin=(void*)kbin_sm60; break;
	case 61: p_devbin=(void*)kbin_sm61; break;
	case 62: p_devbin=(void*)kbin_sm62; break;
	}
	if(cuModuleLoadFatBinary( &p_ctx->module, p_devbin )!=CUDA_SUCCESS){
		cuDevicePrimaryCtxRelease(p_ctx->dev);
		p_ctx->ctx=NULL;
		return ERROR_OUT_OF_MEMORY;
	}
	n=sizeof(g_fftRF_ofs)/sizeof(g_fftRF_ofs[0])-1;
	if(cuMemAlloc(&p_ctx->d_global, g_fftRF_ofs[n]*sizeof(float2) )!=CUDA_SUCCESS){
		cuModuleUnload(p_ctx->module);
		cuDevicePrimaryCtxRelease(p_ctx->dev);
		p_ctx->ctx=NULL;
		return ERROR_OUT_OF_MEMORY;
	}
	for( p=g_fftRF_ofs[(i=0)]; i<n; p=q ){
		q=g_fftRF_ofs[++i];
		fft_calcRF( ((float2*)p_temp)+p, q-p, (p?1.0:2.0)/(q-p) );
	}
	cuMemcpyHtoD( p_ctx->d_global, p_temp, g_fftRF_ofs[n]*sizeof(float2) );
	cuDeviceGetAttribute( &p_ctx->n_sm					, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT			, p_ctx->dev );
	cuDeviceGetAttribute( &p_ctx->align					, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT				, p_ctx->dev );
	cuDeviceGetAttribute( &p_ctx->max_nbx				, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X				, p_ctx->dev );
	cuDeviceGetAttribute( &p_ctx->max_nby				, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y				, p_ctx->dev );
	cuDeviceGetAttribute( &p_ctx->max_block_size		, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X				, p_ctx->dev );
	cuDeviceGetAttribute( &p_ctx->max_smemnb_per_block	, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK	, p_ctx->dev );
	cuCtxPopCurrent(NULL);
	return SUCCESS;
}
void cuda_context_release( cuda_context_t* p_ctx )
{
	if( p_ctx->ctx!=NULL ){
		cuda_context_bind(p_ctx);
		cuMemFree(p_ctx->d_global);
		cuModuleUnload(p_ctx->module);
		cuDevicePrimaryCtxRelease(p_ctx->dev);
		p_ctx->ctx=NULL;
	}
}