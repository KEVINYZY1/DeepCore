#pragma warning( disable:4996 )
#include"../../include/cuda/cuda_ctx.h"
#include"../../include/idc_half.h"

static const unsigned int long long kbin_sm50[]=
{
#include"../../include/dev/common/kbin_sm50.h"
};
static const unsigned int long long kbin_sm52[]=
{
#include"../../include/dev/common/kbin_sm52.h"
};
static const unsigned int long long kbin_sm60[]=
{
#include"../../include/dev/common/kbin_sm60.h"
};
static const unsigned int long long kbin_sm61[]=
{
#include"../../include/dev/common/kbin_sm61.h"
};

static const unsigned int long long kbin_fftconv_sm50[]=
{
#include"../../include/dev/fftconv/kbin_sm50.h"
};
static const unsigned int long long kbin_fftconv_sm52[]=
{
#include"../../include/dev/fftconv/kbin_sm52.h"
};
static const unsigned int long long kbin_fftconv_sm60[]=
{
#include"../../include/dev/fftconv/kbin_sm60.h"
};
static const unsigned int long long kbin_fftconv_sm61[]=
{
#include"../../include/dev/fftconv/kbin_sm61.h"
};

static const unsigned int long long kbin_conv_sm50[]=
{
#include"../../include/dev/conv/kbin_sm50.h"
};
static const unsigned int long long kbin_conv_sm52[]=
{
#include"../../include/dev/conv/kbin_sm52.h"
};
static const unsigned int long long kbin_conv_sm60[]=
{
#include"../../include/dev/conv/kbin_sm60.h"
};
static const unsigned int long long kbin_conv_sm61[]=
{
#include"../../include/dev/conv/kbin_sm61.h"
};

static const unsigned int long long kbin_blas_sm50[]=
{
#include"../../include/dev/blas/kbin_sm50.h"
};
static const unsigned int long long kbin_blas_sm52[]=
{
#include"../../include/dev/blas/kbin_sm52.h"
};
static const unsigned int long long kbin_blas_sm60[]=
{
#include"../../include/dev/blas/kbin_sm60.h"
};
static const unsigned int long long kbin_blas_sm61[]=
{
#include"../../include/dev/blas/kbin_sm61.h"
};

static const unsigned int long long kbin_conv_fp16_sm50[]=
{
#include"../../include/dev/conv_fp16/kbin_sm50.h"
};
static const unsigned int long long kbin_conv_fp16_sm52[]=
{
#include"../../include/dev/conv_fp16/kbin_sm52.h"
};
static const unsigned int long long kbin_conv_fp16_sm60[]=
{
#include"../../include/dev/conv_fp16/kbin_sm60.h"
};
static const unsigned int long long kbin_conv_fp16_sm61[]=
{
#include"../../include/dev/conv_fp16/kbin_sm61.h"
};

static const unsigned int long long kbin_blas_fp16_sm50[]=
{
#include"../../include/dev/blas_fp16/kbin_sm50.h"
};
static const unsigned int long long kbin_blas_fp16_sm52[]=
{
#include"../../include/dev/blas_fp16/kbin_sm52.h"
};
static const unsigned int long long kbin_blas_fp16_sm60[]=
{
#include"../../include/dev/blas_fp16/kbin_sm60.h"
};
static const unsigned int long long kbin_blas_fp16_sm61[]=
{
#include"../../include/dev/blas_fp16/kbin_sm61.h"
};

static const unsigned int long long* p_devbin[][6]=
{
	{ kbin_sm50, kbin_fftconv_sm50, kbin_conv_sm50, kbin_blas_sm50, kbin_conv_fp16_sm50, kbin_blas_fp16_sm50 },
	{ kbin_sm52, kbin_fftconv_sm52, kbin_conv_sm52, kbin_blas_sm52, kbin_conv_fp16_sm52, kbin_blas_fp16_sm52 },
	{ kbin_sm60, kbin_fftconv_sm60, kbin_conv_sm60, kbin_blas_sm60, kbin_conv_fp16_sm60, kbin_blas_fp16_sm60 },
	{ kbin_sm61, kbin_fftconv_sm61, kbin_conv_sm61, kbin_blas_sm61, kbin_conv_fp16_sm61, kbin_blas_fp16_sm61 }
};

/****************************************************************************************************************************************************************
================================================================================================================================================================
/***************************************************************************************************************************************************************/

int cuda_context_create( cuda_context_t* p_ctx, char* p_temp )
{
    int i, n, p, q;	
    cuDriverGetVersion(&i);
    if(i<8000) 
        return idc_error_invalid_driver;
    cuCtxGetCurrent( &p_ctx->ctx );
    p_ctx->status=0;
    if(p_ctx!= NULL){ p_ctx->status=1; }
    cuDevicePrimaryCtxRetain( &p_ctx->ctx, p_ctx->dev );
    cuDevicePrimaryCtxSetFlags( p_ctx->dev, CU_CTX_SCHED_AUTO|CU_CTX_MAP_HOST|CU_CTX_LMEM_RESIZE_TO_MAX );
    cuCtxPushCurrent( p_ctx->ctx );
    switch(p_ctx->arch)
    {
    case 50: i=0; break;
    case 52: i=1; break;
    case 60: i=2; break;
    case 61: i=3; break;
    }    
    cuModuleLoadFatBinary( &p_ctx->module          , p_devbin[i][0] );
    cuModuleLoadFatBinary( &p_ctx->module_fftconv  , p_devbin[i][1] );
    cuModuleLoadFatBinary( &p_ctx->module_conv     , p_devbin[i][2] );
    cuModuleLoadFatBinary( &p_ctx->module_blas     , p_devbin[i][3] );
    cuModuleLoadFatBinary( &p_ctx->module_conv_fp16, p_devbin[i][4] );
    cuModuleLoadFatBinary( &p_ctx->module_blas_fp16, p_devbin[i][5] );
    n=sizeof(g_fftRF_ofs)/sizeof(g_fftRF_ofs[0])-1;
    if(cuMemAlloc(&p_ctx->d_RF[0], g_fftRF_ofs[n]*(sizeof(float2)+sizeof(int)) )!=CUDA_SUCCESS){
        cuModuleUnload( p_ctx->module           );
        cuModuleUnload( p_ctx->module_fftconv   );
        cuModuleUnload( p_ctx->module_conv      );
        cuModuleUnload( p_ctx->module_blas      );
        cuModuleUnload( p_ctx->module_conv_fp16 );
        cuModuleUnload( p_ctx->module_blas_fp16 );
        cuDevicePrimaryCtxRelease(p_ctx->dev);
        p_ctx->ctx=NULL;
        return idc_error_out_of_memory;
    }
    for( p=g_fftRF_ofs[(i=0)]; i<n; p=q ){
        q=g_fftRF_ofs[++i];
        idc_fft_calcRF( ((float2*)p_temp)+p, q-p, 1.0/(q-p) );
    }
    cuMemcpyHtoD( p_ctx->d_RF[0], p_temp, g_fftRF_ofs[n]*sizeof(float2) );
    if(p_ctx->arch==60){
        unsigned short* p_hRF=(unsigned short*)(p_temp+g_fftRF_ofs[n]*sizeof(float2));
        for( i=0; i<(n<<1); ++i ){
            p_hRF[i]=idc_float2half(((float*)p_temp)[i]);
        }
        p_ctx->d_RF[1]=p_ctx->d_RF[0]+g_fftRF_ofs[n]*sizeof(float2);
        cuMemcpyHtoD( p_ctx->d_RF[1], p_hRF, g_fftRF_ofs[n]*sizeof(int) );
    }
    cuDeviceGetAttribute( &p_ctx->n_sm                , CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT       , p_ctx->dev );
    cuDeviceGetAttribute( &p_ctx->cmemnb              , CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY      , p_ctx->dev );
    cuDeviceGetAttribute( &p_ctx->max_nbx             , CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X             , p_ctx->dev );
    cuDeviceGetAttribute( &p_ctx->max_nby             , CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y             , p_ctx->dev );
    cuDeviceGetAttribute( &p_ctx->max_block_size      , CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X            , p_ctx->dev );
    cuDeviceGetAttribute( &p_ctx->max_smemnb_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, p_ctx->dev );
    cuCtxPopCurrent(NULL);
    return idc_success;
}
int cuda_context_get_current( const cuda_context_t* p_ctx, int n_devices )
{
    CUcontext ctx;
    int i=n_devices-1;
    do{
        cuCtxGetCurrent( &ctx );
        if(p_ctx[i].ctx==ctx) break;
    }while((--i)>=0);
    return (i<n_devices?i:-1);
}
void cuda_context_release( cuda_context_t* p_ctx )
{
    if( p_ctx->ctx!=NULL ){
        cuda_context_bind( p_ctx );
        cuMemFree( p_ctx->d_RF[0] ); 
        cuModuleUnload( p_ctx->module           );
        cuModuleUnload( p_ctx->module_fftconv   );
        cuModuleUnload( p_ctx->module_conv      );
        cuModuleUnload( p_ctx->module_blas      );
        cuModuleUnload( p_ctx->module_conv_fp16 );
        cuModuleUnload( p_ctx->module_blas_fp16 );
        if( p_ctx->status==0 ){
            cuDevicePrimaryCtxRelease(p_ctx->dev);
        }
        p_ctx->ctx=NULL;
    }
}
