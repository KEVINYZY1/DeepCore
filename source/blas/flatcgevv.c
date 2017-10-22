#include"../../include/blas/cgemm.h"

static void sflatcgevv_create_kernel( cuda_kernel_t* p_kernel, CUmodule module, int slice_size, int bat, int pnc, int qnc, int prc )
{
    static const char* knames[][3]={ 
        { "dk_sflatcgevv_16x32", "dk_sflatcgevv_16x64", "dk_sflatcgevv_32x32" },
        { "dk_xflatcgevv_16x32", "dk_xflatcgevv_16x64", "dk_xflatcgevv_32x32" },
    };
    int x, y, i, nx, ny;
    x=((pnc>16)&(pnc<=32))|(pnc>48);
    y=((qnc<=32)|((qnc>64)&(qnc<=96)))?0:(x==0);
    i=(x<<1)|y;
    nx=1<<(4+x);
    ny=1<<(5+y);
    cuda_context_create_kernel( p_kernel, module, knames[prc][i] );
    cuda_kernel_sao( p_kernel, AM_3P_5S );
    cuda_kernel_sbl( p_kernel, i>0?256:128, 1 );
    cuda_kernel_sgl( p_kernel, ((pnc+nx-1)/nx)*((qnc+ny-1)/ny), slice_size>>4, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f        );
    cuda_kernel_sep_i32( p_kernel, 4, slice_size );
    cuda_kernel_sep_i32( p_kernel, 5, bat        );
    cuda_kernel_sep_i32( p_kernel, 6, pnc        );
    cuda_kernel_sep_i32( p_kernel, 7, qnc        );
}
static void hflatcgevv_create_kernel( cuda_kernel_t* p_kernel, CUmodule module, int slice_size, int bat, int pnc, int qnc )
{
    static const char* knames[]={ "dk_hflatcgevv_16x32", "dk_hflatcgevv_16x64", "dk_hflatcgevv_32x32", "dk_hflatcgevv_32x64", "dk_hflatcgevv_64x32" };
    static const uint8_t block_size[]={ 63, 127, 127, 255, 255 };
    int x, y, i, nx, ny;
    if((pnc<=16)|((pnc>32)&(pnc<=48))){ x=0; } else
    if((pnc<=32)|((pnc>64)&(pnc<=96))){ x=1; } else { x=2; }
    y=((qnc>32)&(qnc<=64))|(qnc>96);
    i=(x<<1)|y;
    nx=1<<(4+x);
    ny=1<<(5+y);
    cuda_context_create_kernel( p_kernel, module, knames[i] );
    cuda_kernel_sao( p_kernel, AM_3P_5S );
    cuda_kernel_sbl( p_kernel, block_size[i]+1, 1 );
    cuda_kernel_sgl( p_kernel, ((pnc+nx-1)/nx)*((qnc+ny-1)/ny), slice_size>>4, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f        );
    cuda_kernel_sep_i32( p_kernel, 4, slice_size );
    cuda_kernel_sep_i32( p_kernel, 5, bat        );
    cuda_kernel_sep_i32( p_kernel, 6, pnc        );
    cuda_kernel_sep_i32( p_kernel, 7, qnc        );
}
void idc_flatcgevv_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int prc, int slice_size, int bat, int pnc, int qnc )
{
    CUmodule module=p_ctx->module_fftconv;
    if(prc!=1){
        sflatcgevv_create_kernel( p_kernel, module, slice_size, bat, pnc, qnc, prc!=0 );
    } else {
        hflatcgevv_create_kernel( p_kernel, module, slice_size, bat, pnc, qnc );
    }
}