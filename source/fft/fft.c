#pragma warning( disable:4996 )
#include"../../include/fft/fft.h"
#include"../../include/idc_string.h"

void idc_create_fft_kernel_r2c( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc, int is_split, int is_ext, int is_pad, int is_flip )
{
    static const unsigned short block_size[]={ 128, 128, 256, 64, 512 };
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_size[]={ "8x8", "16x16", "32x32", "64x64", "128x128" };
    static uint32_t argmask[][2]={ { AM_3P_4S, AM_3P_6S }, { AM_3P_3S, AM_3P_5S }, { AM_3P_9S, AM_3P_BS } };
    char kname[64]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, s_size[axis] );
    idc_strcat( kname, "_r2c" );
    if(is_split){ idc_strcat( kname, "_split" ); }
    if(is_ext ){ if(is_split==0){ idc_strcat( kname, "_ext" ); } } else
    if(is_pad ){ idc_strcat( kname, "_pad"  ); } else
    if(is_flip){ idc_strcat( kname, "_flip" ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, argmask[is_split?2:(axis>2)][is_pad] );
    cuda_kernel_sbl( p_kernel, block_size[axis], 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_fft_kernel_r2c_opt( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc, int is_flip, int i )
{
    static const unsigned char block_size[]={ 127, 255, 63 };
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_size[]={ "16x16", "32x32", "64x64" };
    static const char* s_fils[]={ "_s3", "_s5", "_s7" };
    char kname[32]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, s_size[axis-1] );
    idc_strcat( kname, "_r2c" );
    if(is_flip){ idc_strcat( kname, "_flip" ); }
    idc_strcat( kname, s_fils[i] );
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P );
    cuda_kernel_sbl( p_kernel, block_size[axis-1]+1, 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_fft_kernel_c2r( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc, int dir, int is_split, int fuse, int relu )
{
    static const unsigned short block_size[]={ 128, 128, 256, 64, 512 };
    static const uint32_t argmask[][4]={ { AM_3P_5S, AM_3P_6S, AM_4P_5S, AM_4P_6S }, { AM_3P_4S, AM_3P_5S, AM_4P_4S, AM_4P_5S }, { AM_3P_AS, AM_3P_BS, AM_4P_AS, AM_4P_BS } };
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_size[]={ "8x8", "16x16", "32x32", "64x64", "128x128" };
    char kname[64]={ "dk_" };
    int b=prc&1;
    int iargmask_x=(fuse<<1)|relu;
    int iargmask_y=is_split?2:(axis>2);
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, s_size[axis] );
    idc_strcat( kname, "_c2r" );
    if(is_split){ idc_strcat( kname, "_splice" ); }
    if(fuse){ idc_strcat( kname, dir==0?"_bias":"_xdiff" ); }
    if(relu){ idc_strcat( kname, dir==0?"_relu":"_drelu" ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, argmask[iargmask_y][iargmask_x] );
    cuda_kernel_sbl( p_kernel, block_size[axis], 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_fft_kernel_c2r_grad( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc )
{
    static const unsigned short block_size[]={ 128, 128, 256, 64, 512 };
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_size[]={ "8x8", "16x16", "32x32", "64x64", "128x128" };
    char kname[32]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, s_size[axis] );
    idc_strcat( kname, "_c2r_grad" );
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P_3S );
    cuda_kernel_sbl( p_kernel, block_size[axis], 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_fft_kernel_c2r_grad_opt( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc, int i )
{
    static const unsigned char block_size[]={ 127, 255, 63 };
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_size[]={ "16x16", "32x32", "64x64" };
    char kname[32]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, s_size[axis-1] );
    idc_strcat( kname, i==0?"_c2r_grad_s3":"_c2r_grad_s5" );
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P_1S );
    cuda_kernel_sbl( p_kernel, block_size[axis-1]+1, 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_cellfft_kernel_r2c( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc, int perm_id, int is_ext, int is_pad, int is_flip )
{
    static const unsigned short block_size[]={ 128, 256, 512 };
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_size[]={ "8x8", "16x16", "32x32" };
    static uint32_t argmask[][2]={ { AM_3P_4S, AM_3P_6S }, { AM_3P_5S, AM_3P_6S }, { AM_3P_AS, AM_3P_CS } };
    char kname[64]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, s_size[axis] );
    idc_strcat( kname, "_r2c" );
    if(perm_id==0){ idc_strcat( kname, "_perm2d" ); } else
    if(perm_id==1){ idc_strcat( kname, "_perm3d" ); } else { idc_strcat( kname, "_split_perm" ); }
    if(is_ext ){ if(perm_id!=2){ idc_strcat( kname, "_ext"  ); } } else
    if(is_pad ){ idc_strcat( kname, "_pad"  ); } else
    if(is_flip){ idc_strcat( kname, "_flip" ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, argmask[perm_id][is_pad] );
    cuda_kernel_sbl( p_kernel, block_size[axis], 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_cellfft_kernel_r2c_opt( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc, int is_flip, int i )
{
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_fils[]={ "_s3", "_s5", "_s7" };
    char kname[32]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, axis==1?"16x16":"32x32" );
    idc_strcat( kname, "_r2c_perm" );
    if(is_flip){ idc_strcat( kname, "_flip" ); }
    idc_strcat( kname, s_fils[i] );
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P_2S );
    cuda_kernel_sbl( p_kernel, axis==1?256:512, 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_cellfft_kernel_c2r( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc, int dir, int perm_id, int fuse, int relu )
{
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_size[]={ "8x8", "16x16", "32x32" };
    static const uint32_t argmask[][4]={ { AM_3P_5S, AM_3P_6S, AM_4P_5S, AM_4P_6S }, { AM_3P_AS, AM_3P_BS, AM_4P_AS, AM_4P_BS} };
    char kname[64]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, s_size[axis] );
    idc_strcat( kname, "_c2r" );
    if(perm_id==0){ idc_strcat( kname, "_perm2d" ); } else
    if(perm_id==1){ idc_strcat( kname, "_perm3d" ); } else { idc_strcat( kname, "_splice_perm" ); }
    if(fuse){ idc_strcat( kname, dir==0?"_bias":"_xdiff" ); }
    if(relu){ idc_strcat( kname, dir==0?"_relu":"_drelu" ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, argmask[perm_id==2][(fuse<<1)|relu] );
    cuda_kernel_sbl( p_kernel, axis==0?128:256, 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_cellfft_kernel_c2r_grad( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc )
{
    static const char* s_prec[]={ "s", "h", "x" };
    static const char* s_size[]={ "8x8", "16x16", "32x32" };
    char kname[32]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, s_size[axis] );
    idc_strcat( kname, "_c2r_grad_perm" );
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P_5S );
    cuda_kernel_sbl( p_kernel, axis==0?128:256, 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}
void idc_create_cellfft_kernel_c2r_grad_opt( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int axis, int prc, int i )
{
    static const char* s_prec[]={ "s", "h", "x" };
    char kname[32]={ "dk_" };
    int b=prc&1;
    idc_strcat( kname, s_prec[prc] );
    idc_strcat( kname, "fft" );
    idc_strcat( kname, axis==1?"16x16":"32x32" );
    idc_strcat( kname, "_c2r_grad_perm" );
    idc_strcat( kname, i==0?"_s3":"_s5" );
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P_3S );
    cuda_kernel_sbl( p_kernel, 256, 1 );
    cuda_kernel_sep_ptr( p_kernel, 2, p_ctx->d_RF[b]+g_fftRF_ofs[axis]*(b?4:8) );
}