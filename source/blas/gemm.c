#include"../../include/blas/gemm.h"

void sgemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, uint32_t mask, int mode, int ng, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
    int prc, is_forward, axis, axis_x, axis_y, bias, relu, tile_x, tile_y, o;
    static const char* s_tile_shape[]={ "128x032", "128x064", "128x128" };
    static const uint32_t argmask[]={AM_3P_7S,AM_3P_8S,AM_4P_7S,AM_4P_8S};
    static const unsigned char block_size[]={127,127,255};
    char kname[64];
    prc=(mask>>1)&0x3;
    axis_x=0;
    axis_y=(cnc>32)+(cnc>64);
    axis=(axis_x<<2)|axis_y;
    tile_x=1<<(axis_x+7);
    tile_y=1<<(axis_y+5);
    is_forward=mode==0;
    bias=((mask>>3)&0x1)&is_forward;
    relu=(mask>>24)&is_forward;
    o=4+bias+relu;
    idc_strcat( kname, "dk_" );
    idc_strcat( kname, prc==0?"s":"x" );
    idc_strcat( kname, mode==0?"gemmnn_":(mode==1?"gemmnt_":"gemmtn_") );
    idc_strcat( kname, s_tile_shape[axis] );
    if(bias){ idc_strcat( kname, "_bias" ); }
    if(relu){ idc_strcat( kname, "_relu" ); }
    if(ng>1){ idc_strcat( kname, "_gp" ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module_blas, kname );
    cuda_kernel_sao( p_kernel, argmask[(bias<<1)|relu] );
    cuda_kernel_sbl( p_kernel, block_size[axis]+1, 1 );
    if(axis_y>=2){
	cuda_kernel_sgl( p_kernel, (anr+tile_x-1)/tile_x, (cnc+tile_y-1)/tile_y, ng );
    } else {
        cuda_kernel_sgl( p_kernel, (anr+tile_x-1)/tile_x, ng, 1 ); 
    }
    cuda_kernel_sep_i32( p_kernel, o++, anr );
    cuda_kernel_sep_i32( p_kernel, o++, bnr );
    cuda_kernel_sep_i32( p_kernel, o++, cnc );
    cuda_kernel_sep_i32( p_kernel, o++, lda );
    cuda_kernel_sep_i32( p_kernel, o++, ldb );
    cuda_kernel_sep_i32( p_kernel, o++, ldc );
}
void hgemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, uint32_t mask, int mode, int ng, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
    int axis_x, axis_y, axis, bias, relu, tile_x, tile_y, o;
    static const char* s_tile_shape[]={ "128x128", "128x256", "256x032", "256x064", "256x128" };
    static const uint32_t argmask[]={ AM_3P_7S, AM_3P_8S, AM_4P_7S, AM_4P_8S };
    static const unsigned char block_size[]={ 127, 255, 127, 127, 255 };
    char kname[64];
    axis_x=((anr<=128)|((anr>256)&(anr<=384)))?0:1;
    axis_y=axis_x>0?(((cnc>128)&(cnc<=256))|(cnc>384)):((cnc>32)+(cnc>64));
    axis=(axis_x<<1)+axis_y;
    tile_x=1<<(axis_x+7);
    tile_y=1<<(axis_y+(axis_x>0?5:7));
    bias=(mask>>3)&0x1;
    relu=mask>>24;
    o=4+bias+relu;
    idc_strcat( kname, "dk_h" );
    idc_strcat( kname, mode==0?"gemmnn_":"gemmnt_" );
    idc_strcat( kname, s_tile_shape[axis] );
    if(bias){ idc_strcat( kname, "_bias" ); }
    if(relu){ idc_strcat( kname, "_relu" ); }
    if(ng>1){ idc_strcat( kname, "_gp" ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module_blas_fp16, kname );
    cuda_kernel_sao( p_kernel, argmask[(bias<<1)|relu] );
    cuda_kernel_sbl( p_kernel, block_size[axis]+1, 1 );
    if(axis_y>=2){
        cuda_kernel_sgl( p_kernel, (anr+tile_x-1)/tile_x, (cnc+tile_y-1)/tile_y, ng );
    } else {
        cuda_kernel_sgl( p_kernel, (anr+tile_x-1)/tile_x, ng, 1 );
    }
    cuda_kernel_sep_i32( p_kernel, o++, anr );
    cuda_kernel_sep_i32( p_kernel, o++, bnr );
    cuda_kernel_sep_i32( p_kernel, o++, cnc );
    cuda_kernel_sep_i32( p_kernel, o++, lda );
    cuda_kernel_sep_i32( p_kernel, o++, ldb );
    cuda_kernel_sep_i32( p_kernel, o++, ldc );
}
void idc_gemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, uint32_t mask, int mode, int ng, int anr, int bnr, int cnc, int lda, int ldb, int ldc )
{
    if((mask&0x6)!=2){
	sgemm_create_kernel( p_kernel, p_ctx, mask, mode, ng, anr, bnr, cnc, lda, ldb, ldc );
    } else {
	hgemm_create_kernel( p_kernel, p_ctx, mask, mode, ng, anr, bnr, cnc, lda, ldb, ldc );
    }
}
void idc_gemm( cuda_kernel_t* p_kernel, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_bias, float alpha, CUstream s )
{
    cuda_kernel_sep_ptr( p_kernel, 0, d_c );
    cuda_kernel_sep_ptr( p_kernel, 1, d_a );
    cuda_kernel_sep_ptr( p_kernel, 2, d_b );
    if(d_bias!=0){
        cuda_kernel_sep_ptr( p_kernel, 3, d_bias );
    }
    cuda_kernel_sep_f32( p_kernel, 3+(d_bias!=0), alpha );
    cuda_kernel_launch( p_kernel, s );
}
void idc_gemm_relu( cuda_kernel_t* p_kernel, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_bias, float alpha, float slope, CUstream s )
{
    cuda_kernel_sep_ptr( p_kernel, 0, d_c );
    cuda_kernel_sep_ptr( p_kernel, 1, d_a );
    cuda_kernel_sep_ptr( p_kernel, 2, d_b );
    if(d_bias!=0){
        cuda_kernel_sep_ptr( p_kernel, 3, d_bias );
    }
    cuda_kernel_sep_f32( p_kernel, 3+(d_bias!=0), alpha );
    cuda_kernel_sep_f32( p_kernel, 4+(d_bias!=0), slope );
    cuda_kernel_launch( p_kernel, s );
}
void idc_gemm_grad( cuda_kernel_t* p_kernel, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, float ratio, CUstream s )
{
    cuda_kernel_sep_ptr( p_kernel, 0, d_c   );
    cuda_kernel_sep_ptr( p_kernel, 1, d_a   );
    cuda_kernel_sep_ptr( p_kernel, 2, d_b   );
    cuda_kernel_sep_f32( p_kernel, 3, ratio );
    cuda_kernel_launch( p_kernel, s );
}
