#include"../../include/blas/cgemm.h"

static void scgemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int prc, int bat, int anr, int bnr, int cnc, int lda, int ldb )
{
    uint32_t ix, iy, nx, ny, nbx, nby;
    static const char* s_tile_size[]={ "016", "032", "064", "128" };
    char kname[32]={"dk_"};
    if((anr<=16)|((anr>32)&(anr<=48))){
        ix=0;
        if((cnc<=32)|((cnc>64)&(cnc<=96))){ iy=0; } else 
        if(((cnc>32)&(cnc<=64))|((cnc>128)&(cnc<=192))){ iy=1; } else
        if(((cnc>96)&(cnc<=128))|((cnc>256)&(cnc<=384))){ iy=2; } else { iy=3; }
    } else
    if(((anr>16)&(anr<=32))|((anr>64)&(anr<=96))){
        ix=1;
        if((cnc<=32)|((cnc>64)&(cnc<=96))){ iy=0; } else 
        if(((cnc>32)&(cnc<=64))|((cnc>128)&(cnc<=192))){ iy=1; } else { iy=2; }
    } else
    if(((anr>48)&(anr<=64))|((anr>128)&(anr<=192))){
        ix=2;
        iy=((cnc>32)&(cnc<=64))|((cnc>128)&(cnc<=192));
    } else {
        ix=3; iy=0;
    }
    idc_strcat( kname, prc==0?"scgemm_":"xcgemm_" );
    idc_strcat( kname, s_tile_size[ix  ] );
    idc_strcat( kname, s_tile_size[iy+1] );
    nx=1<<(4+ix);
    ny=1<<(5+iy);
    nbx=(anr+nx-1)/nx;
    nby=(cnc+ny-1)/ny;
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P_7S );
    cuda_kernel_sbl( p_kernel, (ny*nx)>>4, 1 );
    cuda_kernel_sgl( p_kernel, nbx*nby, bat, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f );
    cuda_kernel_sep_i32( p_kernel, 4, nbx );
    cuda_kernel_sep_i32( p_kernel, 5, anr );
    cuda_kernel_sep_i32( p_kernel, 6, bnr );
    cuda_kernel_sep_i32( p_kernel, 7, cnc );
    cuda_kernel_sep_i32( p_kernel, 8, lda );
    cuda_kernel_sep_i32( p_kernel, 9, ldb );
}
static void hcgemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int layout, int bat, int anr, int bnr, int cnc, int lda, int ldb )
{
    int tile_ix, tile_iy, nx, ny, nbx, nby;
    static const char* s_layout[]={ "cgemmnn_", "cgemmnt_", "cgemmtn_" };
    static const char* s_tile_size[]={ "016", "032", "064", "128", "256" }; 
    static const uint8_t block_size[]={ 31, 31, 63, 127, 255 };
    char kname[32]={ "dk_h" };
    if((anr<=16)|((anr>32)&(anr<=48))){
        tile_ix=0;
        if((cnc<=32)|((cnc>64)&(cnc<=96))){ tile_iy=0; } else
        if(((cnc>32)&(cnc<=64))|((cnc>128)&(cnc<=192))){ tile_iy=1; } else
        if(((cnc>64)&(cnc<=128))|((cnc>256)&(cnc<=384))){ tile_iy=2; } else { tile_iy=3; }		
    } else
    if(((anr>16)&(anr<=32))|((anr>64)&(anr<=96))){
        tile_ix=1;
        if((cnc<=32)|((cnc>64)&(cnc<=96))){ tile_iy=0; } else
        if(((cnc>32)&(cnc<=64))|((cnc>128)&(cnc<=192))){ tile_iy=1; } else
        if(((cnc>64)&(cnc<=128))|((cnc>256)&(cnc<=384))){ tile_iy=2; } else { tile_iy=3; }
    } else
    if(((anr>32)&(anr<=64))|((anr>128)&(anr<=192))){
        tile_ix=2;
        if((cnc<=32)|((cnc>64)&(cnc<=96))){ tile_iy=0; } else
        if(((cnc>32)&(cnc<=64))|((cnc>128)&(cnc<=192))){ tile_iy=1; } else { tile_iy=2; }
    } else
    if((((anr>192)&(anr<=256))|(anr>384))&((cnc<=32)|((cnc>64)&(cnc<=96)))){
        tile_ix=4;
        tile_iy=0;
    } else {
        tile_ix=3;
        tile_iy=((cnc>32)&(cnc<=64))|(cnc>96);
    }
    idc_strcat( kname, s_layout[layout] );
    idc_strcat( kname, s_tile_size[tile_ix] );
    idc_strcat( kname, "x" );
    idc_strcat( kname, s_tile_size[1+tile_iy] );
    nx=1<<(4+tile_ix);
    ny=1<<(5+tile_iy);
    nbx=(anr+nx-1)/nx;
    nby=(cnc+ny-1)/ny;
    cuda_context_create_kernel( p_kernel, p_ctx->module_fftconv, kname );
    cuda_kernel_sao( p_kernel, AM_3P_7S );
    cuda_kernel_sbl( p_kernel, block_size[(tile_ix>0)+(1<<(5+tile_iy))]+1, 1 );
    cuda_kernel_sgl( p_kernel, nbx*nby, bat, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f );
    cuda_kernel_sep_i32( p_kernel, 4, nbx );
    cuda_kernel_sep_i32( p_kernel, 5, anr );
    cuda_kernel_sep_i32( p_kernel, 6, bnr );
    cuda_kernel_sep_i32( p_kernel, 7, cnc );
    cuda_kernel_sep_i32( p_kernel, 8, lda );
    cuda_kernel_sep_i32( p_kernel, 9, ldb );
}
void idc_cgemm_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int layout, int prc, int bat, int anr, int bnr, int cnc, int lda, int ldb )
{
    if(prc!=1){
        scgemm_create_kernel( p_kernel, p_ctx, prc!=0, bat, anr, bnr, cnc, lda, ldb );
    } else {
        hcgemm_create_kernel( p_kernel, p_ctx, layout, bat, anr, bnr, cnc, lda, ldb );
    }
}
void idc_cgemm( cuda_kernel_t* p_kernel, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUstream s )
{
    cuda_kernel_sep_ptr( p_kernel, 0, d_c );
    cuda_kernel_sep_ptr( p_kernel, 1, d_a );
    cuda_kernel_sep_ptr( p_kernel, 2, d_b );
    cuda_kernel_launch( p_kernel, s );
}