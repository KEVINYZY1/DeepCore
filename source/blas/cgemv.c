#include"../../include/blas/cgemm.h"

static void scgemv_create_kernel( cuda_kernel_t* p_kernel, CUmodule module, int prc, int bat, int nr, int nc, int lda, int ldb, int ldc )
{
    cuda_context_create_kernel( p_kernel, module, prc==0?"dk_scgemv":"dk_xcgemv" );
    cuda_kernel_sao( p_kernel, AM_3P_6S );
    cuda_kernel_sgl( p_kernel, (nr+127)>>7, bat, 1 );
    cuda_kernel_sbl( p_kernel, 128, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f );
    cuda_kernel_sep_i32( p_kernel, 4, nr  );
    cuda_kernel_sep_i32( p_kernel, 5, nc  );
    cuda_kernel_sep_i32( p_kernel, 6, lda );
    cuda_kernel_sep_i32( p_kernel, 7, ldb );
    cuda_kernel_sep_i32( p_kernel, 8, ldc );
}
static void hcgemv_create_kernel( cuda_kernel_t* p_kernel, CUmodule module, int dir, int bat, int nr, int nc, int lda, int ldb, int ldc )
{
    cuda_context_create_kernel( p_kernel, module, dir==0?"dk_hcgemvf":"dk_hcgemvb" );
    cuda_kernel_sao( p_kernel, AM_3P_6S );
    cuda_kernel_sgl( p_kernel, (nr+127)>>7, bat, 1 );
    cuda_kernel_sbl( p_kernel, 64, 1 );
    cuda_kernel_sep_f32( p_kernel, 3, 1.f );
    cuda_kernel_sep_i32( p_kernel, 4, nr  );
    cuda_kernel_sep_i32( p_kernel, 5, nc  );
    cuda_kernel_sep_i32( p_kernel, 6, lda );
    cuda_kernel_sep_i32( p_kernel, 7, ldb );
    cuda_kernel_sep_i32( p_kernel, 8, ldc );
}
void idc_cgemv_create_kernel( cuda_kernel_t* p_kernel, const cuda_context_t* p_ctx, int dir, int prc, int bat, int nr, int nc, int lda, int ldb, int ldc )
{
    CUmodule module=p_ctx->module_fftconv;
    if(prc!=1){
        scgemv_create_kernel( p_kernel, module, prc, bat, nr, nc, lda, ldb, ldc );
    } else {
        hcgemv_create_kernel( p_kernel, module, dir, bat, nr, nc, lda, ldb, ldc );
    }
}