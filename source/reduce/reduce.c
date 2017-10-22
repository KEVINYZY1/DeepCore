#include"../../include/reduce/reduce.h"
#include"../../include/idc_string.h"

int idc_reduce_createOp( idc_reduceOp_t* Op, cuda_context_t* p_ctx, uint32_t mask, int size, int pitch, int batch )
{
    static const char* s_radix[]={ "_b32", "_b64", "_b128", "_b256" };
    static const char* s_op[]={ "add", "max", "min", "dpxx", "dpxy" };
    char kname[32]={ "dk_" };
    cuda_kernel_t* p_kernel=&Op->kernel;
    int prc=(mask>>1)&0x3;
    int op =(mask>>3)&0x7;
    int i=(size>=512)+(size>=4096)+(size>=8192)+(size>=65536*16);
    int block_size=1<<(5+(i<3?i:3));
    int n=size>>16;
    int gdx=i<4?batch:(n<256?n:256);
    idc_strcat( kname, prc==0?"s":(prc==1?"h":"x") );
    idc_strcat( kname, "reduce_" );
    idc_strcat( kname, s_op[op] );
    if(i<4){ idc_strcat( kname, s_radix[i] ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module, kname );
    cuda_kernel_sao( p_kernel, i<4?AM_3P_3S:AM_5P_3S );
    cuda_kernel_sbl( p_kernel, block_size, 1 );
    cuda_kernel_sgl( p_kernel, gdx, i<4?1:batch, 1 );
    cuda_kernel_sep_i32( p_kernel, i<4?4:6, size  );
    cuda_kernel_sep_i32( p_kernel, i<4?5:7, pitch );
    Op->d_temp=0;
    if(i==4){
        if( cuMemAlloc( &Op->d_temp, batch*(gdx+1)*4 )!=CUDA_SUCCESS )
            return idc_error_out_of_device_memory;	
        cuda_kernel_sep_ptr( p_kernel, 3, Op->d_temp+batch*gdx*4 );
        cuda_kernel_sep_ptr( p_kernel, 4, Op->d_temp );
    }
    return idc_success;
}
void idc_reduce_launch( idc_reduceOp_t* Op, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, float scale, CUstream s )
{
    cuda_kernel_t* p_kernel=&Op->kernel;
    cuda_kernel_sep_ptr( p_kernel, 0, d_c );
    cuda_kernel_sep_ptr( p_kernel, 1, d_a );
    cuda_kernel_sep_ptr( p_kernel, 2, d_b );
    cuda_kernel_sep_f32( p_kernel, Op->d_temp!=0?5:3, scale );
    cuda_kernel_launch( p_kernel, s );
}
void idc_reduce_releaseOp( idc_reduceOp_t* Op )
{
    if(Op->d_temp){ cuMemFree(Op->d_temp); Op->d_temp=0; }
}