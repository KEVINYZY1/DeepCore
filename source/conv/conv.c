#include"../../include/conv/conv.h"

static void generate_slider_indices( uint32_t* p_indices, int prc, int lda, int ldb, int anr, int bnr, int dx, int fx, int fy, int pnc, int qnc, int su, int sv, int ng )
{
    int b, enb, nc, u, v, c, i, n;
	b=prc&0x1; 
	nc=(pnc+b)>>b;
	enb=prc<2?4:2;
    for( c=0; c<nc; ++c ){
        for( v=0; v<fy; ++v ){
            for( u=0; u<fx; ++u ){
                *p_indices=(c*lda+(v*sv*dx+u*su))*enb; ++p_indices;
            }
        }
    }	
    if((bnr&7)!=0){
        for( i=(ng*nc*lda-(lda>anr))*enb, n=ldb-bnr; n>0; --n ){ *p_indices=i; ++p_indices; }
    }
    if(b!=0)
    {
        int a=fx*fy*enb;
        for( c=0; c<nc; ++c ){
			uint32_t idx=2*c*a;
            for( i=0; i<a; ++i ){
                *p_indices=idx+i;   ++p_indices;
                *p_indices=idx+i+a; ++p_indices;
            }
        }
        if((bnr&7)!=0){
            for( i=(ldb*qnc-1)*enb, n=ldb-bnr; n>0; --n ){ *p_indices=i; ++p_indices; }
        }
    }
}
static void sconv_create_kernel( idc_convOp_t* Op, uint32_t* p_temp, const cuda_context_t* p_ctx, int inc, int onc, int bnr, int cnr, int bias, int relu, int non_cached, int ng, int prc )
{
    int ix, iy, sx, sy, tile_y;
    static const uint32_t argmask[]={ AM_3P_AS, AM_3P_BS, AM_4P_AS, AM_4P_BS, AM_5P_AS, AM_5P_BS };
    static const char* s_size[]={ "032x256", "largep5", "064x064", "064x128", "064x256", "largep6", "128x032", "128x064", "128x128", "largep7" };
    static const uint8_t block_size[]={ 255, 63, 127, 255, 127, 127, 255 };
	static const uint8_t ofs[]={ 0, 1, 4 };
    char kname[64]={ "dk_" };
    cuda_kernel_t* p_kernel=&Op->kernel;
    ix=(cnr<=32)?0:(((cnr<=64)|((cnr>128)&(cnr<=192)))?1:2);
	if(ix==0){ iy=8; } else
	if(ix==1){ iy=(onc>64)+(((onc>128)&(onc<=256))|(onc>384)); } else { iy=(onc>32)+(onc>64); }
    sx=5+ix;
    sy=(ix==0)?8:((ix==1?6:5)+iy);
    tile_y=1<<sy;
    idc_strcat( kname, prc==0?"sconv_":"xconv_" );
    idc_strcat( kname, s_size[ofs[ix]+iy+non_cached] );
    if(bias){ idc_strcat( kname, "_bias" ); }
    if(relu){ idc_strcat( kname, "_relu" ); }
	if((ng>1)&(non_cached==0)){ idc_strcat( kname, "_gp" ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module_conv, kname );
    cuda_kernel_sao( p_kernel, argmask[(non_cached<<2)|(bias<<1)|relu] );
    cuda_kernel_sbl( p_kernel, block_size[ofs[ix]+iy]+1, 1 );
	if((ix==0)|((ix>0)&(ix>=iy))){
        cuda_kernel_sgl( p_kernel, (cnr+(1<<sx)-1)>>sx, (onc+tile_y-1)>>sy, ng );
	} else {
		cuda_kernel_sgl( p_kernel, (cnr+(1<<sx)-1)>>sx, ng, 1 );
	}
}
static void hconv_create_kernel( idc_convOp_t* Op, uint32_t* p_temp, const cuda_context_t* p_ctx, int inc, int onc, int bnr, int cnr, int bias, int relu, int non_cached, int ng )
{
    int ix, iy, tile_x, tile_y;
    static const uint32_t argmask[]={ AM_3P_AS, AM_3P_BS, AM_4P_AS, AM_4P_BS, AM_5P_AS, AM_5P_BS };
    static const char* s_size[]={ "128x128", "128x256", "largep7", "256x032", "256x064", "256x128", "largep8" };
    static const uint8_t block_size[]={ 127, 255, 127, 127, 255 };
    static const uint8_t ofs[]={ 0, 2 };
    char kname[64]={ "dk_hconv_" };
    cuda_kernel_t* p_kernel=&Op->kernel;
	ix=((cnr>128)&(cnr<=256))|(cnr>384);
	if(ix==0){ iy=((onc>128)&(onc<=256))|(onc>384); } else { iy=(onc>32)+(onc>64); }
    tile_x=1<<(7+ix);
    tile_y=1<<((ix==0?7:5)+iy);
    idc_strcat( kname, s_size[ofs[ix]+iy+non_cached] );
    if(bias){ idc_strcat( kname, "_bias" ); }
    if(relu){ idc_strcat( kname, "_relu" ); }
	if((ng>1)&(non_cached==0)){ idc_strcat( kname, "_gp" ); }
    cuda_context_create_kernel( p_kernel, p_ctx->module_conv_fp16, kname );
    cuda_kernel_sao( p_kernel, argmask[(non_cached<<2)|(bias<<1)|relu] );
    cuda_kernel_sbl( p_kernel, block_size[ofs[ix]+iy], 1 );
	if((ix==0)|((ix>0)&(iy>1))){
		cuda_kernel_sgl( p_kernel, (cnr+tile_x-1)/tile_x, (onc+tile_y-1)/tile_y, ng );
	} else {
		cuda_kernel_sgl( p_kernel, (cnr+tile_x-1)/tile_x, ng, 1 );
	}
}

int idc_conv_createOp( idc_convOp_t* Op, uint32_t* p_temp, const cuda_context_t* p_ctx, int ng, uint32_t mask, int pnx, int pny, int pnc, int ldp, int fnx, int fny, int ldf, int qnx, int qny, int qnc, int ldq, int bat, int su, int sv )
{
    int prc, i, bias, relu, anr, bnr, cnr, enb, non_cached;		
    cuda_kernel_t* p_kernel=&Op->kernel;
    prc=(mask>>1)&0x3;
    bias=(mask>> 3)&0x1;
    relu=(mask>>24)&0x1;
    anr=bat*pny*pnx;
    bnr=pnc*fny*fnx;
    cnr=bat*qny*qnx;
    cnr=IDC_AFFIS(cnr,2);
	enb=prc<2?4:2;
    Op->d_indices=Op->d_indices_cmem=0;
	Op->indices_nb=ldf*(prc!=1?4:6);        
	if(cuMemAlloc( &Op->d_indices, Op->indices_nb )!=CUDA_SUCCESS)
        return idc_error_out_of_device_memory;
    non_cached=Op->indices_nb>p_ctx->cmemnb;
    if(prc!=1){
        sconv_create_kernel( Op, p_temp, p_ctx, pnc, qnc, bnr, cnr, bias, relu, non_cached, ng, prc );
    } else {
        hconv_create_kernel( Op, p_temp, p_ctx, pnc, qnc, bnr, cnr, bias, relu, non_cached, ng );
    }
    if(non_cached==0){
        cuModuleGetGlobal( &Op->d_indices_cmem, NULL, p_ctx->module_conv, "cmem" );
    }
    generate_slider_indices( p_temp, prc, ldp, ldf, anr, bnr, pnx, fnx, fny, pnc, qnc, su, sv, ng );
    cuMemcpyHtoD( Op->d_indices, p_temp, Op->indices_nb );
    if(non_cached==0){
        cuda_kernel_sep_ptr( p_kernel, 3, Op->d_indices );
    }
    bnr=ldf;
    ldq*=enb;
	ldp*=enb;
    ldf*=enb;
    i=non_cached+bias+relu;
    cuda_kernel_sep_i32( p_kernel, 4+i, ldq );
	if(ng>1){ cuda_kernel_sep_i32( p_kernel, 5+i, ldp ); ++i; }
    cuda_kernel_sep_i32( p_kernel, 5+i, ldf );
    cuda_kernel_sep_i32( p_kernel, 6+i, qnx );
    cuda_kernel_sep_i32( p_kernel, 7+i, qny );
    cuda_kernel_sep_i32( p_kernel, 8+i, pnx );
    cuda_kernel_sep_i32( p_kernel, 9+i, pny );
    cuda_kernel_sep_i32( p_kernel,10+i, bnr );
    cuda_kernel_sep_i32( p_kernel,11+i, cnr );
    cuda_kernel_sep_i32( p_kernel,12+i, qnc );
    return idc_success;
}
void idc_conv( idc_convOp_t* Op, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_bias, float alpha, CUstream s )
{
    cuda_kernel_t* p=&Op->kernel;
    uint32_t o=(Op->d_indices_cmem==0)&(Op->d_indices!=0);
    if(Op->d_indices_cmem!=0){
        cuMemcpyDtoDAsync( Op->d_indices_cmem, Op->d_indices, Op->indices_nb, s );
    }
    cuda_kernel_sep_ptr( p, 0, d_c );
    cuda_kernel_sep_ptr( p, 1, d_a );
    cuda_kernel_sep_ptr( p, 2, d_b );
    if(d_bias!=0){ cuda_kernel_sep_ptr( p, 3+o, d_bias ); ++o; }
    cuda_kernel_sep_f32( p, 3+o, alpha );
    cuda_kernel_launch( p, s );
}
void idc_conv_relu( idc_convOp_t* Op, CUdeviceptr d_c, CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_bias, float alpha, float slope, CUstream s )
{
    cuda_kernel_t* p=&Op->kernel;
    uint32_t o=(Op->d_indices_cmem==0)&(Op->d_indices!=0);
    if(Op->d_indices_cmem!=0){
        cuMemcpyDtoDAsync( Op->d_indices_cmem, Op->d_indices, Op->indices_nb, s );
    }
    cuda_kernel_sep_ptr( p, 0, d_c );
    cuda_kernel_sep_ptr( p, 1, d_a );
    cuda_kernel_sep_ptr( p, 2, d_b );
    if(d_bias!=0){ cuda_kernel_sep_ptr( p, 3+o, d_bias ); ++o; }
    cuda_kernel_sep_f32( p, 3+o, alpha );
    cuda_kernel_sep_f32( p, 4+o, slope );
    cuda_kernel_launch( p, s );
}
void idc_conv_releaseOp( idc_convOp_t* Op )
{
    if(Op->d_indices!=0){
        cuMemFree(Op->d_indices);
        Op->d_indices=0;
        Op->d_indices_cmem=0;
    }
}