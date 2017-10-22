#pragma warning(disable:4101)
#include"../include/deepcore.h"
#include"../include/idc_tensor_shape.h"
#include"../include/cuda/cuda_platform.h"
#include"../include/conv/conv.h"
#include"../include/conv/fftconv.h"
#include"../include/blas/gemm.h"
#include"../include/reduce/reduce.h"
#include"../include/bnorm/bnorm.h"
#pragma comment( lib, "cuda.lib" )
#define as_devptr(p) (CUdeviceptr)((uintptr_t)(p))

static cuda_platform_t* g_pPlat=NULL;
static cuda_context_t * g_pCtx =NULL;
static char           * g_pTemp=NULL;

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_init()
{
    int i, s, status;
    if((g_pPlat=(cuda_platform_t*)malloc(sizeof(cuda_platform_t)))==NULL){
        status=dc_error_out_of_memory; goto exit;
	}
    if((status=cuda_platform_init(g_pPlat))!=dc_success)
		goto err0;
    if((g_pCtx=(cuda_context_t*)calloc(g_pPlat->n_devices,sizeof(cuda_context_t)))==NULL){
		status=dc_error_out_of_memory;
    	goto err0;
    }
    if((g_pTemp=(char*)malloc(1<<24))==0){
		status=dc_error_out_of_memory;
    	goto err1;
    }
    for( s=0; (s<g_pPlat->n_sdevices)&(status==dc_success); ++s ){
        for( i=g_pPlat->slist[s]; (i<g_pPlat->slist[s+1])&(status==dc_success); ++i ){
            g_pCtx[i].arch=g_pPlat->sarch[s];
            status=cuda_context_create(&g_pCtx[i],g_pTemp);
        }
    }
    if(status!=dc_success){
        while(i>=0){ cuda_context_release( &g_pCtx[i] ); }
        goto err2;
    }
err2:
	free(g_pTemp);
err1:
	free(g_pCtx);
err0:
    free(g_pPlat); 
	g_pPlat=NULL;
exit:
    return (dc_status_t)status;
}
DEEPCOREAPIENTRY int DEEPCOREAPICALL dc_get_device_count(){ return g_pPlat->n_devices; }
DEEPCOREAPIENTRY int DEEPCOREAPICALL dc_get_superdevice_count(){ return g_pPlat->n_sdevices; }
DEEPCOREAPIENTRY int DEEPCOREAPICALL dc_get_superdevice_size( int i ){ return (g_pPlat->slist[i+1]-g_pPlat->slist[i]); }
DEEPCOREAPIENTRY int DEEPCOREAPICALL dc_get_superdevice_device_id( int sdev, int dev ){ return (g_pPlat->slist[sdev]+dev); }
DEEPCOREAPIENTRY int DEEPCOREAPICALL dc_get_optimal_superdevice_id(){ return g_pPlat->opt_sdev_id; }
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_set_device( int dev )
{
    if((dev<0)|(dev>=g_pPlat->n_devices)) return dc_error_out_of_range;
    cuda_context_bind(&g_pCtx[dev]);
    return dc_success;
}
DEEPCOREAPIENTRY uint64_t DEEPCOREAPICALL dc_create_tensor_shape( int prc, uint32_t size, uint32_t ncbt )
{
    uint64_t nx=(uint64_t)((size>> 0)&0xffff);
    uint64_t ny=(uint64_t)((size>>16)&0xffff);
    uint64_t bt=(uint64_t)((ncbt>> 0)&0xffff);
    uint64_t nc=(uint64_t)((ncbt>>16)&0xffff);
    return ((((uint64_t)prc)<<62)|((nc-1)<<33)|((bt-1)<<18)|((ny-1)<<9)|(nx-1));	
}
DEEPCOREAPIENTRY uint64_t DEEPCOREAPICALL dc_create_tensor_shape_filter( int prc, uint32_t size, uint32_t chan )
{
    uint64_t nx =(uint64_t)((size>> 0)&0xffff);
    uint64_t ny =(uint64_t)((size>>16)&0xffff);
    uint64_t pnc=(uint64_t)((chan>> 0)&0xffff);
    uint64_t qnc=(uint64_t)((chan>>16)&0xffff);
    return ((((uint64_t)prc)<<62)|0x0100000000000000L|((qnc-1)<<25)|((pnc-1)<<10)|((ny-1)<<5)|(nx-1));	
}
DEEPCOREAPIENTRY uint64_t DEEPCOREAPICALL dc_create_tensor_shape_linear( int prc, int ne )
{
    return ((((uint64_t)prc)<<62)|0x0200000000000000L|((uint64_t)ne));	
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_tensor( void** p_devptr, uint64_t shape )
{
    CUdeviceptr devptr;
    uint32_t tt, prc, b, nx, ny, bt, nc, valid_size, pitch, ext, enb, size;
    tt=((uint32_t)(shape>>56))&0x3;
    prc=((uint32_t)(shape>>62))&0x3;
    b=prc&1;
    if(tt==0){
        nx=(((uint32_t)(shape>> 0))&0x01ff)+1;
        ny=(((uint32_t)(shape>> 9))&0x01ff)+1;
        bt=(((uint32_t)(shape>>18))&0x7fff)+1;
        nc=(((uint32_t)(shape>>33))&0x7fff)+1;
        valid_size=bt*ny*nx;
        if(valid_size<=32){
            pitch=idc_minls(valid_size);
        } else
        if((valid_size>32)&(valid_size<=48)){
            pitch=IDC_AFFIS(valid_size,16);
        } else
        if((valid_size>64)&(valid_size<=96)){
            pitch=IDC_AFFIS(valid_size,32);
        } else 
        if(((valid_size>48)&(valid_size<=64))|((valid_size>128)&(valid_size<=192))){
            pitch=IDC_AFFIS(valid_size,64);
        } else {
            pitch=IDC_AFFIS(valid_size,128);
        }
        size=pitch*(nc>>b)+1;
    } else
    if(tt==1){
        nx=(((uint32_t)(shape>> 0))&0x001f)+1;
        ny=(((uint32_t)(shape>> 5))&0x001f)+1;
        bt=(((uint32_t)(shape>>10))&0x7fff)+1;
        nc=(((uint32_t)(shape>>25))&0x7fff)+1;
        valid_size=bt*ny*nx;
        pitch=IDC_AFFIS(valid_size,8);
        size=(nc>>b)*pitch;
    } else {
        size=((uint32_t)(shape))>>b;
    }
    enb=prc<2?4:2;
    if(cuMemAlloc( &devptr, size*enb )!=CUDA_SUCCESS)
        return dc_error_out_of_device_memory;
    if(tt==0){
        cuMemsetD8( devptr+size*enb, 0, enb ); 
    }
    *p_devptr=(void*)devptr;
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_release_tensor( void* p_devptr )
{
    if(p_devptr!=0){ cuMemFree(as_devptr(p_devptr)); }
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_tensor_zero( void* p, uint64_t shape, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape;
    idc_get_tensor_shape( &ishape, shape );
    if(ishape.ncol>1){
        cuMemsetD2D8Async( as_devptr(p), ishape.pitch, 0, ishape.size, ishape.ncol, s );
    } else {
        cuMemsetD8Async( as_devptr(p), 0, ishape.size, s );
    }
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_tensor_copy( void* p_dst, uint64_t shape_dst, const void* p_src, uint64_t shape_src, size_t vnb, size_t h, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape_dst, ishape_src;
    idc_get_tensor_shape( &ishape_dst, shape_dst );
    idc_get_tensor_shape( &ishape_src, shape_src );
    if((ishape_src.size<vnb)|(ishape_dst.size<vnb)|(ishape_src.ncol<h)|(ishape_dst.ncol<h))
        return dc_error_out_of_range;
    mem2d.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.dstDevice    =as_devptr(p_dst);
    mem2d.dstPitch     =ishape_dst.pitch;	
    mem2d.dstXInBytes  =0;
    mem2d.dstY         =0;
    mem2d.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.srcDevice    =as_devptr(p_src);
    mem2d.srcPitch     =ishape_src.pitch;	
    mem2d.srcXInBytes  =0;
    mem2d.srcY         =0;
    mem2d.WidthInBytes =vnb;
    mem2d.Height       =h;
    cuMemcpy2DAsync( &mem2d, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_tensor_store( void* p_dst, uint64_t shape, const void* p_src, size_t src_pitch, size_t vnb, size_t h, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape;
    idc_get_tensor_shape( &ishape, shape );
    if((ishape.size<vnb)|(ishape.ncol<h))
        return dc_error_out_of_range;
    mem2d.dstMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.dstDevice    =as_devptr(p_dst);
    mem2d.dstPitch     =ishape.pitch;	
    mem2d.dstXInBytes  =0;
    mem2d.dstY         =0;
    mem2d.srcMemoryType=CU_MEMORYTYPE_HOST;
    mem2d.srcHost      =p_src;
    mem2d.srcPitch     =src_pitch;	
    mem2d.srcXInBytes  =0;
    mem2d.srcY         =0;
    mem2d.WidthInBytes =vnb;
    mem2d.Height       =h;
    cuMemcpy2DAsync( &mem2d, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_tensor_load( void* p_dst, size_t dst_pitch, const void* p_src, uint64_t shape, size_t vnb, size_t h, CUstream s )
{
    CUDA_MEMCPY2D mem2d;
    idc_tensor_shape_t ishape;
    idc_get_tensor_shape( &ishape, shape );
    if((ishape.size<vnb)|(ishape.ncol<h))
        return dc_error_out_of_range;
    mem2d.dstMemoryType=CU_MEMORYTYPE_HOST;
    mem2d.dstHost      =p_dst;
    mem2d.dstPitch     =dst_pitch;	
    mem2d.dstXInBytes  =0;
    mem2d.dstY         =0;
    mem2d.srcMemoryType=CU_MEMORYTYPE_DEVICE;
    mem2d.srcDevice    =as_devptr(p_src);
    mem2d.srcPitch     =ishape.pitch;	
    mem2d.srcXInBytes  =0;
    mem2d.srcY         =0;
    mem2d.WidthInBytes =vnb;
    mem2d.Height       =h;
    cuMemcpy2DAsync( &mem2d, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_convOp( dc_convOp* Op, int ng, uint32_t mask, uint64_t pshape, uint64_t fshape, uint64_t qshape, uint32_t pad, uint32_t st )
{
    idc_op_param_t param;
    int idev, prc, su, sv, s;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    idc_get_op_param( &param, pshape, fshape, qshape );
    su=st&0xffff;
    sv=(st>>16)&0xffff;
    if((((param.pnx-param.fnx)/su+1)!=param.qnx)|(((param.pny-param.fny)/sv+1)!=param.qny))
        return dc_error_mismatch;
    if((*Op=(dc_convOp)malloc(sizeof(idc_convOp_t)))==0)
        return dc_error_out_of_memory;
    prc=param.prc<<1;
    if((g_pCtx[idev].arch!=60)&(prc==dcMaskPrecisionHalf)){ prc=dcMaskPrecisionMixed; }
    s=idc_conv_createOp( (idc_convOp_t*)(*Op), (uint32_t*)g_pTemp, &g_pCtx[idev], ng, prc|mask, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.ldf, param.qnx, param.qny, param.qnc, param.ldq, param.bat, su, sv );
    if(s!=idc_success){ 
        free((void*)(*Op));
    }
    return (dc_status_t)s;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_fftconvOp( dc_fftconvOp* Op, size_t* p_auxsize, uint32_t mask, uint64_t pshape, uint64_t fshape, uint64_t qshape, uint32_t pad )
{
    idc_op_param_t param;
    int dir, prc, idev, snx, sny, dnx, dny, pad_x, pad_y;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    if((mask&0x1)&((mask&dcMaskActivationRelu)!=0)&((mask&dcMaskAddBiasOrMulDiff)!=0)) 
        return dc_error_mutually_exclusive;
    idc_get_op_param( &param, pshape, fshape, qshape );
    dir=mask&0x1;
    pad_x=pad&0xffff;
    pad_y=(pad>>16)&0xffff;
    snx=dir?param.qnx:param.pnx;
    sny=dir?param.qny:param.pny;
    dnx=dir?param.pnx:param.qnx;
    dny=dir?param.pny:param.qny;
    snx+=pad_x<<1;
    sny+=pad_y<<1;
    if(((snx-param.fnx+1)!=dnx)|((sny-param.fny+1)!=dny))
        return dc_error_mismatch;
    if((*Op=(dc_fftconvOp)malloc(sizeof(idc_fftconvOp_t)))==0)
        return dc_error_out_of_memory;
    prc=param.prc<<1;
    if((g_pCtx[idev].arch!=60)&(prc==dcMaskPrecisionHalf)){ prc=dcMaskPrecisionMixed; }
    *p_auxsize=idc_fftconv_createOp( (idc_fftconvOp_t*)(*Op), &g_pCtx[idev], prc|mask, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.qnx, param.qny, param.qnc, param.ldq, param.bat, pad_x, pad_y );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_fftconvOp_grad( dc_fftconvOp* Op, size_t* p_auxsize, uint32_t mask, uint64_t pshape, uint64_t fshape, uint64_t qshape, uint32_t pad )
{
	idc_op_param_t param;
	int prc, idev, pad_x, pad_y;
	idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
	if(idev<0) return dc_error_no_active_device;
	idc_get_op_param( &param, pshape, fshape, qshape );
	pad_x=pad&0xffff;
	pad_y=(pad>>16)&0xffff;
	if(((param.pnx+(pad_x<<1)-param.qnx+1)!=param.fnx)|((param.pny+(pad_y<<1)-param.qny+1)!=param.fny))
		return dc_error_mismatch;
	if((*Op=(dc_fftconvOp)malloc(sizeof(idc_fftconvOp_t)))==0)
		return dc_error_out_of_memory;
	prc=param.prc;
	if((g_pCtx[idev].arch!=60)&(prc==dcMaskPrecisionHalf)){ prc=dcMaskPrecisionMixed; }
	*p_auxsize=idc_fftconv_createOp_grad( (idc_fftconvOp_t*)(*Op), &g_pCtx[idev], prc>>1, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.qnx, param.qny, param.qnc, param.ldq, param.bat, pad_x, pad_y );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_cellconvOp( dc_cellconvOp* Op, size_t* p_auxsize, unsigned int mask, uint64_t pshape, uint64_t fshape, uint64_t qshape, uint32_t pad )
{
    idc_op_param_t param;
    int dir, prc, idev, snx, sny, dnx, dny, pad_x, pad_y;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    if((mask&0x1)&((mask&dcMaskActivationRelu)!=0)&((mask&dcMaskAddBiasOrMulDiff)!=0))
        return dc_error_mutually_exclusive;
    idc_get_op_param( &param, pshape, fshape, qshape );
    dir=mask&0x1;
    pad_x=pad&0xffff;
    pad_y=(pad>>16)&0xffff;
    snx=dir?param.qnx:param.pnx;
    sny=dir?param.qny:param.pny;
    dnx=dir?param.pnx:param.qnx;
    dny=dir?param.pny:param.qny;
    snx+=pad_x<<1;
    sny+=pad_y<<1;
    if(((snx-param.fnx+1)!=dnx)|((sny-param.fny+1)!=dny))
        return dc_error_mismatch;
    if((*Op=(dc_cellconvOp)malloc(sizeof(idc_fftconvOp_t)))==0)
        return dc_error_out_of_memory;
    prc=param.prc<<1;
    if((g_pCtx[idev].arch!=60)&(prc==dcMaskPrecisionHalf)){ prc=dcMaskPrecisionMixed; }
    *p_auxsize=idc_cellconv_createOp( (idc_fftconvOp_t*)(*Op), &g_pCtx[idev], prc|mask, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.qnx, param.qny, param.qnc, param.ldq, param.bat, pad_x, pad_y );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_cellconvOp_grad( dc_cellconvOp* Op, size_t* p_auxsize, uint32_t mask, uint64_t pshape, uint64_t fshape, uint64_t qshape, uint32_t pad )
{
    idc_op_param_t param;
    int prc, idev, pad_x, pad_y;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    idc_get_op_param( &param, pshape, fshape, qshape );
    pad_x=pad&0xffff;
    pad_y=(pad>>16)&0xffff;
    if(((param.pnx+(pad_x<<1)-param.qnx+1)!=param.fnx)|((param.pny+(pad_y<<1)-param.qny+1)!=param.fny))
        return dc_error_mismatch;
    if((*Op=(dc_cellconvOp)malloc(sizeof(idc_fftconvOp_t)))==0)
        return dc_error_out_of_memory;
    prc=param.prc;
    if((g_pCtx[idev].arch!=60)&(prc==dcMaskPrecisionHalf)){ prc=dcMaskPrecisionMixed; }
    *p_auxsize=idc_cellconv_createOp_grad( (idc_fftconvOp_t*)(*Op), &g_pCtx[idev], prc>>1, param.pnx, param.pny, param.pnc, param.ldp, param.fnx, param.fny, param.qnx, param.qny, param.qnc, param.ldq, param.bat, pad_x, pad_y );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_gemmOp( dc_gemmOp* Op, int ng, uint32_t mask, uint64_t ashape, uint64_t bshape, uint64_t cshape )
{
    idc_op_param_t param;
    int idev, prc, mode, n, anr, bnr, cnc, lda, ldb, ldc, enb;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    idc_get_op_param( &param, ashape, bshape, cshape );
    if((param.pnx!=param.qnx)|(param.pny!=param.qny))
        return dc_error_mismatch;
    if((*Op=(dc_gemmOp)malloc(sizeof(cuda_kernel_t)))==0)
        return dc_error_out_of_memory;
    prc=param.prc<<1;
    if((g_pCtx[idev].arch!=60)&(prc==dcMaskPrecisionHalf)){ prc=dcMaskPrecisionMixed; }
    mode=mask&0x3;
    n=param.bat*param.pny*param.pnx;
    if(mode==0){
        anr=n;
        bnr=param.pnc;
        cnc=param.qnc;
        lda=param.ldp;
        ldb=param.ldf;
        ldc=param.ldq;
    } else 
    if(mode==1){
        anr=n;
        bnr=param.qnc;
        cnc=param.pnc;
        lda=param.ldq;
        ldb=param.ldf;
        ldc=param.ldp;
    } else {
        bnr=n;
        anr=param.pnc;
        cnc=param.qnc;
        lda=param.ldf;
        ldb=param.ldp;
        ldc=param.ldq;
    }
    enb=(prc&0x2)!=4?4:2;
    idc_gemm_create_kernel( (cuda_kernel_t*)(*Op), &g_pCtx[idev], prc|mask, mode, ng, anr, bnr, cnc, lda*enb, ldb*enb, ldc*enb );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_reduceOp( dc_reduceOp* Op, uint32_t mask, uint64_t shape )
{
    idc_tensor_shape_t ishape;
    int idev, prc, s;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    if((*Op=(dc_reduceOp)malloc(sizeof(idc_reduceOp_t)))==0)
        return dc_error_out_of_memory;
    idc_get_tensor_shape( &ishape, shape );
    prc=((uint32_t)(shape>>61))&0x6;
    return ((dc_status_t)idc_reduce_createOp( (idc_reduceOp_t*)(*Op), &g_pCtx[idev], prc|mask, ishape.size>>2, ishape.pitch>>2, ishape.ncol ));
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_bnormOp( dc_bnormOp* Op, uint32_t mask, uint64_t shape2d, uint64_t shape1d )
{
    idc_tensor_shape_t ishape2d, ishape1d;
    int idev, prc, s;
    idev=cuda_context_get_current( g_pCtx, g_pPlat->n_devices );
    if(idev<0) return dc_error_no_active_device;
    if((*Op=(dc_bnormOp)malloc(sizeof(cuda_kernel_t)))==0)
        return dc_error_out_of_memory;
    idc_get_tensor_shape( &ishape2d, shape2d );
    idc_get_tensor_shape( &ishape1d, shape1d );
    if(ishape1d.size<ishape2d.ncol)
        return dc_error_mismatch;
    idc_bnorm_createOp( (cuda_kernel_t*)(*Op), &g_pCtx[idev], mask, ishape2d.size, ishape2d.pitch, ishape2d.ncol );
    return idc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_conv( dc_convOp Op, void* d_dst, const void* d_src, const void* d_filter, const void* d_bias, float alpha, CUstream s )
{
    idc_conv( (idc_convOp_t*)Op, as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_bias), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_conv_relu( dc_convOp Op, void* d_dst, const void* d_src, const void* d_filter, const void* d_bias, float alpha, float slope, CUstream s )
{
    idc_conv_relu( (idc_convOp_t*)Op, as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_bias), alpha, slope, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv( dc_fftconvOp Op, void* d_aux, void* d_dst, const void* d_src, const void* d_filter, const float* d_x, float alpha, CUstream s )
{
    idc_fftconv( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_x), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv_relu( dc_fftconvOp Op, void* d_aux, void* d_dst, const void* d_src, const void* d_filter, const float* d_x, float alpha, float slope, CUstream s )
{
    idc_fftconv_relu( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_x), alpha, slope, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv_grad( dc_fftconvOp Op, void* d_aux, void* d_filter, const void* d_p, const void* d_q, float ratio, CUstream s )
{
    idc_fftconv_grad( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_filter), as_devptr(d_p), as_devptr(d_q), ratio, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv( dc_cellconvOp Op, void* d_aux, void* d_dst, const void* d_src, const void* d_filter, const float* d_x, float alpha, CUstream s )
{
    idc_cellconv( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_x), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv_relu( dc_cellconvOp Op, void* d_aux, void* d_dst, const void* d_src, const void* d_filter, const float* d_x, float alpha, float slope, CUstream s )
{
    idc_cellconv_relu( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_x), alpha, slope, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv_grad( dc_cellconvOp Op, void* d_aux, void* d_filter, const void* d_p, const void* d_q, float ratio, CUstream s )
{
    idc_cellconv_grad( (idc_fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_filter), as_devptr(d_p), as_devptr(d_q), ratio, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_gemm( dc_gemmOp Op, void* d_c, const void* d_a, const void* d_b, const void* d_bias, float alpha, CUstream s )
{
    idc_gemm( (cuda_kernel_t*)Op, as_devptr(d_c), as_devptr(d_a), as_devptr(d_b), as_devptr(d_bias), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_gemm_relu( dc_gemmOp Op, void* d_c, const void* d_a, const void* d_b, const void* d_bias, float alpha, float slope, CUstream s )
{
    idc_gemm_relu( (cuda_kernel_t*)Op, as_devptr(d_c), as_devptr(d_a), as_devptr(d_b), as_devptr(d_bias), alpha, slope, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_gemm_grad( dc_gemmOp Op, void* d_c, const void* d_a, const void* d_b, float alpha, CUstream s )
{
    idc_gemm_grad( (cuda_kernel_t*)Op, as_devptr(d_c), as_devptr(d_a), as_devptr(d_b), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_pooling( dc_poolingOp Op, void* dst, const void* src, CUstream s )
{
    idc_pooling_launch( (idc_poolingOp_t*)Op, as_devptr(dst), as_devptr(src), s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_reduce( dc_reduceOp Op, void* c, const void* a, const void* b, float alpha, CUstream s )
{
    idc_reduce_launch( (idc_reduceOp_t*)Op, as_devptr(c), as_devptr(a), as_devptr(b), alpha, s );
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_bnorm( dc_bnormOp Op, void* y, void* mean, void* vari, const void* x, const void* gamm, const void* beta, CUstream s )
{
    idc_bnorm( (cuda_kernel_t*)Op, as_devptr(y), as_devptr(mean), as_devptr(vari), as_devptr(x), as_devptr(gamm), as_devptr(beta), s );
    return idc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_bnorm_grad( dc_bnormOp Op, void* dx, void* dg, void* db, const void* x, const void* dy, const void* gamm, const void* mean, const void* vari, CUstream s )
{
    idc_bnorm_grad( (cuda_kernel_t*)Op, as_devptr(dx), as_devptr(dg), as_devptr(db), as_devptr(x), as_devptr(dy), as_devptr(gamm), as_devptr(mean), as_devptr(vari), s );
    return idc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_convOp( dc_convOp Op )
{
    idc_conv_releaseOp((idc_convOp_t*)Op);
    free((void*)Op);
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_fftconvOp( dc_fftconvOp Op )
{
    free((void*)Op);
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_cellconvOp( dc_cellconvOp Op )
{
    free((void*)Op);
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_reduceOp( dc_reduceOp Op )
{
    idc_reduce_releaseOp((idc_reduceOp_t*)Op);
    free((void*)Op);
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_bnormOp( dc_bnormOp Op )
{
    free((void*)Op);
    return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_exit()
{
    if(g_pPlat!=NULL){
        int i;
        for( i=0; i<g_pPlat->n_devices; ++i ){ 
            cuda_context_release( &g_pCtx[i] ); 
        }
        free(g_pPlat); g_pPlat=NULL;
        free(g_pCtx);
        free(g_pTemp);
    }
    return dc_success;
}
