#include"../include/DeepCore.h"
#include"../include/cuda/cuda_platform.h"
#include"../include/conv/conv.h"
#include"../include/conv/fftconv.h"
#include"../include/conv/cellconv.h"
#include"../include/pooling/pooling.h"
#include"../include/bias/bias.h"
#include"../include/activation/activation.h"

static cuda_platform_t	*	g_pPlat	=NULL;
static cuda_context_t	*	g_pCtx	=NULL;
static char				*	g_pTemp	=NULL;

#define as_devptr(p) (CUdeviceptr)((uintptr_t)(p))

#pragma comment( lib, "cuda.lib" )

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_init()
{
	int i, s, status;
	if((g_pPlat=(cuda_platform_t*)malloc(sizeof(cuda_platform_t)))==NULL)
		return dc_error_out_of_memory;
	if((status=cuda_platform_init(g_pPlat))!=dc_success){
		free(g_pPlat); g_pPlat=NULL;
		return (dc_status_t)status;
	}
	if((g_pCtx=(cuda_context_t*)calloc(g_pPlat->n_devices,sizeof(cuda_context_t)))==NULL){
		free(g_pPlat); g_pPlat=NULL;
		return dc_error_out_of_memory;
	}
	if((g_pTemp=(char*)malloc(1<<24))==0)
	{
		free(g_pPlat); g_pPlat=NULL;
		free(g_pCtx);
		return dc_error_out_of_memory;
	}
	for( s=0; (s<g_pPlat->n_sdevices)&(status==dc_success); ++s )
	{
		for( i=g_pPlat->slist[s]; (i<g_pPlat->slist[s+1])&(status==dc_success); ++i ){
			g_pCtx[i].arch=g_pPlat->sarch[s];
			status=cuda_context_create(&g_pCtx[i],g_pTemp);
		}
	}
	if(status!=dc_success)
	{
		while(i>=0){
			cuda_context_release( &g_pCtx[i] ); 
		}
		free(g_pPlat); g_pPlat=NULL;
		free(g_pCtx);
		free(g_pTemp);
	}
	return (dc_status_t)status;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_set_device( int dev )
{
	cuda_context_bind(&g_pCtx[dev]);
	return dc_success;
}
DEEPCOREAPIENTRY uint64_t DEEPCOREAPICALL dc_create_tensor_shape( unsigned int mask, int n0, int n1, int n2 )
{
	uint64_t shape;
	int tt=mask&0xfc;
	shape=((uint64_t)mask)<<56;
	if(tt==0){
		shape|=(((uint64_t)n2)<<29)|(((uint64_t)n1)<<13)|((uint64_t)n0);	
	} else 
	if(tt=dcMaskTensorTypeFilter){		
		shape|=(((uint64_t)n2)<<20)|(((uint64_t)n1)<<5)|((uint64_t)n0);
	} else
	if(tt==dcMaskTensorTypeBias){
		shape|=n0;
	} else {
		shape=0;
	}
	return shape;
}
DEEPCOREAPIENTRY size_t DEEPCOREAPICALL dc_tensor_pitch( uint64_t shape )
{
	unsigned int mask, tt, pitch;
	mask=(unsigned int)(shape>>56);
	tt=mask&0xfc;
	if(tt!=dcMaskTensorTypeFilter)
	{
		int n=((int)(shape>> 0))&0x1fff;
		int b=((int)(shape>>13))&0xffff;
		int c=((int)(shape>>29))&0x7fff;
		if(tt==dcMaskTensorTypeFullConnection){
			pitch=AFFIS(c*n*n,64);
		} else
		if(tt==dcMaskTensorTypeBias){
			pitch=1;
		} else {
			pitch=AFFIS(b*n*n,64);
		}
	} 
	else 
	{
		int n=(shape>>0)&0x00ff;
		int c=(shape>>5)&0xffff;
		pitch=AFFIS(c*n*n,64);
	}
	return (pitch*((mask&0x3)?2:4));
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_tensor( void** p_devptr, uint64_t shape )
{
	CUdeviceptr devptr;
	unsigned int mask, size, pitch, tt;
	size_t nb;
	mask=(unsigned int)(shape>>56);
	tt=mask&0xfc;
	if(tt!=dcMaskTensorTypeFilter)
	{
		int n=(shape>> 0)&0x1fff;
		int b=(shape>>13)&0xffff;
		int c=(shape>>29)&0x7fff;
		if(tt==dcMaskTensorTypeFullConnection){
			pitch=AFFIS(c*n*n,64);
			size=c*pitch;
		} else
		if(tt==dcMaskTensorTypeBias){
			pitch=1;
			size=c;
		} else {
			pitch=AFFIS(b*n*n,64);
			size=b*pitch;
		}
	} 
	else 
	{
		int fs=(shape>> 0)&0x00ff;
		int nc=(shape>> 5)&0x7fff;
		int nf=(shape>>20)&0x7fff;
		size=nc*fs*fs;
		pitch=AFFIS(size,64);
		size=nf*pitch;
	}
	nb=(size+(tt==0)*(pitch==size))*((mask&0x3)?2:4);
	if(cuMemAlloc( &devptr, nb )!=CUDA_SUCCESS)
		return dc_error_out_of_device_memory;
	cuMemsetD8( devptr, 0, nb );
	*p_devptr=(void*)devptr;
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_convOp( dc_convOp* Op, int idev, unsigned int mask, uint64_t dat_shape, uint64_t fil_shape, int st )
{
	int ds, fs, bat, inc, onc, prc, s;
	if((*Op=(dc_convOp)malloc(sizeof(convOp_t)))==0)
		return dc_error_out_of_memory;	
	ds =((int)(dat_shape>> 0))&0x1fff;
	bat=((int)(dat_shape>>13))&0xffff;
	inc=((int)(dat_shape>>29))&0x7fff;
	prc=((int)(dat_shape>>56))&0x3;
	fs =((int)(fil_shape>> 0))&0x1fff;
	onc=((int)(fil_shape>> 5))&0x7fff;
	s=conv_createOp( (convOp_t*)(*Op), (unsigned int*)g_pTemp, &g_pCtx[idev], prc, ds, fs, bat, inc, onc, st );
	if(s!=SUCCESS){ 
		free((void*)(*Op));
	}
	return (dc_status_t)s;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_fftconvOp( dc_fftconvOp* Op, size_t* p_auxsize, int idev, unsigned int mask, uint64_t pshape, uint64_t qshape )
{
	int ps, qs, bt, nc, nf, prc, opcode, pad, m;
	if((idev<0)|(idev>=g_pPlat->n_devices))
		return dc_error_invalid_value;	
	if((((int)(qshape>>56))&0xfc)==dcMaskTensorTypeFilter){
		qs=((int)(qshape>>0))&0x001f;
		nf=((int)(qshape>>5))&0x7fff;
	} else {		
		qs=((int)(qshape>> 0))&0x1fff;
		nf=((int)(qshape>>31))&0x7fff;
	}
	ps =((int)(pshape>> 0))&0x1fff;
	bt =((int)(pshape>>13))&0xffff;
	nc =((int)(pshape>>29))&0x7fff;
	prc=((int)(pshape>>56))&0x3;
	opcode=mask&0x2;
	pad=(opcode==1)?(qs-1):0;
	if((ps+(pad<<1))>128)
		return dc_error_out_of_maxsize;
	if((*Op=(dc_fftconvOp)malloc(sizeof(fftconvOp_t)))==0)
		return dc_error_out_of_memory;
	m=(mask<<2)|prc;
	if(opcode!=2){
		*p_auxsize=fftconv_createOp( (fftconvOp_t*)(*Op), &g_pCtx[idev], m, ps, qs, bt, nc, nf );
	} else {
		*p_auxsize=fftconv_createOp_filter( (fftconvOp_t*)(*Op), &g_pCtx[idev], m, ps, qs, bt, nc, nf );
	}
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_cellconvOp( dc_cellconvOp* Op, size_t* p_auxsize, int idev, unsigned int mask, uint64_t pshape, uint64_t qshape )
{
	int ps, qs, bt, nc, nf, prc, opcode, m;
	if((idev<0)|(idev>=g_pPlat->n_devices))
		return dc_error_invalid_value;	
	if((*Op=(dc_cellconvOp)malloc(sizeof(cellconvOp_t)))==0)
		return dc_error_out_of_memory;
	if((((int)(qshape>>56))&0xfc)==dcMaskTensorTypeFilter){
		qs=((int)(qshape>>0))&0x001f;
		nf=((int)(qshape>>5))&0x7fff;
	} else {		
		qs=((int)(qshape>> 0))&0x1fff;
		nf=((int)(qshape>>31))&0x7fff;
	}	
	ps =((int)(pshape>> 0))&0x1fff;
	bt =((int)(pshape>>13))&0xffff;
	nc =((int)(pshape>>29))&0x7fff;
	prc=((int)(pshape>>56))&0x3;
	opcode=mask&0x2;
	if((opcode==2)&(ps>32))
		return dc_error_out_of_maxsize;
	m=(mask<<2)|prc;
	if(opcode!=2){
		*p_auxsize=cellconv_createOp( (cellconvOp_t*)(*Op), &g_pCtx[idev], m, ps, qs, bt, nc, nf );
	} else {
		*p_auxsize=cellconv_createOp_filter( (cellconvOp_t*)(*Op), &g_pCtx[idev], m, ps, qs, bt, nc, nf );
	}
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_poolingOp( dc_poolingOp* Op, int idev, unsigned int is_max, uint64_t shape, int n, int st )
{
	int s, b, c, p;
	if((idev<0)|(idev>=g_pPlat->n_devices))
		return dc_error_invalid_value;
	if((*Op=(dc_poolingOp)malloc(sizeof(poolingOp_t)))==0)
		return dc_error_out_of_memory;	
	s=((int)(shape>> 0))&0x1fff;
	b=((int)(shape>>13))&0xffff;
	c=((int)(shape>>29))&0x7fff;
	p=((int)(shape>>56))&0x3;
	return ((dc_status_t)pooling_createOp( (poolingOp_t*)(*Op), &g_pCtx[idev], is_max, p, s, b, c, n, st ));
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_biasOp( dc_biasOp* Op, size_t* p_auxnb, int idev, uint64_t shape )
{
	unsigned int n, b, c, p, size;
	if((idev<0)|(idev>=g_pPlat->n_devices))
		return dc_error_invalid_value;
	if((*Op=(dc_biasOp)malloc(sizeof(biasOp_t)))==0)
		return dc_error_out_of_memory;	
	n=((int)(shape>> 0))&0x1fff;
	b=((int)(shape>>13))&0xffff;
	c=((int)(shape>>29))&0x7fff;
	p=((int)(shape>>56))&0x3;
	size=b*n*n;
	*p_auxnb=bias_createOp( (biasOp_t*)(*Op), &g_pCtx[idev], p, size, c );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_activationOp( dc_activationOp* Op, int idev, unsigned int opcode, uint64_t shape )
{
	int n, b, c, p;
	if((idev<0)|(idev>=g_pPlat->n_devices))
		return dc_error_invalid_value;
	if((*Op=(dc_activationOp)malloc(sizeof(activationOp_t)))==0)
		return dc_error_out_of_memory;
	n=((int)(shape>> 0))&0x1fff;
	b=((int)(shape>>16))&0xffff;
	c=((int)(shape>>31))&0x7fff;
	p=((int)(shape>>56))&0x3;
	activation_createOp( (activationOp_t*)(*Op), &g_pCtx[idev], p, opcode, n, b, c );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_conv( dc_convOp Op, void* d_dst, const void* d_src, const void* d_filter, const float* d_bias, float alpha, float* p_beta, CUstream s )
{
	conv( (convOp_t*)Op, as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_bias), alpha, p_beta, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv( dc_fftconvOp Op, void* d_aux, void* d_dst, const void* d_src, const void* d_filter, const float* d_x, float alpha, float* p_beta, CUstream s )
{
	fftconv( (fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_x), alpha, p_beta, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv_filter( dc_fftconvOp Op, void* d_aux, void* d_filter, const void* d_p, const void* d_q, float alpha, CUstream s )
{
	fftconv_filter( (fftconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_filter), as_devptr(d_p), as_devptr(d_q), alpha, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv( dc_cellconvOp Op, void* d_aux, void* d_dst, const void* d_src, const void* d_filter, const float* d_x, float alpha, float* p_beta, CUstream s )
{
	cellconv( (cellconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_dst), as_devptr(d_src), as_devptr(d_filter), as_devptr(d_x), alpha, p_beta, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv_filter( dc_cellconvOp Op, void* d_aux, void* d_filter, const void* d_p, const void* d_q, float alpha, CUstream s )
{
	cellconv_filter( (cellconvOp_t*)Op, as_devptr(d_aux), as_devptr(d_filter), as_devptr(d_p), as_devptr(d_q), alpha, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_pooling( dc_poolingOp Op, void* d_dst, const void* d_src, int dir, CUstream s )
{
	pooling_launch( (poolingOp_t*)Op, as_devptr(d_dst), as_devptr(d_src), dir, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_update_bias( dc_biasOp Op, void* d_temp, void* d_bias, const void* d_acti, float ratio, CUstream s )
{
	bias_update( (biasOp_t*)Op, as_devptr(d_temp), as_devptr(d_bias), as_devptr(d_acti), ratio, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_activate_fprop( dc_activationOp Op, void* d_dst, const void* d_src, const void* d_bias, float alpha, CUstream s )
{
	activation_fprop( (activationOp_t*)Op, as_devptr(d_dst), as_devptr(d_src), as_devptr(d_bias), alpha, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_activate_bprop( dc_activationOp Op, void* d_ydiff, const void* ydata, const void* d_xdiff, const void* d_xdata, float alpha, CUstream s )
{
	activation_bprop( (activationOp_t*)Op, as_devptr(d_ydiff), as_devptr(ydata), as_devptr(d_xdiff), as_devptr(d_xdata), alpha, s );
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_convOp( dc_convOp Op )
{
	conv_releaseOp((convOp_t*)Op);
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
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_poolingOp( dc_poolingOp Op )
{
	pooling_releaseOp((poolingOp_t*)Op);
	free((void*)Op);
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_biasOp( dc_biasOp Op )
{
	free((void*)Op);
	return dc_success;
}
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_activationOp( dc_activationOp Op )
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