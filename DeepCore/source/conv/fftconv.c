#include"../../include/conv/fftconv.h"

size_t fftconv_createOp( fftconvOp_t* Op, const cuda_context_t* p_ctx, unsigned int mask, int ds, int fs, int bat, int pnc, int qnc )
{
	unsigned int prc, dir, align, inc, onc, fft_size, os, pad, Sxy, lda, ldb;
	cuda_kernel_t* p_kernel;
		
	prc=mask&0x3;
	dir=(mask>>2)&0x1;
	align=prc?32:16;
	pad=dir?(fs-1):0;
	inc=dir?qnc:pnc;
	onc=dir?pnc:qnc;
	ds+=(pad<<1);
	fft_size=fft_get_exec_size(ds);
	os=ds-pad;
	lda=(bat>1)?bat:inc;	
	lda=AFFIS(lda,align);
	ldb=AFFIS(onc,align);
	align=p_ctx->align/(prc?2:4);
	Sxy=((fft_size>>1)+1)*fft_size;

	{
		int is_ext=(ds!=fft_size)&(dir==0);
		int i=(fft_size<=64?0:4)+is_ext+(dir<<1);
		p_kernel=&Op->kfft[0];
		create_fft_kernel_r2c( p_kernel, p_ctx, i, prc );
		cuda_kernel_sgl( p_kernel, bat, inc );
		cuda_kernel_sep_i32( p_kernel, 3, AFFIS(bat*ds*ds,align)>>(1^is_ext) );
		if(is_ext|dir){
			cuda_kernel_sep_i32( p_kernel, 4, ds );
		}
		if(dir){
			cuda_kernel_sep_i32( p_kernel, 5, pad );
		}
	}

	{
		p_kernel=&Op->kfft[1];
		create_fft_kernel_r2c( p_kernel, p_ctx, (fft_size<=64?0:4)+(dir?3:1), prc );
		cuda_kernel_sgl( p_kernel, pnc, qnc );
		cuda_kernel_sep_i32( p_kernel, 3, AFFIS(pnc*fs*fs,align) );
		cuda_kernel_sep_i32( p_kernel, 4, fs );
	}

	{
		int fused=(mask>>3)&0x1;
		int acti_op=mask>>4;
		int i=10*(fft_size>64)+fused*(dir?6:3)+(acti_op&0xf);
		p_kernel=&Op->kfft[2];
		create_fft_kernel_c2r( p_kernel, p_ctx, i, prc );
		cuda_kernel_sgl( p_kernel, bat, onc );
		cuda_kernel_sep_i32( p_kernel, 3, AFFIS(bat*os*os,align) );
		cuda_kernel_sep_i32( p_kernel, 4, os );
	}

	if(bat>1){
		perm3d_create_kernel( &Op->kperm[0], p_ctx, prc, 0, Sxy, bat, inc, Sxy, lda );
		perm3d_create_kernel( &Op->kperm[2], p_ctx, prc, 2, bat, onc, Sxy, lda, Sxy );
		cgemm_create_kernel( &Op->kcgemm, p_ctx , prc, Sxy, bat, inc, onc, lda, ldb, lda );
	} else {
		perm2d_create_kernel( &Op->kperm[0], p_ctx, prc, Sxy, inc, Sxy, lda );
		perm2d_create_kernel( &Op->kperm[2], p_ctx, prc, onc, Sxy, ldb, Sxy );
		cgemv_create_kernel( &Op->kcgemm, p_ctx, prc, Sxy, inc, onc, lda, ldb, ldb );
	}
	perm3d_create_kernel( &Op->kperm[1], p_ctx, prc, 1^dir, Sxy, pnc, qnc, Sxy, ldb );
	cuda_kernel_sep_f32( &Op->kcgemm, 3, (float)(1.0/(fft_size*fft_size)) );

	Sxy*=(prc?4:8);
	Op->divpt[0]=Sxy*bat*inc;
	Op->divpt[1]=Sxy*lda*((bat>1)?inc:1);
	Op->divpt[2]=Op->divpt[1]+Sxy*inc*ldb;
	return (Op->divpt[2]<<1);
}
size_t fftconv_createOp_filter( fftconvOp_t* Op, const cuda_context_t* p_ctx, unsigned int mask, int psize, int pnc, int qsize, int qnc, int bat )
{
	unsigned int prc, align, enb, fft_size, Sxy, lda, ldb, a, b;
	cuda_kernel_t* p_kernel;

	prc=mask&0x3;
	enb=prc?2:4;
	align=prc?32:16;
	lda=AFFIS(pnc,align);
	ldb=AFFIS(qnc,align);
	align=p_ctx->align/(prc?2:4);
	fft_size=psize<=64?64:128;
	Sxy=((fft_size>>1)+1)*fft_size;

	{
		int is_ext=psize!=fft_size;
		int i=(fft_size<=64?0:4)+is_ext;
		p_kernel=&Op->kfft[0];
		create_fft_kernel_r2c( p_kernel, p_ctx, i, prc );
		cuda_kernel_sgl( p_kernel, bat, pnc );
		cuda_kernel_sep_i32( p_kernel, 3, AFFIS(bat*psize*psize,align)>>(1^is_ext) );
		if(is_ext){
			cuda_kernel_sep_i32( p_kernel, 4, psize );
			cuda_kernel_sep_i32( p_kernel, 5, psize );
		}
	}

	{
		p_kernel=&Op->kfft[1];
		create_fft_kernel_r2c( p_kernel, p_ctx, fft_size<=64?3:7, prc );
		cuda_kernel_sgl( p_kernel, bat, qnc );
		cuda_kernel_sep_i32( p_kernel, 3, AFFIS(bat*qsize*qsize,align) );
		cuda_kernel_sep_i32( p_kernel, 4, qsize );
		cuda_kernel_sep_i32( p_kernel, 5, qsize );
	}

	{
		int n=psize-qsize+1;
		p_kernel=&Op->kfft[2];
		create_fft_kernel_c2r( p_kernel, p_ctx, fft_size<=64?9:19, prc );
		cuda_kernel_sgl( p_kernel, pnc, qnc );
		cuda_kernel_sep_i32( p_kernel, 3, AFFIS(pnc*n*n,align) );
		cuda_kernel_sep_i32( p_kernel, 4, n );
		cuda_kernel_sep_i32( p_kernel, 5, n );
	}

	if(bat>1){
		perm3d_create_kernel( &Op->kperm[0], p_ctx, prc, 1, Sxy, bat, pnc, Sxy, lda );
		perm3d_create_kernel( &Op->kperm[1], p_ctx, prc, 1, Sxy, bat, qnc, Sxy, ldb );
		cgemm_create_kernel( &Op->kcgemm, p_ctx, prc, Sxy, pnc, bat, qnc, lda, ldb, lda );
	} else {
		perm2d_create_kernel( &Op->kperm[0], p_ctx, prc, Sxy, pnc, Sxy, lda );
		perm2d_create_kernel( &Op->kperm[1], p_ctx, prc, Sxy, qnc, Sxy, ldb );
		cgevv_create_kernel( &Op->kcgemm, p_ctx, prc, Sxy, pnc, qnc, lda, ldb, lda );
	}
	perm3d_create_kernel( &Op->kperm[2], p_ctx, prc, 2, pnc, qnc, Sxy, lda, Sxy );
	cuda_kernel_sep_f32( &Op->kcgemm, 3, (float)(1.0/(bat*fft_size*fft_size)) );

	Sxy*=(prc?4:8);
	a=bat*(lda+ldb);
	b=lda*qnc;
	Op->divpt[0]=Sxy*bat*pnc;
	Op->divpt[1]=Sxy*bat*lda;
	Op->divpt[2]=Sxy*(a>b?a:b);
	return (Op->divpt[2]<<1);
}
void fftconv( fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_boa, float alpha, float* p_beta, CUstream s )
{
	CUdeviceptr d_aa=d_aux;
	CUdeviceptr d_ba=d_aa+Op->divpt[2];
	CUdeviceptr d_ab=d_aa+Op->divpt[0];
	CUdeviceptr d_bb=d_ba+Op->divpt[1];
	fft2d( &Op->kfft[0], d_aa, d_source, s );
	fft2d( &Op->kfft[1], d_ab, d_filter, s );
	permute( &Op->kperm[0], d_ba, d_aa, s );
	permute( &Op->kperm[1], d_bb, d_ab, s );
	cgemm( &Op->kcgemm, d_aa, d_ba, d_bb, alpha, s );
	permute( &Op->kperm[2], d_ba, d_aa, s );
	if(d_boa!=0){ cuda_kernel_sep_ptr( &Op->kfft[2], 3, d_boa ); }
	if(p_beta!=NULL){ cuda_kernel_sep_f32( &Op->kfft[2], 6+(d_boa!=0), *p_beta ); }
	fft2d( &Op->kfft[2], d_target, d_ba, s );
}
void fftconv_filter( fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_z, CUdeviceptr d_x, CUdeviceptr d_y, float alpha, CUstream s )
{
	CUdeviceptr d_aa=d_aux;
	CUdeviceptr d_ba=d_aa+Op->divpt[2];
	CUdeviceptr d_ab=d_aa+Op->divpt[0];
	CUdeviceptr d_bb=d_ba+Op->divpt[1];
	fft2d( &Op->kfft[0], d_aa, d_x, s );
	fft2d( &Op->kfft[1], d_ab, d_y, s );
	permute( &Op->kperm[0], d_ba, d_aa, s );
	permute( &Op->kperm[1], d_bb, d_ab, s );
	cgemm( &Op->kcgemm, d_aa, d_ba, d_bb, alpha, s );
	permute( &Op->kperm[2], d_ba, d_aa, s );
	fft2d( &Op->kfft[2], d_z, d_ba, s );
}