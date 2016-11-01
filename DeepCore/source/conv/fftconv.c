#include"../../include/conv/fftconv.h"

int fftconv_createOp( fftconvOp_t* Op, size_t* p_auxnb, const cuda_context_t* p_ctx, unsigned int mask, int ds, int fs, int bat, int pnc, int qnc )
{
	int prc, dir, align, inc, onc, pad, fft_size, os, Sxy, axis, radix;
	cuda_kernel_t* p_kernel;

	dir=mask&0x1;
	inc=dir?qnc:pnc;
	onc=dir?pnc:qnc;
	Op->n_kernels=(bat*onc)<=65536?4:7;
	if((Op->p_kernel=(cuda_kernel_t*)malloc(Op->n_kernels*sizeof(cuda_kernel_t)))==0)
		return ERROR_OUT_OF_MEMORY;
	pad=fs-1;
	Sxy=ds+(dir?(pad<<1):0);
	fft_size=fft_get_exec_size(Sxy);
	if(fft_size>128) return ERROR_OUT_OF_MAX_SIZE;	
	prc=(mask>>1)&0x3;
	os=Sxy-pad;
	fft_size=fft_size<16?16:fft_size;
	axis=__bffs(fft_size)-4;
	align=prc?(BASE_PITCH/2):(BASE_PITCH/4);
	radix=fft_size<32?8:4;

	{
		int is_ext=(ds!=fft_size)&(dir==0);
		int ldr=AFFIS(bat*ds*ds,align);
		p_kernel=&Op->p_kernel[0];
		create_fft_kernel_r2c( p_kernel, p_ctx, (axis<<2)+is_ext+(dir<<1), prc );
		if(fft_size>32)
		{
			cuda_kernel_sgl( p_kernel, bat, inc, 1 );
			cuda_kernel_sep_i32( p_kernel, 3, (is_ext|dir)?ldr:(ldr>>1) );
			if(is_ext|dir){
				cuda_kernel_sep_i32( p_kernel, 4, ds );
				cuda_kernel_sep_i32( p_kernel, 5, ds );
			}
			if(dir){
				cuda_kernel_sep_i32( p_kernel, 6, pad );
				cuda_kernel_sep_i32( p_kernel, 7, pad );
			}
		}
		else
		{
			int n=bat*inc;
			cuda_kernel_sgl( p_kernel, (n+radix-1)/radix, 1, 1 );
			cuda_kernel_sep_i32( p_kernel, 3, ldr );
			if(is_ext|dir){
				cuda_kernel_sep_i32( p_kernel, 4, ds );
				cuda_kernel_sep_i32( p_kernel, 5, ds );
			}
			if(dir==0){
				cuda_kernel_sep_i32( p_kernel, 4+(is_ext<<1), bat );
				cuda_kernel_sep_i32( p_kernel, 5+(is_ext<<1), n   );
			} else {
				cuda_kernel_sep_i32( p_kernel, 6, pad );
				cuda_kernel_sep_i32( p_kernel, 7, pad );
				cuda_kernel_sep_i32( p_kernel, 8, bat );
				cuda_kernel_sep_i32( p_kernel, 9, n   );
			}
		}
	}

	{
		int ldr=AFFIS(pnc*fs*fs,align);
		p_kernel=&Op->p_kernel[1];
		create_fft_kernel_r2c( p_kernel, p_ctx, (axis<<2)+(dir?3:1), prc );
		cuda_kernel_sep_i32( p_kernel, 3, ldr );
		cuda_kernel_sep_i32( p_kernel, 4, fs  );
		cuda_kernel_sep_i32( p_kernel, 5, fs  );
		if(fft_size>32){
			cuda_kernel_sgl( p_kernel, pnc, qnc, 1 );
		} else {
			int n=pnc*qnc;
			cuda_kernel_sgl( p_kernel, (n+radix-1)/radix, 1, 1 );
			cuda_kernel_sep_i32( p_kernel, 6, pnc );
			cuda_kernel_sep_i32( p_kernel, 7, n   );
		}
	}

	{
		int fused=(mask>>3)&0x1;
		int atvop=mask>>4;
		int ldr=AFFIS(bat*os*os,align);
		int o=fused+(atvop!=0);
		p_kernel=&Op->p_kernel[2];
		create_fft_kernel_c2r( p_kernel, p_ctx, 10*axis+fused*(dir?6:3)+atvop, prc );
		cuda_kernel_sep_i32( p_kernel, o+3, ldr );
		cuda_kernel_sep_i32( p_kernel, o+4, os  );
		cuda_kernel_sep_i32( p_kernel, o+5, os  );
		if(fft_size>32){
			cuda_kernel_sgl( p_kernel, bat, onc, 1 );
		} else {
			int n=bat*onc;
			cuda_kernel_sgl( p_kernel, (n+radix-1)/radix, 1, 1 );
			cuda_kernel_sep_i32( p_kernel, o+6, bat );
			cuda_kernel_sep_i32( p_kernel, o+7, n   );
		}
	}

	Sxy=((fft_size>>1)+1)*fft_size;
	if(Op->n_kernels==4)
	{		
		cgemm_flat_create_kernel( &Op->p_kernel[3], p_ctx, prc, Sxy, bat, inc, onc, dir );
		Sxy*=(prc?4:8);
		Op->divpt[0]=Sxy*bat*inc;
		Op->divpt[1]=Sxy*inc*onc;
		*p_auxnb=Op->divpt[0]+Op->divpt[1]+Sxy*bat*onc;
	}
	else
	{
		int n=prc?32:16;
		int lda=bat>1?bat:inc;
		int ldb=AFFIS(onc,n);
		lda=AFFIS(lda,n);
		if(bat>1){
			cgemm_create_kernel( &Op->p_kernel[3], p_ctx, prc, Sxy, bat, inc, onc, lda, ldb, lda );
			perm3d_create_kernel( &Op->p_kernel[4], p_ctx, prc, 0, Sxy, bat, inc, Sxy, lda );
			perm3d_create_kernel( &Op->p_kernel[6], p_ctx, prc, 2, bat, onc, Sxy, lda, Sxy );
		} else {
			cgemv_create_kernel( &Op->p_kernel[3], p_ctx, prc, Sxy, inc, onc, lda, ldb, ldb );
			perm2d_create_kernel( &Op->p_kernel[4], p_ctx, prc, Sxy, inc, Sxy, lda );
			perm2d_create_kernel( &Op->p_kernel[6], p_ctx, prc, onc, Sxy, ldb, Sxy );
		}
		perm3d_create_kernel( &Op->p_kernel[5], p_ctx, prc, 1^dir, Sxy, pnc, qnc, Sxy, ldb );

		Sxy*=(prc?4:8);
		Op->divpt[0]=Sxy*bat*inc;
		Op->divpt[1]=Sxy*lda*(bat>1?inc:1);
		Op->divpt[2]=Op->divpt[1]+Sxy*(inc>onc?inc:onc)*ldb;
		*p_auxnb=(Op->divpt[2]<<1);
	}
	cuda_kernel_sep_f32( &Op->p_kernel[3], 3, (float)(1.0/(fft_size*fft_size)) );
	return SUCCESS;
}
int fftconv_createOp_filter( fftconvOp_t* Op, size_t* p_auxnb, const cuda_context_t* p_ctx, unsigned int mask, int psize, int pnc, int qsize, int qnc, int bat )
{
	unsigned int prc, align, enb, fft_size, axis, Sxy, lda, ldb, radix, a, b;
	cuda_kernel_t* p_kernel;

	Op->n_kernels=7;
	if((Op->p_kernel=(cuda_kernel_t*)malloc(Op->n_kernels*sizeof(cuda_kernel_t)))==0)
		return ERROR_OUT_OF_MEMORY;
	if(psize>128) return ERROR_OUT_OF_MAX_SIZE;

	prc=(mask>>1)&0x3;
	enb=prc?2:4;
	align=prc?32:16;
	lda=AFFIS(pnc,align);
	ldb=AFFIS(qnc,align);
	align=prc?(BASE_PITCH/2):(BASE_PITCH/4);
	fft_size=fft_get_exec_size(psize);
	fft_size=fft_size<=16?16:fft_size;
	axis=__bffs(fft_size)-4;
	Sxy=((fft_size>>1)+1)*fft_size;
	radix=fft_size<32?8:4;

	{
		int is_ext=psize!=fft_size;
		int i=(axis<<2)+is_ext;
		int ldr=AFFIS(bat*psize*psize,align);
		p_kernel=&Op->p_kernel[0];
		create_fft_kernel_r2c( p_kernel, p_ctx, i, prc );
		if(fft_size>32)
		{
			cuda_kernel_sgl( p_kernel, bat, pnc, 1 );
			cuda_kernel_sep_i32( p_kernel, 3, ldr>>(1^is_ext) );
			if(is_ext){
				cuda_kernel_sep_i32( p_kernel, 4, psize );
				cuda_kernel_sep_i32( p_kernel, 5, psize );
			}
		}
		else
		{
			int n=bat*pnc;
			cuda_kernel_sgl( p_kernel, (n+radix-1)/radix, 1, 1 );
			cuda_kernel_sep_i32( p_kernel, 3, ldr   );
			cuda_kernel_sep_i32( p_kernel, 4, psize );
			cuda_kernel_sep_i32( p_kernel, 5, psize );			
			cuda_kernel_sep_i32( p_kernel, 6, bat   );
			cuda_kernel_sep_i32( p_kernel, 7, n     );
		}
	}

	{
		int ldr=AFFIS(bat*qsize*qsize,align);
		p_kernel=&Op->p_kernel[1];
		create_fft_kernel_r2c( p_kernel, p_ctx, 10*axis+3, prc );
		cuda_kernel_sep_i32( p_kernel, 3, ldr   );
		cuda_kernel_sep_i32( p_kernel, 4, qsize );
		cuda_kernel_sep_i32( p_kernel, 5, qsize );
		if(fft_size>32){
			cuda_kernel_sgl( p_kernel, bat, qnc, 1 );
		} else {
			int n=bat*qnc;
			cuda_kernel_sgl( p_kernel, (n+radix-1)/radix, 1, 1 );
			cuda_kernel_sep_i32( p_kernel, 6, bat );
			cuda_kernel_sep_i32( p_kernel, 7, n   );
		}
	}

	{
		int os=psize-qsize+1;
		int ldr=AFFIS(pnc*os*os,align);
		int n=pnc*qnc;
		p_kernel=&Op->p_kernel[2];
		create_fft_kernel_c2r( p_kernel, p_ctx, 10*axis+9, prc );
		cuda_kernel_sep_i32( p_kernel, 3, ldr );
		cuda_kernel_sep_i32( p_kernel, 4, os  );
		cuda_kernel_sep_i32( p_kernel, 5, os  );
		if(fft_size>32){
			cuda_kernel_sgl( p_kernel, pnc, qnc, 1 );
		} else {
			cuda_kernel_sgl( p_kernel, (n+radix-1)/radix, 1, 1 );
			cuda_kernel_sep_i32( p_kernel, 6, pnc );
			cuda_kernel_sep_i32( p_kernel, 7, n   );
		}
	}

	if(bat>1){
		cgemm_create_kernel( &Op->p_kernel[3], p_ctx, prc, Sxy, pnc, bat, qnc, lda, ldb, lda );
		perm3d_create_kernel( &Op->p_kernel[4], p_ctx, prc, 1, Sxy, bat, pnc, Sxy, lda );
		perm3d_create_kernel( &Op->p_kernel[6], p_ctx, prc, 1, Sxy, bat, qnc, Sxy, ldb );
	} else {
		cgevv_create_kernel( &Op->p_kernel[3], p_ctx, prc, Sxy, pnc, qnc, lda, ldb, lda );
		perm2d_create_kernel( &Op->p_kernel[4], p_ctx, prc, Sxy, pnc, Sxy, lda );
		perm2d_create_kernel( &Op->p_kernel[6], p_ctx, prc, Sxy, qnc, Sxy, ldb );
	}
	perm3d_create_kernel( &Op->p_kernel[5], p_ctx, prc, 2, pnc, qnc, Sxy, lda, Sxy );
	cuda_kernel_sep_f32( &Op->p_kernel[3], 3, (float)(1.0/(bat*fft_size*fft_size)) );

	Sxy*=(prc?4:8);
	a=bat*(lda+ldb);
	b=lda*qnc;
	Op->divpt[0]=Sxy*bat*pnc;
	Op->divpt[1]=Sxy*bat*lda;
	Op->divpt[2]=Sxy*(a>b?a:b);
	*p_auxnb=Op->divpt[2]<<1;
	return SUCCESS;
}
void fftconv( fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_bias_or_atv, const float* alpha, CUstream s )
{
	if(Op->n_kernels==4)
	{	
		CUdeviceptr d_a=d_aux;
		CUdeviceptr d_b=d_a+Op->divpt[0];
		CUdeviceptr d_c=d_b+Op->divpt[1];
		fft2d_r2c( &Op->p_kernel[0], d_a, d_source, s );
		fft2d_r2c( &Op->p_kernel[1], d_b, d_filter, s );
		cgemm( &Op->p_kernel[3], d_c, d_a, d_b, 1.f, s );
		fft2d_c2r( &Op->p_kernel[2], d_target, d_c, d_bias_or_atv, alpha, s );
	}
	else
	{
		CUdeviceptr d_aa=d_aux;
		CUdeviceptr d_ab=d_aa+Op->divpt[0];
		CUdeviceptr d_ba=d_aa+Op->divpt[2];
		CUdeviceptr d_bb=d_ba+Op->divpt[1];
		fft2d_r2c( &Op->p_kernel[0], d_aa, d_source, s );
		fft2d_r2c( &Op->p_kernel[1], d_ab, d_filter, s );
		permute( &Op->p_kernel[4], d_ba, d_aa, s );
		permute( &Op->p_kernel[5], d_bb, d_ab, s );
		cgemm( &Op->p_kernel[3], d_aa, d_ba, d_bb, 1.f, s );
		permute( &Op->p_kernel[6], d_ba, d_aa, s );
		fft2d_c2r( &Op->p_kernel[2], d_target, d_ba, d_bias_or_atv, alpha, s );
	}
}
void fftconv_filter( fftconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_z, CUdeviceptr d_x, CUdeviceptr d_y, float ratio, CUstream s )
{
	CUdeviceptr d_aa=d_aux;
	CUdeviceptr d_ba=d_aa+Op->divpt[2];
	CUdeviceptr d_ab=d_aa+Op->divpt[0];
	CUdeviceptr d_bb=d_ba+Op->divpt[1];
	fft2d_r2c( &Op->p_kernel[0], d_aa, d_x, s );
	fft2d_r2c( &Op->p_kernel[1], d_ab, d_y, s );
	permute( &Op->p_kernel[4], d_ba, d_aa, s );
	permute( &Op->p_kernel[5], d_bb, d_ab, s );
	cgemm( &Op->p_kernel[3], d_aa, d_ba, d_bb, 1.f, s );
	permute( &Op->p_kernel[6], d_ba, d_aa, s );
	fft2d_c2r( &Op->p_kernel[2], d_z, d_ba, 0, &ratio, s );
}
void fftconv_releaseOp( fftconvOp_t* Op )
{
	if(Op->p_kernel!=NULL){
		free(Op->p_kernel);
		Op->p_kernel=NULL;
	}
}