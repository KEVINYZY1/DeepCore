#include"../../include/conv/cellconv.h"

size_t cellconv_createOp( cellconvOp_t* Op, const cuda_context_t* p_ctx, unsigned int mask, int ds, int fs, int bat, int pnc, int qnc )
{
	unsigned int prc, dir, align, inc, onc, os, pad, d, grid, cell_size, n, n_cells, lda, ldb;
	cuda_kernel_t* p_kernel;
	size_t aux_size;

	prc=mask&0x3;
	dir=(mask>>2)&0x1;
	pad=dir?(fs-1):0;
	inc=dir?qnc:pnc;
	onc=dir?pnc:qnc;
	ds+=(pad<<1);
	os=ds-fs+1;
	cell_size=(ds<=16)?16:32;
	d=cell_size-pad;	
	grid=(os+d-1)/d;
	n_cells=bat*grid*grid;
	n=((cell_size>>1)+1)*cell_size;
	align=prc?32:16;
	lda=(n_cells>1)?n_cells:inc;
	lda=AFFIS(lda,align);
	ldb=AFFIS(onc,align);
	align=p_ctx->align/(prc?2:4);

	{
		int is_ext=(ds!=cell_size)&(dir==0);
		int i=(ds<=32)*(4*(n_cells==1)+is_ext+(dir<<1))+(ds>32)*(8+dir);	
		int lds=AFFIS(bat*ds*ds,align);
		int qds=ds*ds;
		p_kernel=&Op->kfft[0];
		create_cellfft_kernel_r2c( &Op->kfft[0], p_ctx, (cell_size>16?8:0)+i, prc );
		cuda_kernel_sgl( p_kernel, ((bat>1?bat:inc)+15)>>4, bat>1?inc:1 );
		if(bat>1)
		{				
			cuda_kernel_sep_i32( p_kernel, 3, dir?inc:bat );
			cuda_kernel_sep_i32( p_kernel, 4, ds	      );
			cuda_kernel_sep_i32( p_kernel, 5, ds	      );
			cuda_kernel_sep_i32( p_kernel, 6, lda	      );
			if((ds<=32)&(pad==0)){
				cuda_kernel_sep_i32( p_kernel, 7, dir?qds:lds );
				cuda_kernel_sep_i32( p_kernel, 8, dir?lds:qds );
			} else {
				cuda_kernel_sep_i32( p_kernel, 7, grid>1?grid:pad );
				if(grid>1){
					cuda_kernel_sep_i32( p_kernel, 8, pad?pad:d );
				}
			}
		}
		else
		{
			cuda_kernel_sep_i32( p_kernel, 3, inc );
			cuda_kernel_sep_i32( p_kernel, 4, lda );
			if(ds!=cell_size){
				cuda_kernel_sep_i32( p_kernel, 5, lds );
			}
			if(pad!=0){
				cuda_kernel_sep_i32( p_kernel, 6, pad );
			}
		}
	}

	{
		int lds=AFFIS(pnc*fs*fs,align);
		int qfs=fs*fs;
		p_kernel=&Op->kfft[1];
		create_cellfft_kernel_r2c( p_kernel, p_ctx, (cell_size>16?8:0)+(dir==0?1:3), prc );
		cuda_kernel_sgl( p_kernel, (onc+15)>>4, inc );
		cuda_kernel_sep_i32( p_kernel, 3, onc	      );
		cuda_kernel_sep_i32( p_kernel, 4, fs	      );
		cuda_kernel_sep_i32( p_kernel, 5, fs	      );
		cuda_kernel_sep_i32( p_kernel, 6, ldb	      );
		cuda_kernel_sep_i32( p_kernel, 7, dir?qfs:lds );
		cuda_kernel_sep_i32( p_kernel, 8, dir?lds:qfs );
	}

	{
		int fused=(mask>>3)&0x1;
		int acti_op=mask>>4;
		int o=fused?4:3;
		int radix=cell_size>16?8:16;
		int naxis=n_cells>1?n_cells:onc;
		int i=19*(cell_size>16)+18*(grid>1)+9*(n_cells==1)+fused*(dir?6:3)+acti_op;
		p_kernel=&Op->kfft[2];
		create_cellfft_kernel_c2r( p_kernel, p_ctx, i, prc );
		cuda_kernel_sgl( p_kernel, (naxis+radix-1)/radix, n_cells>1?onc:1 );
		cuda_kernel_sep_i32( p_kernel, o+0, naxis		  );
		cuda_kernel_sep_i32( p_kernel, o+1, os			  );
		cuda_kernel_sep_i32( p_kernel, o+2, os			  );
		cuda_kernel_sep_i32( p_kernel, o+3, AFFIS(bat*os*os,align));
		cuda_kernel_sep_i32( p_kernel, o+4, n_cells>1?lda:ldb	  );
		if(grid>1){
			cuda_kernel_sep_i32( p_kernel, o+5, grid );
			cuda_kernel_sep_i32( p_kernel, o+6, d	 );
		}
	}

	if(n_cells>1){
		cgemm_create_kernel( &Op->kcgemm, p_ctx, prc, n, n_cells, inc, onc, lda, ldb, lda );
	} else {
		cgemv_create_kernel( &Op->kcgemm, p_ctx, prc, n, inc, onc, lda, ldb, ldb );
	}
	cuda_kernel_sep_f32( &Op->kcgemm, 3, 1.f/(cell_size*cell_size) );

	n*=(prc?4:8);
	Op->adivpt=n*(n_cells>1?inc:1)*lda;
	Op->bdivpt=n*inc*ldb;
	aux_size=Op->adivpt+Op->bdivpt+n*(n_cells>1?(onc*lda):ldb);
	return aux_size;
}
size_t cellconv_createOp_filter( cellconvOp_t* Op, const cuda_context_t* p_ctx, unsigned int mask, int pn, int pnc, int qn, int qnc, int bat )
{
	unsigned int prc, align, cell_size, n, lda, ldb, i;
	cuda_kernel_t* p_kernel;
	size_t aux_size;
	
	prc=mask&0x3;
	cell_size=(pn<=16)?16:32;
	align=prc?32:16;
	lda=AFFIS(pnc,align);
	ldb=AFFIS(qnc,align);
	align=p_ctx->align/(prc?2:4);
	
	{
		int qpn=pn*pn;
		i=(cell_size>16?8:0)+(bat==1?4:0);
		p_kernel=&Op->kfft[0];
		create_cellfft_kernel_r2c( p_kernel, p_ctx, i+(pn!=cell_size), prc );
		cuda_kernel_sgl( p_kernel, (pnc+15)>>4, bat );
		cuda_kernel_sep_i32( p_kernel, 3, pnc );
		cuda_kernel_sep_i32( p_kernel, 4, pn  );
		cuda_kernel_sep_i32( p_kernel, 5, pn  );
		cuda_kernel_sep_i32( p_kernel, 6, lda );
		cuda_kernel_sep_i32( p_kernel, 7, AFFIS(bat*qpn,align) );
		if(bat>1){
			cuda_kernel_sep_i32( p_kernel, 8, qpn );
		}
	}

	{
		int qqn=qn*qn;
		p_kernel=&Op->kfft[1];
		create_cellfft_kernel_r2c( p_kernel, p_ctx, i+3, prc );
		cuda_kernel_sgl( p_kernel, (qnc+15)>>4, bat );
		cuda_kernel_sep_i32( p_kernel, 3, qnc );
		cuda_kernel_sep_i32( p_kernel, 4, qn  );
		cuda_kernel_sep_i32( p_kernel, 5, qn  );
		cuda_kernel_sep_i32( p_kernel, 6, ldb );
		cuda_kernel_sep_i32( p_kernel, 7, AFFIS(bat*qqn,align) );
		if(bat>1){
			cuda_kernel_sep_i32( p_kernel, 8, qqn );
		}
	}

	{
		int radix=cell_size<=16?16:8;
		int d=pn-qn+1;
		p_kernel=&Op->kfft[2];
		create_cellfft_kernel_c2r( p_kernel, p_ctx, cell_size<=16?18:46, prc );
		cuda_kernel_sgl( p_kernel, (pnc+radix-1)/radix, qnc );
		cuda_kernel_sep_i32( p_kernel, 3, pnc );
		cuda_kernel_sep_i32( p_kernel, 4, d   );
		cuda_kernel_sep_i32( p_kernel, 5, d   );
		cuda_kernel_sep_i32( p_kernel, 6, AFFIS(pnc*d*d,align));
		cuda_kernel_sep_i32( p_kernel, 7, lda );
	}

	n=((cell_size>>1)+1)*cell_size;
	if(bat>1){
		cgemm_create_kernel( &Op->kcgemm, p_ctx, prc, n, pnc, bat, qnc, lda, ldb, lda );
	} else {
		cgevv_create_kernel( &Op->kcgemm, p_ctx, prc, n, pnc, qnc, lda, ldb, lda );
	}
	cuda_kernel_sep_f32( &Op->kcgemm, 3, (float)(1.0/(bat*cell_size*cell_size)) );

	n*=(prc?4:8);
	Op->adivpt=n*bat*lda;
	Op->bdivpt=n*bat*ldb;
	aux_size=Op->adivpt+Op->bdivpt+n*qnc*lda;
	return aux_size;
}
void cellconv( cellconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_boa, float alpha, float* p_beta, CUstream s )
{
	CUdeviceptr d_a=d_aux;
	CUdeviceptr d_b=d_a+Op->adivpt;
	CUdeviceptr d_c=d_b+Op->bdivpt;	
	fft2d( &Op->kfft[0], d_a, d_source, s );
	fft2d( &Op->kfft[1], d_b, d_filter, s );
	cgemm( &Op->kcgemm, d_c, d_a, d_b, alpha, s );
	if(d_boa!=0){ cuda_kernel_sep_ptr( &Op->kfft[2], 3, d_boa ); }
	if(p_beta!=NULL){ cuda_kernel_sep_f32( &Op->kfft[2], 9+(d_boa!=0), *p_beta ); }
	fft2d( &Op->kfft[2], d_target, d_c, s );
}
void cellconv_filter( cellconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_z, CUdeviceptr d_x, CUdeviceptr d_y, float alpha, CUstream s )
{
	CUdeviceptr d_a=d_aux;
	CUdeviceptr d_b=d_a+Op->adivpt;
	CUdeviceptr d_c=d_b+Op->bdivpt;	
	fft2d( &Op->kfft[0], d_a, d_x, s );
	fft2d( &Op->kfft[1], d_b, d_y, s );
	cgemm( &Op->kcgemm, d_c, d_a, d_b, alpha, s );
	fft2d( &Op->kfft[2], d_z, d_c, s );
}
