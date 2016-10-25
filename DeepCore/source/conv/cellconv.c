#include"../../include/conv/cellconv.h"

size_t cellconv_createOp( cellconvOp_t* Op, const cuda_context_t* p_ctx, unsigned int mask, int ds, int fs, int bat, int pnc, int qnc )
{
	unsigned int prc, dir, align, inc, onc, os, pad, d, grid, cell_size, n, n_cells, lda, ldb;
	cuda_kernel_t* p_kernel;
	size_t aux_size;

	dir=mask&0x1;
	prc=(mask>>1)&0x3;	
	pad=fs-1;
	inc=dir?qnc:pnc;
	onc=dir?pnc:qnc;
	n=ds+(dir?(pad<<1):0);
	os=n-pad;
	cell_size=n<=16?16:32;
	d=cell_size-pad;	
	grid=(os+d-1)/d;
	n_cells=bat*grid*grid;
	align=prc?32:16;
	lda=(n_cells>1)?n_cells:inc;
	lda=AFFIS(lda,align);
	ldb=AFFIS(onc,align);
	align=prc?(BASE_PITCH/2):(BASE_PITCH/4);

	{
		int is_ext=(ds!=cell_size)&(dir==0);
		int i=(n<=32)*(((cell_size>16)<<3)+((n_cells==1)<<2)+is_ext+(dir<<1))+(n>32)*(16+dir);
		int qds=ds*ds;
		int ldr=AFFIS(bat*qds,align);
		p_kernel=&Op->kfft[0];
		create_cellfft_kernel_r2c( &Op->kfft[0], p_ctx, i, prc );
		switch(i)
		{
		case 0:
		case 1:
		case 8:		
		case 9:
			is_ext=(is_ext|dir)<<1;
			cuda_kernel_sgl( p_kernel, (bat+15)>>4, inc );
			cuda_kernel_sep_i32( p_kernel, 3, bat );
			if(is_ext!=0){			
				cuda_kernel_sep_i32( p_kernel, 4, ds  );
				cuda_kernel_sep_i32( p_kernel, 5, ds  );
			}
			cuda_kernel_sep_i32( p_kernel, 4+is_ext, lda );
			cuda_kernel_sep_i32( p_kernel, 5+is_ext, qds );
			cuda_kernel_sep_i32( p_kernel, 6+is_ext, ldr );
			break;
		case  2:
		case 10:
			cuda_kernel_sgl( p_kernel, (bat+15)>>4, inc );
			cuda_kernel_sep_i32( p_kernel, 3, bat );
			cuda_kernel_sep_i32( p_kernel, 4, ds  );
			cuda_kernel_sep_i32( p_kernel, 5, ds  );
			cuda_kernel_sep_i32( p_kernel, 6, lda );
			cuda_kernel_sep_i32( p_kernel, 7, ldr );
			cuda_kernel_sep_i32( p_kernel, 8, pad );
			cuda_kernel_sep_i32( p_kernel, 9, pad );
			break;
		case  4:
		case 12:
			cuda_kernel_sgl( p_kernel, (inc+15)>>4, 1 );
			cuda_kernel_sep_i32( p_kernel, 3, inc );
			cuda_kernel_sep_i32( p_kernel, 4, lda );
			break;
		case  5:
		case  6:
		case 13:
		case 14:
			cuda_kernel_sgl( p_kernel, (inc+15)>>4, 1 );
			cuda_kernel_sep_i32( p_kernel, 3, inc );
			cuda_kernel_sep_i32( p_kernel, 4, ds  );
			cuda_kernel_sep_i32( p_kernel, 5, ds  );
			cuda_kernel_sep_i32( p_kernel, 6, lda );
			cuda_kernel_sep_i32( p_kernel, 7, ldr );
			if(dir){			
				cuda_kernel_sep_i32( p_kernel, 8, pad );
				cuda_kernel_sep_i32( p_kernel, 9, pad );
			}
			break;
		case 16:
		case 17:
			cuda_kernel_sgl( p_kernel, (n_cells+15)>>4, inc );
			cuda_kernel_sep_i32( p_kernel,  3, n_cells   );
			cuda_kernel_sep_i32( p_kernel,	4, ds        );
			cuda_kernel_sep_i32( p_kernel,	5, ds        );
			cuda_kernel_sep_i32( p_kernel,	6, lda       );
			cuda_kernel_sep_i32( p_kernel,	7, ldr       );
			cuda_kernel_sep_i32( p_kernel,	8, grid	     );
			cuda_kernel_sep_i32( p_kernel,  9, grid	     );
			cuda_kernel_sep_i32( p_kernel, 10, dir?pad:d );
			cuda_kernel_sep_i32( p_kernel, 11, dir?pad:d );
			break;
		}
	}

	{
		int ldr=AFFIS(pnc*fs*fs,align);
		int qfs=fs*fs;
		p_kernel=&Op->kfft[1];
		create_cellfft_kernel_r2c( p_kernel, p_ctx, (cell_size<32?0:8)+(dir==0?1:3), prc );
		cuda_kernel_sgl( p_kernel, (onc+15)>>4, inc );
		cuda_kernel_sep_i32( p_kernel, 3, onc	      );
		cuda_kernel_sep_i32( p_kernel, 4, fs	      );
		cuda_kernel_sep_i32( p_kernel, 5, fs	      );
		cuda_kernel_sep_i32( p_kernel, 6, ldb	      );
		cuda_kernel_sep_i32( p_kernel, 7, dir?qfs:ldr );
		cuda_kernel_sep_i32( p_kernel, 8, dir?ldr:qfs );
	}

	{
		int fused=(mask>>3)&0x1;
		int atvop=mask>>4;
		int o=fused?(atvop?5:4):(atvop?4:3);
		int radix=cell_size<32?16:8;
		int count=n_cells>1?n_cells:onc;
		int i=(cell_size<32?0:19)+18*(grid>1)+9*(n_cells==1)+fused*(dir?6:3)+atvop;
		int ldr=AFFIS(bat*os*os,align);
		p_kernel=&Op->kfft[2];
		create_cellfft_kernel_c2r( p_kernel, p_ctx, i, prc );
		cuda_kernel_sgl( p_kernel, (count+radix-1)/radix, n_cells>1?onc:1 );
		cuda_kernel_sep_i32( p_kernel, o+0, count             );
		cuda_kernel_sep_i32( p_kernel, o+1, os                );
		cuda_kernel_sep_i32( p_kernel, o+2, os                );
		cuda_kernel_sep_i32( p_kernel, o+3, ldr               );
		cuda_kernel_sep_i32( p_kernel, o+4, n_cells>1?lda:ldb );
		if(grid>1){
			cuda_kernel_sep_i32( p_kernel, o+5, grid );
			cuda_kernel_sep_i32( p_kernel, o+6, grid );
			cuda_kernel_sep_i32( p_kernel, o+7, d	 );
			cuda_kernel_sep_i32( p_kernel, o+8, d	 );
		}
	}

	n=((cell_size>>1)+1)*cell_size;
	if(n_cells>1){
		cgemm_create_kernel( &Op->kcgemm, p_ctx, prc, n, n_cells, inc, onc, lda, ldb, lda );
	} else {
		cgemv_create_kernel( &Op->kcgemm, p_ctx, prc, n, inc, onc, lda, ldb, ldb );
	}
	cuda_kernel_sep_f32( &Op->kcgemm, 3, (float)(1.0/(cell_size*cell_size)) );

	n*=(prc?4:8);
	Op->adivpt=n*(n_cells>1?inc:1)*lda;
	Op->bdivpt=n*inc*ldb;
	aux_size=Op->adivpt+Op->bdivpt+n*(n_cells>1?(onc*lda):ldb);
	return aux_size;
}
size_t cellconv_createOp_filter( cellconvOp_t* Op, const cuda_context_t* p_ctx, unsigned int mask, int pn, int pnc, int qn, int qnc, int bat )
{
	unsigned int prc, enb, align, cell_size, n, lda, ldb, i;
	cuda_kernel_t* p_kernel;
	size_t aux_size;
	
	prc=(mask>>1)&0x3;
	enb=prc?2:4;
	cell_size=(pn<=16)?16:32;
	align=prc?32:16;
	lda=AFFIS(pnc,align);
	ldb=AFFIS(qnc,align);
	align=prc?(BASE_PITCH/2):(BASE_PITCH/4);
	
	{
		int qpn=pn*pn;
		int ldr=AFFIS(bat*qpn,align);
		int is_ext=pn!=cell_size;
		i=(cell_size<=16?0:8)+(bat>1?0:4);
		p_kernel=&Op->kfft[0];
		create_cellfft_kernel_r2c( p_kernel, p_ctx, i+is_ext, prc );
		cuda_kernel_sgl( p_kernel, (pnc+15)>>4, bat );
		if(bat>1)
		{
			cuda_kernel_sep_i32( p_kernel, 3, pnc );
			if(is_ext){
				cuda_kernel_sep_i32( p_kernel, 4, pn );
				cuda_kernel_sep_i32( p_kernel, 5, pn );
			}
			cuda_kernel_sep_i32( p_kernel, 4+(is_ext<<1), lda );
			cuda_kernel_sep_i32( p_kernel, 4+(is_ext<<1), ldr );
			cuda_kernel_sep_i32( p_kernel, 4+(is_ext<<1), qpn );
		}
		else
		{
			if(is_ext){
				cuda_kernel_sep_i32( p_kernel, 3, pnc );
				cuda_kernel_sep_i32( p_kernel, 4, pn  );
				cuda_kernel_sep_i32( p_kernel, 5, pn  );
				cuda_kernel_sep_i32( p_kernel, 6, lda );
				cuda_kernel_sep_i32( p_kernel, 7, ldr );
			} else {
				cuda_kernel_sep_i32( p_kernel, 3, pnc );
				cuda_kernel_sep_i32( p_kernel, 4, lda );
			}
		}
	}

	{
		int qqn=qn*qn;
		int ldr=AFFIS(bat*qqn,align);
		p_kernel=&Op->kfft[1];
		create_cellfft_kernel_r2c( p_kernel, p_ctx, i+3, prc );
		cuda_kernel_sgl( p_kernel, (qnc+15)>>4, bat );
		cuda_kernel_sep_i32( p_kernel, 3, qnc );
		cuda_kernel_sep_i32( p_kernel, 4, qn  );
		cuda_kernel_sep_i32( p_kernel, 5, qn  );
		cuda_kernel_sep_i32( p_kernel, 6, ldb );
		cuda_kernel_sep_i32( p_kernel, 7, ldr );
		if(bat>1){
			cuda_kernel_sep_i32( p_kernel, 8, qqn );
		}
	}

	{
		int radix=cell_size<=16?16:8;
		int d=pn-qn+1;
		int ldr=AFFIS(pnc*d*d,align);
		p_kernel=&Op->kfft[2];
		create_cellfft_kernel_c2r( p_kernel, p_ctx, cell_size<=16?18:46, prc );
		cuda_kernel_sgl( p_kernel, (pnc+radix-1)/radix, qnc );
		cuda_kernel_sep_i32( p_kernel, 3, pnc );
		cuda_kernel_sep_i32( p_kernel, 4, d   );
		cuda_kernel_sep_i32( p_kernel, 5, d   );
		cuda_kernel_sep_i32( p_kernel, 6, ldr );
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
void cellconv( cellconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_target, CUdeviceptr d_source, CUdeviceptr d_filter, CUdeviceptr d_bias_or_atv, const float* alpha, CUstream s )
{
	CUdeviceptr d_a=d_aux;
	CUdeviceptr d_b=d_a+Op->adivpt;
	CUdeviceptr d_c=d_b+Op->bdivpt;	
	fft2d_r2c( &Op->kfft[0], d_a, d_source, s );
	fft2d_r2c( &Op->kfft[1], d_b, d_filter, s );
	cgemm( &Op->kcgemm, d_c, d_a, d_b, 1.f, s );
	fft2d_c2r( &Op->kfft[2], d_target, d_c, d_bias_or_atv, alpha, s );
}
void cellconv_filter( cellconvOp_t* Op, CUdeviceptr d_aux, CUdeviceptr d_z, CUdeviceptr d_x, CUdeviceptr d_y, float ratio, CUstream s )
{
	CUdeviceptr d_a=d_aux;
	CUdeviceptr d_b=d_a+Op->adivpt;
	CUdeviceptr d_c=d_b+Op->bdivpt;	
	fft2d_r2c( &Op->kfft[0], d_a, d_x, s );
	fft2d_r2c( &Op->kfft[1], d_b, d_y, s );
	cgemm( &Op->kcgemm, d_c, d_a, d_b, 1.f, s );
	fft2d_c2r( &Op->kfft[2], d_z, d_c, 0, &ratio, s );
}
