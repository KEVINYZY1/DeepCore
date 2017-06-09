
#include<stdio.h>
#include<stdlib.h>
#include"cudnn.h"
#include"deepcore.h"
#pragma comment( lib, "cudart.lib" )
#pragma comment( lib, "cudnn.lib" )
#pragma comment( lib, "deepcore.lib" )
#pragma comment( lib, "cuda.lib" )

int main()
{
	cudnnTensorDescriptor_t p_shape, q_shape;
	cudnnFilterDescriptor_t w_shape;	
	cudnnConvolutionDescriptor_t conv_desc;
	CUdeviceptr d_p, d_w, d_q,d_aux;
	float dt0, dt1;
	size_t auxnb;

	int  fil_size[]={3};
	int  dat_size[]={56,48,40,24,12,6};
	int  bat_size[]={1,2,4,8,16,32,64};
	int2 chl_size[]={{32,32},{32,64},{32,96},{64,64},{64,96},{64,128},{64,192},{128,128},{128,192},{128,256},{128,384},{256,256}};
	//int fil_size[]={5};
	//int dat_size[]={224,112,56};
	//int bat_size[]={64};
	//int chl_size[]={32,64};
	int n_fil=sizeof(fil_size)/sizeof(fil_size[0]);
	int n_dat=sizeof(dat_size)/sizeof(dat_size[0]);
	int n_chl=sizeof(chl_size)/sizeof(chl_size[0]);
	int n_bat=sizeof(bat_size)/sizeof(bat_size[0]);
	FILE* fp=fopen( "result.txt", "wt" );

	dc_cellconvOp fftconvOp;
	if(dc_init()!=dc_success){
		printf( "error : deepcore init failed!\n" );
		exit(0);
	}
	dc_set_device(0);

	cudnnHandle_t handle;
	cudnnCreate( &handle );
	cudnnCreateTensorDescriptor( &p_shape );
	cudnnCreateTensorDescriptor( &q_shape );	
	cudnnCreateFilterDescriptor( &w_shape );
	cudnnCreateConvolutionDescriptor( &conv_desc );
	float alpha=1.f, beta=0.f;
	CUevent se, ee;
	cuEventCreate( &se, CU_EVENT_BLOCKING_SYNC );
	cuEventCreate( &ee, CU_EVENT_BLOCKING_SYNC );

	int mask=dcMaskDirectionBackward;
	for( int i_fil=0; i_fil<n_fil; ++i_fil )
	{
		int fs=fil_size[i_fil];
		int pad=fs-1;
		for( int i_dat=0; i_dat<n_dat; ++i_dat )
		{
			int ps=dat_size[i_dat];
			int qs=ps-fs+1;
			for( int i_bat=0; i_bat<n_bat; ++i_bat )
			{
				int bat=bat_size[i_bat];
				for( int i_nc=0; i_nc<n_chl; ++i_nc )
				{
					int pnc=chl_size[i_nc].x;
					int qnc=chl_size[i_nc].y;
					cudnnSetTensor4dDescriptor( p_shape, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bat, pnc, ps, ps );
					cudnnSetTensor4dDescriptor( q_shape, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bat, qnc, qs+2*pad, qs+2*pad );
					cudnnSetFilter4dDescriptor( w_shape, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, qnc, pnc, fs, fs );
					cudnnSetConvolution2dDescriptor( conv_desc, pad, pad, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT );
					cudnnStatus_t s;
					s=cudnnGetConvolutionBackwardDataWorkspaceSize( handle, w_shape, q_shape, conv_desc, p_shape, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING, &auxnb );
					if(s!=CUDNN_STATUS_SUCCESS){
						printf( "error : %s\n", cudnnGetErrorString(s));
					}
					cuMemAlloc(&d_p, bat*pnc*ps*ps*sizeof(float));
					cuMemAlloc(&d_q, bat*qnc*qs*qs*sizeof(float));
					cuMemAlloc(&d_w, qnc*pnc*fs*fs*sizeof(float));
					if(auxnb!=0){
						cuMemAlloc(&d_aux, auxnb);
					}
					cuEventRecord( se, NULL );
					for( int i=0; i<5; ++i ){
						s=cudnnConvolutionBackwardData( handle, &alpha, w_shape, (const void*)d_w, q_shape, (const void*)d_q, conv_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING, (void*)d_aux, auxnb, &beta, p_shape, (void*)d_p );
						if(s!=CUDNN_STATUS_SUCCESS){
							printf( "error : %s\n", cudnnGetErrorString(s));
							break;
						}
					}
					cuEventRecord( ee, NULL );
					cuEventSynchronize( ee );
					cuEventElapsedTime( &dt0, se, ee );
					cuMemFree(d_p);
					cuMemFree(d_q);
					cuMemFree(d_w);
					cuMemFree(d_aux);
					
					uint64_t s_p=dc_create_tensor_shape( dcMaskPrecisionFloat, (ps<<16)|ps, (pnc<<16)|bat );
					uint64_t s_q=dc_create_tensor_shape( dcMaskPrecisionFloat, (qs<<16)|qs, (qnc<<16)|bat );
					uint64_t s_w=dc_create_tensor_shape_filter( dcMaskPrecisionFloat, (fs<<16)|fs, (qnc<<16)|pnc );
					if(dc_create_cellconvOp( &fftconvOp, &auxnb, mask, s_p, s_w, s_q, (pad<<16)|pad )!=dc_success){
						printf( "error : Op create failed!\n" );
						goto __exit__;
					}
					if(dc_create_tensor( (void**)&d_p, s_p )!=dc_success){
						printf( "error : [%d][%d][%d][%d] : p tensor create failed!\n", i_fil, i_dat, i_bat, i_nc );
						fclose(fp);
						exit(0);
					}
					if(dc_create_tensor( (void**)&d_q, s_q )!=dc_success){
						printf( "error : [%d][%d][%d][%d] : q tensor create failed!\n", i_fil, i_dat, i_bat, i_nc );
						fclose(fp);
						exit(0);
					}
					if(dc_create_tensor( (void**)&d_w, s_w )!=dc_success){
						printf( "error : [%d][%d][%d][%d] : w tensor create failed!\n", i_fil, i_dat, i_bat, i_nc );
						fclose(fp);
						exit(0);
					}
					cuMemAlloc(&d_aux,auxnb);
					cudaEventRecord( se, NULL );
					for( int i=0; i<5; ++i ){
						dc_cellconv( fftconvOp, (void*)d_aux, (void*)d_p, (const void*)d_q, (const void*)d_w, NULL, 1.f, NULL );
					}
					cuEventRecord( ee, NULL );
					cuEventSynchronize( ee );
					cuEventElapsedTime( &dt1, se, ee );
					if(cuCtxSynchronize()!=CUDA_SUCCESS){
						printf( "error : [%d][%d][%d][%d] : sample exec failed!\n", i_fil, i_dat, i_bat, i_nc );
						dc_destroy_cellconvOp(fftconvOp);
						goto __exit__;
					}
					fprintf( fp, "[fil_size=%d][dat_size=%d][bat=%d][inc=%d][onc=%d] : %f\n", fs, ps, bat, qnc, pnc, dt0/dt1 );
					dc_destroy_cellconvOp(fftconvOp);
					cuMemFree(d_p);
					cuMemFree(d_q);
					cuMemFree(d_w);
					cuMemFree(d_aux);
				}
			}
		}
	}
__exit__:
	fclose(fp);
	cudnnDestroyTensorDescriptor( p_shape );
	cudnnDestroyTensorDescriptor( q_shape );
	cudnnDestroyFilterDescriptor( w_shape );
	cudnnDestroyConvolutionDescriptor( conv_desc );
	cudnnDestroy( handle );
	dc_exit();
}