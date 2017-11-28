#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"../common/common.h"
#include"../deepcore/deepcore.h"
#pragma comment( lib, "../deepcore/deepcore.lib" )
#pragma comment( lib, "cuda.lib" )


int main()
{
	if(dc_init()!=dc_success){
		printf( "error : deepcore init failed!\n" );
		exit(0);
	}
	dc_set_device(0);

	tensor_shape_t shapes[]={{56,3,32,64,8},{24,3,32,64,1},{10,3,32,32,1},{4,3,32,96,16}};

	for( int e=0; e<sizeof(shapes)/sizeof(shapes[0]); e++ )
	{
		int pn=shapes[e].ds;
		int fn=shapes[e].fs;
		int qn=pn-fn+1;
		int pnc=shapes[e].pnc;
		int qnc=shapes[e].qnc;
		int bat=shapes[e].bat;
		int pad=fn-1;
		uint64_t qshape=dc_create_tensor_shape( dcMaskPrecisionFloat, (qn<<16)|qn, (qnc<<16)|bat );
		uint64_t pshape=dc_create_tensor_shape( dcMaskPrecisionFloat, (pn<<16)|pn, (pnc<<16)|bat );
		uint64_t kshape=dc_create_tensor_shape_filter( dcMaskPrecisionFloat, (fn<<16)|fn, (qnc<<16)|pnc );

		dc_fftconvOp Op;
		size_t auxnb;
		if(dc_create_fftconvOp( &Op, &auxnb, dcMaskPrecisionFloat|dcMaskDirectionBackward, pshape, kshape, qshape, (pad<<8)|pad )!=dc_success){
			printf( "error : cellconvOp create failed!\n" );
			dc_exit();
			return 0;
		}

		void *d_a, *d_b, *d_c;
		CUdeviceptr auxbuf;
		dc_create_tensor( (void**)&d_a, qshape );
		dc_create_tensor( (void**)&d_b, kshape );
		dc_create_tensor( (void**)&d_c, pshape );
		cuMemAlloc( &auxbuf, auxnb );
		
		float* a=new float[bat*qnc*qn*qn];
		float* b=new float[pnc*qnc*fn*fn];
		float* c=new float[bat*pnc*pn*pn];
		float* d=new float[bat*pnc*pn*pn];

		for( int i=0; i<qnc; ++i )
		{
			for( int z=0; z<bat; ++z ){
				for( int y=0; y<qn; ++y ){
					for( int x=0; x<qn; ++x ){
						a[((i*bat+z)*qn+y)*qn+x]=((float)rand())/RAND_MAX;
					}
				}
			}
		}
		for( int i=0; i<fn*fn*pnc*qnc; ++i ){
			b[i]=((float)rand())/RAND_MAX;
		}

		for( int i=0; i<pnc; ++i ){
			for( int s=0; s<bat; ++s ){
				conv( &c[(i*bat+s)*pn*pn], &a[s*qn*qn], &b[i*fn*fn], 1, true, qn, qn, fn, fn, pn, pn, qnc, bat, pad, pad, pnc*fn*fn );
			}
		}

		dc_tensor_store( d_a, qshape, a, bat*qn*qn*sizeof(float), bat*qn*qn*sizeof(float), qnc, NULL );
		dc_tensor_store( d_b, kshape, b, pnc*fn*fn*sizeof(float), pnc*fn*fn*sizeof(float), qnc, NULL );
		if(dc_fftconv( Op, (void*)auxbuf, d_c, d_a, d_b, NULL, 1.f, NULL )!=dc_success){
			printf( "error: conv exec failed!\n" );
		}
		dc_tensor_load( d, bat*pn*pn*sizeof(float), d_c, pshape, bat*pn*pn*sizeof(float), pnc, NULL );
		cuCtxSynchronize();

		bool is_ok=check( c, d, bat*pnc*pn*pn );
		if(!is_ok){
			printf( "examples[%d] is compute failed!\n", e );
			goto __LAB0;
		}

	__LAB0:
		dc_release_tensor( d_a );
		dc_release_tensor( d_b );
		dc_release_tensor( d_c );
		dc_destroy_fftconvOp(Op);
		cuMemFree(auxbuf);
		delete[] a;
		delete[] b;
		delete[] c;
		delete[] d;
		if(!is_ok) break;
	}
	dc_exit();
}