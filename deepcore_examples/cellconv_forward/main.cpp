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

	tensor_shape_t shapes[]={{48,3,32,64,8},{40,3,32,64,1},{24,3,32,64,32},{18,3,32,32,1},{10,3,32,32,32}};

	for( int e=0; e<sizeof(shapes)/sizeof(shapes[0]); e++ )
	{
		int pn=shapes[e].ds;
		int fn=shapes[e].fs;
		int qn=pn-fn+1;
		int pnc=shapes[e].pnc;
		int qnc=shapes[e].qnc;
		int bat=shapes[e].bat;
		uint64_t qshape=dc_create_tensor_shape( dcMaskPrecisionFloat, (qn<<16)|qn, (qnc<<16)|bat );
		uint64_t pshape=dc_create_tensor_shape( dcMaskPrecisionFloat, (pn<<16)|pn, (pnc<<16)|bat );
		uint64_t kshape=dc_create_tensor_shape_filter( dcMaskPrecisionFloat, (fn<<16)|fn, (qnc<<16)|pnc );

		dc_cellconvOp Op;
		size_t auxnb;
		if(dc_create_cellconvOp( &Op, &auxnb, dcMaskPrecisionFloat|dcMaskDirectionForward, pshape, kshape, qshape, 0 )!=dc_success){
			printf( "error : cellconvOp create failed!\n" );
			dc_exit();
			return 0;
		}

		void *d_a, *d_b, *d_c;
		CUdeviceptr auxbuf;
		dc_create_tensor( (void**)&d_a, pshape );
		dc_create_tensor( (void**)&d_b, kshape );
		dc_create_tensor( (void**)&d_c, qshape );
		cuMemAlloc( &auxbuf, auxnb );
		
		float* a=new float[bat*pnc*pn*pn];
		float* b=new float[qnc*pnc*fn*fn];
		float* c=new float[bat*qnc*qn*qn];
		float* d=new float[bat*qnc*qn*qn];

		for( int i=0; i<pnc; ++i )
		{
			for( int z=0; z<bat; ++z ){
				for( int y=0; y<pn; ++y ){
					for( int x=0; x<pn; ++x ){
						a[((i*bat+z)*pn+y)*pn+x]=((float)rand())/RAND_MAX;
					}
				}
			}
		}
		for( int i=0; i<fn*fn*pnc*qnc; ++i ){
			b[i]=((float)rand())/RAND_MAX;
		}

		for( int i=0; i<qnc; ++i ){
			for( int s=0; s<bat; ++s ){
				conv( &c[(i*bat+s)*qn*qn], &a[s*pn*pn], &b[i*pnc*fn*fn], 0, false, pn, pn, fn, fn, qn, qn, pnc, bat, 0, 0, fn*fn );
			}
		}

		dc_tensor_store( d_a, pshape, a, bat*pn*pn*sizeof(float), bat*pn*pn*sizeof(float), pnc, NULL );
		dc_tensor_store( d_b, kshape, b, pnc*fn*fn*sizeof(float), pnc*fn*fn*sizeof(float), qnc, NULL );
		if(dc_cellconv( Op, (void*)auxbuf, d_c, d_a, d_b, NULL, 1.f, NULL )!=dc_success){
			printf( "error: conv exec failed!\n" );
		}
		dc_tensor_load( d, bat*qn*qn*sizeof(float), d_c, qshape, bat*qn*qn*sizeof(float), qnc, NULL );
		cuCtxSynchronize();

		bool is_ok=check( c, d, bat*qnc*qn*qn );
		if(!is_ok){
			printf( "examples[%d] is compute failed!\n", e );
			goto __LAB0;
		}

	__LAB0:
		dc_release_tensor( d_a );
		dc_release_tensor( d_b );
		dc_release_tensor( d_c );
		dc_destroy_cellconvOp(Op);
		cuMemFree(auxbuf);
		delete[] a;
		delete[] b;
		delete[] c;
		delete[] d;
		if(!is_ok) break;
	}
	dc_exit();
}