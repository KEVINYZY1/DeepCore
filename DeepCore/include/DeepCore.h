#ifndef __DeepCore_h__
#define __DeepCore_h__

#if defined(_MSC_VER)&&(defined(_WIN32)||defined(_WIN64))
	#if defined(_DLL)||defined(_WINDLL)
		#define DEEPCOREAPIENTRY __declspec(dllexport)
	#else
		#define DEEPCOREAPIENTRY __declspec(dllimport)
	#endif
	#if	defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
		#define DEEPCOREAPICALL
	#else
		#define DEEPCOREAPICALL __stdcall
	#endif
#else
	#define DEEPCOREAPIENTRY
	#define DEEPCOREAPICALL
#endif

#include<stdint.h>
#include<cuda.h>

#define dcMaskPrecisionFloat              0x0
#define dcMaskPrecisionMixed              0x2
#define dcMaskPrecisionHalf               0x6

#define dcMaskTensorTypeFilter            0x4
#define dcMaskTensorTypeBias              0x5
#define dcMaskTensorTypeFullConnection    0x6

#define dcMaskConvFused					  0x8
#define dcMaskConvFilter                  0x10

#define	dcMaskActivationRelu              0x10
#define	dcMaskActivationElu               0x11

#define dcMaskPoolingAvg                  0x0
#define dcMaskPoolingMax                  0x4

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct __dc_convOp      *  dc_convOp;
typedef struct __dc_fftconvOp	*  dc_fftconvOp;
typedef struct __dc_cellconvOp	*  dc_cellconvOp;
typedef struct __dc_poolingOp   *  dc_poolingOp;
typedef struct __dc_biasOp      *  dc_biasOp;
typedef struct __dc_activationOp*  dc_activationOp;

typedef enum dc_status{
	dc_success=0                 ,
	dc_error_invalid_value       ,
	dc_error_invalid_device      ,
	dc_error_out_of_maxsize      ,
	dc_error_out_of_memory       ,
	dc_error_out_of_device_memory,
	dc_cuda_status				
} dc_status_t;

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_init();
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_set_device( int );
DEEPCOREAPIENTRY uint64_t    DEEPCOREAPICALL dc_create_tensor_shape( unsigned int, int, int, int );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_tensor( void**, uint64_t );

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_convOp( dc_convOp*, int, unsigned int, uint64_t, uint64_t, int );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_fftconvOp( dc_fftconvOp*, size_t*, int, unsigned int, uint64_t, uint64_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_cellconvOp( dc_cellconvOp*, size_t*, int, unsigned int, uint64_t, uint64_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_poolingOp( dc_poolingOp*, int, unsigned int, uint64_t, int, int );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_biasOp( dc_biasOp*, size_t*, int, uint64_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_activationOp( dc_activationOp*, int, unsigned int, uint64_t );
				 			 
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_conv( dc_convOp, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv( dc_fftconvOp, void*, void*, const void*, const void*, const void*, const float*, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv_filter( dc_fftconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv( dc_cellconvOp, void*, void*, const void*, const void*, const void*, const float*, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv_filter( dc_cellconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_pooling( dc_poolingOp, void*, const void*, int, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_update_bias( dc_biasOp, void*, void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_activate_fprop( dc_activationOp, void*, void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_activate_bprop( dc_activationOp, void*, const void*, const void*, const void*, float, CUstream );
				 			 
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_convOp( dc_convOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_fftconvOp( dc_fftconvOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_cellconvOp( dc_cellconvOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_poolingOp( dc_poolingOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_biasOp( dc_biasOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_activationOp( dc_activationOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_exit();

#ifdef __cplusplus
}
#endif

#endif
