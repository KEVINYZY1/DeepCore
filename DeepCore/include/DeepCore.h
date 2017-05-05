#ifndef __DeepCore_h__
#define __DeepCore_h__

#if defined(_MSC_VER)&&(defined(_WIN32)||defined(_WIN64))
	#ifdef _DLL
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

#define dcMaskDirectionForward          0x00000000
#define dcMaskDirectionBackward         0x00000001

#define dcMaskPrecisionFloat            0x00000000
#define dcMaskPrecisionHalf             0x00000002
#define dcMaskPrecisionMixed            0x00000004

#define dcMaskConvFused					0x00000008
#define dcMaskConvBias					0x00000008
#define dcMaskConvBatchNormalization	0x00000010

#define	dcMaskActivationRelu            0x01000000
#define	dcMaskActivationElu             0x02000000
#define	dcMaskActivationTanh            0x03000000
#define	dcMaskActivationSigm            0x04000000

#define dcMaskPoolingAvg                0x00000000
#define dcMaskPoolingMax                0x00000002

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct __dc_convOp      *  dc_convOp;
typedef struct __dc_fftconvOp	*  dc_fftconvOp;
typedef struct __dc_cellconvOp	*  dc_cellconvOp;
typedef struct __dc_poolingOp   *  dc_poolingOp;

typedef enum dc_status{
	dc_success=0                 ,
	dc_error_invalid_value       ,
	dc_error_invalid_device      ,
	dc_error_out_of_range        ,
	dc_error_out_of_maxsize      ,
	dc_error_out_of_memory       ,
	dc_error_out_of_device_memory
} dc_status_t;

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_init();
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_device_count();
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_superdevice_count();
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_superdevice_size( int );
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_superdevice_device_id( int, int );
DEEPCOREAPIENTRY int         DEEPCOREAPICALL dc_get_optimal_superdevice_id();
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_set_device( int );

DEEPCOREAPIENTRY uint64_t    DEEPCOREAPICALL dc_create_tensor_shape( uint32_t, uint32_t, uint32_t );
DEEPCOREAPIENTRY uint64_t    DEEPCOREAPICALL dc_create_tensor_shape_filter( uint32_t, uint32_t, uint32_t );
DEEPCOREAPIENTRY uint64_t    DEEPCOREAPICALL dc_create_tensor_shape_bias( uint32_t, uint32_t );
DEEPCOREAPIENTRY uint64_t    DEEPCOREAPICALL dc_create_tensor_shape_fc( uint32_t, uint32_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_tensor( void**, uint64_t );

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_convOp( dc_convOp*, int, uint32_t, uint64_t, uint64_t, uint32_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_fftconvOp( dc_fftconvOp*, size_t*, int, uint32_t, uint64_t, uint64_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_cellconvOp( dc_cellconvOp*, size_t*, int, uint32_t, uint64_t, uint64_t, uint32_t );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_create_poolingOp( dc_poolingOp*, int, uint32_t, uint64_t, uint32_t, uint32_t );

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_conv( dc_convOp, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_conv_relu( dc_convOp, void*, const void*, const void*, const void*, float, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv( dc_fftconvOp, void*, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv_relu( dc_fftconvOp, void*, void*, const void*, const void*, const void*, float, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_fftconv_grad( dc_fftconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv( dc_cellconvOp, void*, void*, const void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv_relu( dc_cellconvOp, void*, void*, const void*, const void*, const void*, float, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_cellconv_grad( dc_cellconvOp, void*, void*, const void*, const void*, float, CUstream );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_pooling( dc_poolingOp, int, void*, const void*, CUstream );

DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_convOp( dc_convOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_fftconvOp( dc_fftconvOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_cellconvOp( dc_cellconvOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_destroy_poolingOp( dc_poolingOp );
DEEPCOREAPIENTRY dc_status_t DEEPCOREAPICALL dc_exit();

#ifdef __cplusplus
}
#endif

#endif
