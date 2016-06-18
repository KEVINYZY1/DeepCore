#ifndef __dnnCore_h__
#define __dnnCore_h__

#if defined(_MSC_VER)&&(defined(_WIN32)||defined(_WIN64))
	#if defined(_DLL)||defined(_WINDLL)
		#define __APIENTRY__ __declspec(dllexport)
	#else
		#define __APIENTRY__ __declspec(dllimport)
	#endif
	#if	defined(_M_X64)||defined(_M_AMD64)||defined(__x86_64)||defined(_M_IA64)||defined(__LP64__)
		#define __APICALL__
	#else
		#define __APICALL__ __stdcall
	#endif
#else
	#define __APIENTRY__
	#define __APICALL__
#endif

#include<stdint.h>
#include<cuda.h>

#define dnnPrecisionMaskFloat		0x0
#define dnnPrecisionMaskMixed		0x1
#define dnnPrecisionMaskHalf		0x2

#define dnnTensorTypeMaskFilter		0x4
#define dnnTensorTypeMaskBias		0x5
#define dnnTensorTypeMaskFullConnection	0x6

#define dnnPoolingMaskAvg		0x0
#define dnnPoolingMaskMax		0x4

#define	dnnActivationMaskReLU		0x0
#define	dnnActivationMaskTanh		0x6
#define	dnnActivationMaskSigmod		0x7

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct __dnnConvOp		*	dnnConvOp;
typedef struct __dnnFFTConvOp		*	dnnFFTConvOp;
typedef struct __dnnCellConvOp		*	dnnCellConvOp;
typedef struct __dnnPoolingOp		*	dnnPoolingOp;
typedef struct __dnnBiasOp		*	dnnBiasOp;
typedef struct __dnnActivationOp	*	dnnActivationOp;

typedef enum dnnStatus{
	dnnSuccess=0			,
	dnnErrorInvalidValue		,
	dnnErrorInvalidDevice		,
	dnnErrorOutOfMaxSize		,
	dnnErrorOutOfMemory		,
	dnnErrorOutOfDeviceMemory	
} dnnStatus_t;

__APIENTRY__	dnnStatus_t	__APICALL__ dnnInit();
__APIENTRY__	dnnStatus_t	__APICALL__ dnnSetDevice( int );

__APIENTRY__	uint64_t	__APICALL__ dnnCreateTensorShape( unsigned int, unsigned int, unsigned int, unsigned int, unsigned int );
__APIENTRY__	size_t		__APICALL__ dnnTensorPitch( uint64_t );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnCreateTensor( void**, uint64_t );

__APIENTRY__	dnnStatus_t	__APICALL__ dnnCreateConvOp( dnnConvOp*, int, uint64_t, uint64_t, int, int );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnCreateFFTConvOp( dnnFFTConvOp*, size_t*, int, unsigned int, uint64_t, uint64_t );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnCreateCellConvOp( dnnCellConvOp*, size_t*, int, unsigned int, uint64_t, uint64_t );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnCreatePoolingOp( dnnPoolingOp*, int, unsigned int, uint64_t, int, int );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnCreateBiasOp( dnnBiasOp*, int, unsigned int, uint64_t );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnCreateActivationOp( dnnActivationOp*, int, unsigned int );

__APIENTRY__	dnnStatus_t	__APICALL__ dnnConv( dnnConvOp, void*, const void*, const void*, float, int, CUstream );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnFFTConv( dnnFFTConvOp, void*, void*, const void*, const void*, float, CUstream );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnCellConv( dnnCellConvOp, void*, void*, const void*, const void*, float, CUstream );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnPooling( dnnPoolingOp, void*, const void*, int, CUstream );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnUpdateBias( dnnBiasOp, void*, const void*, float, CUstream );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnActivateFP( dnnActivationOp, void*, void*, const void*, uint64_t, float, CUstream );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnActivateBP( dnnActivationOp, void*, const void*, const void*, uint64_t, float, CUstream );

__APIENTRY__	dnnStatus_t	__APICALL__ dnnDestroyConvOp( dnnConvOp );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnDestroyFFTConvOp( dnnFFTConvOp );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnDestroyCellConvOp( dnnCellConvOp );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnDestroyPoolingOp( dnnPoolingOp );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnDestroyBiasOp( dnnBiasOp );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnDestroyActivationOp( dnnActivationOp );
__APIENTRY__	dnnStatus_t	__APICALL__ dnnExit();

#ifdef __cplusplus
}
#endif

#endif
