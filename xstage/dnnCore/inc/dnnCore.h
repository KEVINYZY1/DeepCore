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

#ifndef uint64
	#ifdef _MSC_VER
		#define uint64 unsigned __int64
	#else
		#define uint64 unsigned long long
	#endif
#endif

#include<stdlib.h>
#include<cuda.h>

#define dnnPrecisionMaskFloat			0x0
#define dnnPrecisionMaskMixed			0x1
#define dnnPrecisionMaskHalf			0x2

#define dnnTensorTypeMaskGeneral		0x0
#define dnnTensorTypeMaskFilter			0x4
#define dnnTensorTypeMaskFullConnection		0xc

#define dnnPoolingMaskAvg			0x0
#define dnnPoolingMaskMax			0x4

#define	dnnActivationMaskReLU			0x0
#define	dnnActivationMaskPReLU			0x4
#define	dnnActivationMaskLeakyReLU		0xc

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct __dnnConvOp		*	dnnConvOp;
typedef struct __dnnFFTConvOp		*	dnnFFTConvOp;
typedef struct __dnnCelledConvOp	*	dnnCelledConvOp;
typedef struct __dnnPoolingOp		*	dnnPoolingOp;
typedef struct __dnnBiasOp		*	dnnBiasOp;
typedef struct __dnnFilterOp		*	dnnFilterOp;
typedef struct __dnnActivationOp	*	dnnActivationOp;
typedef struct __dnnFullConnectionOp	*	dnnFullConnectionOp;

typedef enum dnnStatus{
	dnnSuccess=0			,
	dnnErrorInvalidValue	,
	dnnErrorInvalidDevice	,
	dnnErrorOutOfMaxSize	,
	dnnErrorOutOfMemory		,
	dnnErrorOutOfDeviceMemory
} dnnStatus_t;

__APIENTRY__	dnnStatus_t		__APICALL__ dnnInit();
__APIENTRY__	int			__APICALL__ dnnSuperdeviceCount();
__APIENTRY__	int			__APICALL__ dnnSuperdeviceSize( int );
__APIENTRY__	int			__APICALL__ dnnOptimalSuperdeviceID();
__APIENTRY__	void			__APICALL__ dnnSetDevice( int, int );

__APIENTRY__	uint64			__APICALL__ dnnCreateTensorShape( unsigned int, unsigned int, unsigned int, unsigned int, unsigned int );
__APIENTRY__	size_t			__APICALL__ dnnGetTensorPitch( uint64 );
__APIENTRY__	size_t			__APICALL__ dnnGetTensorSize( uint64 );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateTensor( const void**, uint64 );

__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateConvOp( dnnConvOp*, int, uint64, uint64, int, int );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreatePoolingOp( dnnPoolingOp*, int, unsigned int, uint64, int, int );
__APIENTRY__	dnnStatus_t		__APICALL__ dnnCreateBiasOp( dnnBiasOp*, int, unsigned int, uint64 );

__APIENTRY__	void			__APICALL__ dnnConv( dnnConvOp, void*, const void*, const void*, float, int, CUstream );
__APIENTRY__	void			__APICALL__ dnnPooling( dnnPoolingOp, void*, const void*, int, CUstream );
__APIENTRY__	void			__APICALL__ dnnBiasComputeDx( dnnBiasOp*, void*, const void*, CUstream );
__APIENTRY__	void			__APICALL__ dnnBiasUpdate( dnnBiasOp*, void*, const void*, CUstream );

__APIENTRY__	void			__APICALL__ dnnDestroyConvOp( dnnConvOp );
__APIENTRY__	void			__APICALL__ dnnDestroyPoolingOp( dnnPoolingOp );
__APIENTRY__	void			__APICALL__ dnnDestroyBiasOp( dnnBiasOp );
__APIENTRY__	void			__APICALL__ dnnExit();

#ifdef __cplusplus
}
#endif

#endif
