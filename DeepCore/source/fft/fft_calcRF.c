#include"../../include/fft/fft_calcRF.h"

#define PI 3.1415926535897931e+0

#define mMakeRF(p,s){			\
	(p).x=(float)cos((s)*-PI);	\
	(p).y=(float)sin((s)*-PI);	\
}
void fft_calcRF( float2* const p, int n, double rt )
{
	int i=0;
	do{ mMakeRF(p[i],i*rt) }while((++i)<n);
}