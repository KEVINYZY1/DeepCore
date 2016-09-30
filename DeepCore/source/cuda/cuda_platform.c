#include"../../include/cuda/cuda_platform.h"

static int __get_devices( CUdevice* const p_device, int* const p_arch )
{	
	struct{ int x, y; } cc;
	CUdevice device;
	int i, n_devices, sm, n_valided;	
	cuDeviceGetCount( &n_devices );
	if( n_devices<=0 ) return -1;
	for( i=0, n_valided=0; i<n_devices; ++i )
	{
		cuDeviceGet( &device, i );
		cuDeviceGetAttribute( &cc.x, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device );
		cuDeviceGetAttribute( &cc.y, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device );
		sm=10*cc.x+cc.y;
		if(sm<35) continue;
		p_device[n_valided]=device;
		p_arch[n_valided]=sm;
		++n_valided;
	}
	return n_valided;
}
static float __get_device_peak( float n, int cc )
{
	int n_cores_per_SM;
	switch(cc)
	{
	case 35: 
	case 37: n_cores_per_SM=192; break;
	case 50: 
	case 52: 
	case 53: 
	case 61:
	case 62: n_cores_per_SM=128; break;
	case 60: n_cores_per_SM= 64; break;
	}
	return (n*n_cores_per_SM);
}
static int __get_optimal_superdevice( cuda_platform_t* const p_plat )
{
	int i, n, optimal_id;
	float gflops, max_gflops;
	if( p_plat->n_sdevices==1 ) return 0;
	for( i=0, max_gflops=0; i<p_plat->n_sdevices; ++i )
	{
		n=p_plat->slist[i+1]-p_plat->slist[i];
		gflops=__get_device_peak( p_plat->nSM[i]*(0.000001f*p_plat->clock_rate[i]), p_plat->sarch[i] );
		gflops*=(n-0.1f*(n-1));
		if( gflops>max_gflops ){
			optimal_id=i; max_gflops=gflops;
		}
	}
	return optimal_id;
}

int cuda_platform_init( cuda_platform_t* const p_platform )
{
	CUdevice devices[MAX_DEVICES_PER_NODE], dev;
	char str[MAX_DEVICES_PER_NODE][64], q[MAX_DEVICES_PER_NODE];
	int cc[MAX_DEVICES_PER_NODE], i, k, idev, n, ns, p;

	if(cuInit(0)!=CUDA_SUCCESS) return 2;
	n=__get_devices( &devices[0], &cc[0] );
	if(n<=0) return 2;
	if(n==1)
	{
		p_platform->n_devices=1;
		p_platform->n_sdevices=1;
		p_platform->devices[0]=devices[0]; 
		p_platform->slist[0]=0; 
		p_platform->slist[1]=1; 
		p_platform->sarch[0]=cc[0]; 
		p_platform->opt_sdev_id=0;
	}
	else
	{
		for( i=0; i<n; ++i ){
			cuDeviceGetName( &str[i][0], 64, devices[i] );
		}
		p_platform->slist[0]=idev=ns=0; 
		for( i=0; i<n; ++i ){ q[i]=i; }
		do{
			k=q[0]; p_platform->devices[idev]=devices[k]; p_platform->sarch[ns]=cc[k];
			for( i=1, p=0; i<n; ++i )
			{
				if( strcmp( &str[k][0], &str[q[i]][0] )==0 ){
					p_platform->devices[++idev]=devices[q[i]];
				} else {
					q[p++]=q[i];
				}
			}
			p_platform->slist[ns+1]=++idev; ++ns;
		}while((n=p)>0);
		p_platform->n_sdevices=ns;	
	}
	for( i=0; i<p_platform->n_sdevices; ++i ){
		dev=p_platform->devices[p_platform->slist[i]];		
		cuDeviceGetAttribute( &p_platform->nSM[i], CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT	, dev );
		cuDeviceGetAttribute( &p_platform->clock_rate[i], CU_DEVICE_ATTRIBUTE_CLOCK_RATE	, dev );
	}
	if( p_platform->n_sdevices>1 ){
		p_platform->opt_sdev_id=__get_optimal_superdevice(p_platform);
	}
	return 0;
}
