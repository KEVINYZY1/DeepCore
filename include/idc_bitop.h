#ifndef __idc_bitop_h__
#define __idc_bitop_h__

#include<stdint.h>
#include<intrin.h>

__forceinline uint32_t idc_bhs( uint32_t n )
{
    unsigned long i;
#ifdef _MSC_VER
    _BitScanReverse( &i, n );
#else
    i=32-__buildin_clz(n);
#endif
    return i;
}
__forceinline uint32_t idc_popc( uint32_t n )
{
    uint32_t i;
#ifdef _MSC_VER
    i=_mm_popcnt_u32(n);
#else
    i=__buildin_popcount(n);
#endif
    return i;
}
__forceinline uint32_t idc_minls( uint32_t n )
{
    uint32_t p=idc_bhs(n);
    return (1<<(p+(idc_popc(n)>1)));
}

#endif