#ifndef __idc_string_h__
#define __idc_string_h__

__forceinline void idc_strcat( char* __restrict p_dst, const char* __restrict p_src )
{
    while( *p_dst ){ p_dst++; }
    while( *p_dst++=*p_src++ );
}
__forceinline int idc_strcmp( const char * src, const char * dst )
{
    int c=0;
    while(!(c=*(unsigned char*)src-*(unsigned char *)dst)&&*dst){ ++src, ++dst; }
    c=c<0?-1:(c>0?1:c);
    return c;
}

#endif