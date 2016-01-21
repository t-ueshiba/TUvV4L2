/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_DUP_H)
#define __TU_SIMD_X86_DUP_H

#include "TU/simd/x86/unpack.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  N-tuple generators							*
************************************************************************/
// 複製数：N = 2, 4, 8, 16,...;
// 全体をN個の部分に分けたときの複製区間：0 <= I < N
template <size_t N, size_t I, class T> vec<T>	n_tuple(vec<T> x)	;

template <size_t I, class T> inline vec<T>
dup(vec<T> x)
{
    return n_tuple<2, I>(x);
}

template <size_t I, class T> inline vec<T>
quadup(vec<T> x)
{
    return n_tuple<4, I>(x);
}
    
template <size_t I, class T> inline vec<T>
octup(vec<T> x)
{
    return n_tuple<8, I>(x);
}
    
#define SIMD_N_TUPLE(type)						\
    template <> inline vec<type>					\
    n_tuple<2, 0>(vec<type> x)		{return unpack<false>(x, x);}	\
    template <> inline vec<type>					\
    n_tuple<2, 1>(vec<type> x)		{return unpack<true>(x, x);}

template <size_t N, size_t I, class T> inline vec<T>
n_tuple(vec<T> x)
{
    return n_tuple<2, (I&0x1)>(n_tuple<(N>>1), (I>>1)>(x));
}

SIMD_N_TUPLE(int8_t)
SIMD_N_TUPLE(int16_t)
SIMD_N_TUPLE(int32_t)
SIMD_N_TUPLE(u_int8_t)
SIMD_N_TUPLE(u_int16_t)
SIMD_N_TUPLE(u_int32_t)
#if defined(SSE)
  SIMD_N_TUPLE(float)
#  if defined(SSE2)
  SIMD_N_TUPLE(int64_t)
  SIMD_N_TUPLE(u_int64_t)
  SIMD_N_TUPLE(double)
#  endif
#endif

#undef SIMD_N_TUPLE
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_DUP_H
