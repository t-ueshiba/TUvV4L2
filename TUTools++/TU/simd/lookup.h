/*!
  \file		lookup.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトルに収められたindexによりmemory lookupを行う関数の定義
*/
#if !defined(TU_SIMD_LOOKUP_H)
#define TU_SIMD_LOOKUP_H

#include "TU/simd/cvt.h"
#include "TU/simd/insert_extract.h"
#include "TU/simd/arithmetic.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Lookup								*
************************************************************************/
template <class P, class T,
	  typename std::enable_if<(vec<P>::size == vec<T>::size)>::type*
	  = nullptr>
vec<P>	lookup(const P* p, vec<T> idx)					;

template <class P, class T,
	  typename std::enable_if<(vec<P>::size > vec<T>::size)>::type*
	  = nullptr>
vec<T>	lookup(const P* p, vec<T> idx)					;

template <class P> inline auto
lookup(const P* p, Is32vec row, Is32vec col, int32_t stride)
    -> decltype(lookup(p, row*stride + col))
{
    return lookup(p, row*stride + col);
}

template <class P> auto
lookup(const P* p, Is16vec row, Is16vec col, int32_t stride)
    -> decltype(lookup(p, col))						;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/lookup.h"
#elif defined(NEON)
#  include "TU/simd/arm/lookup.h"
#endif

#endif	// !TU_SIMD_LOOKUP_H
