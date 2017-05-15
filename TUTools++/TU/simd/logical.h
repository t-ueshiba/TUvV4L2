/*
 *  $Id$
 */
#if !defined(__TU_SIMD_LOGICAL_H)
#define __TU_SIMD_LOGICAL_H

#include "TU/tuple.h"
#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Logical operators							*
************************************************************************/
template <class T> vec<T>	operator ~(vec<T> x)			;
template <class T> vec<T>	operator &(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator |(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator ^(vec<T> x, vec<T> y)		;
template <class T> vec<T>	andnot(vec<T> x, vec<T> y)		;

template <class T> inline vec<T>&
vec<T>::andnot(vec<T> x)	{ return *this = simd::andnot(x, *this); }

/************************************************************************
*  Logical operators for vec tuples					*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
andnot(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return andnot(x, y); }, l, r);
}

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/logical.h"
#elif defined(NEON)
#  include "TU/simd/arm/logical.h"
#endif

#endif	// !__TU_SIMD_LOGICAL_H
