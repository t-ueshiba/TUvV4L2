/*
 *  $Id$
 */
#if !defined(__TU_SIMD_LOOKUP_H)
#define __TU_SIMD_LOOKUP_H

#include "TU/simd/insert_extract.h"
#include "TU/simd/arithmetic.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Lookup								*
************************************************************************/
template <class T, class P, class S>
typename std::enable_if<vec<T>::size == vec<S>::size, vec<T> >::type
lookup(const P* p, vec<S> idx)						;
    
template <class A, class S>
typename std::enable_if<vec<TU::detail::element_t<A> >::size ==
			vec<S>::size, vec<TU::detail::element_t<A> > >::type
lookup(const A& a, vec<S> row, vec<S> col)				;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/lookup.h"
#elif defined(NEON)
#  include "TU/simd/arm/lookup.h"
#endif

#endif	// !__TU_SIMD_LOOKUP_H
