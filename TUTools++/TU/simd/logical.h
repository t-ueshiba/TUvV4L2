/*
 *  $Id$
 */
#if !defined(__TU_SIMD_LOGICAL_H)
#define __TU_SIMD_LOGICAL_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Logical operators							*
************************************************************************/
template <class T> static vec<T>	operator ~(vec<T> x)		;
template <class T> static vec<T>	operator &(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator |(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator ^(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	andnot(vec<T> x, vec<T> y)	;

template <class T> inline vec<T>&
vec<T>::andnot(vec x)		{ return *this = simd::andnot(x, *this); }
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/logical.h"
#elif defined(NEON)
#  include "TU/simd/arm/logical.h"
#endif

#endif	// !__TU_SIMD_LOGICAL_H
