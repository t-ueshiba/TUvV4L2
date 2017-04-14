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
namespace detail
{
  struct generic_andnot
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x, vec<T_> y) const
		{
		    return andnot(x, y);
		}
  };
}

template <class HEAD, class TAIL> inline auto
andnot(const boost::tuples::cons<HEAD, TAIL>& x,
       const boost::tuples::cons<HEAD, TAIL>& y)
    -> decltype(boost::tuples::cons_transform(detail::generic_andnot(), x, y))
{
    return boost::tuples::cons_transform(detail::generic_andnot(), x, y);
}
    

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/logical.h"
#elif defined(NEON)
#  include "TU/simd/arm/logical.h"
#endif

#endif	// !__TU_SIMD_LOGICAL_H
