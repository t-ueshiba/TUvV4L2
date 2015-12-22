/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARITHMETIC_H)
#define __TU_SIMD_ARITHMETIC_H

#include "TU/simd/vec.h"
#include "TU/simd/logical.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class T> vec<T>	operator +(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator -(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator *(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator /(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator %(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator -(vec<T> x)			;
template <class T> vec<T>	mulhi(vec<T> x, vec<T> y)		;
template <class T> vec<T>	min(vec<T> x, vec<T> y)			;
template <class T> vec<T>	max(vec<T> x, vec<T> y)			;
template <class T> vec<T>	rcp(vec<T> x)				;
template <class T> vec<T>	sqrt(vec<T> x)				;
template <class T> vec<T>	rsqrt(vec<T> x)				;

/************************************************************************
*  Average values							*
************************************************************************/
template <class T> vec<T>	avg(vec<T> x, vec<T> y)			;
template <class T> vec<T>	sub_avg(vec<T> x, vec<T> y)		;

/************************************************************************
*  Absolute values							*
************************************************************************/
template <class T> vec<T>	abs(vec<T> x)		;
template <> inline Iu8vec	abs(Iu8vec x)		{ return x; }
template <> inline Iu16vec	abs(Iu16vec x)		{ return x; }
template <> inline Iu32vec	abs(Iu32vec x)		{ return x; }
template <> inline Iu64vec	abs(Iu64vec x)		{ return x; }

/************************************************************************
*  Absolute differences							*
************************************************************************/
template <class T> vec<T>	diff(vec<T> x, vec<T> y)		;
  
/************************************************************************
*  Arithmetic operators for vec tuples					*
************************************************************************/
namespace detail
{
  struct generic_min
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x, vec<T_> y) const	{ return min(x, y); }
  };

  struct generic_max
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x, vec<T_> y) const	{ return max(x, y); }
  };

  struct generic_avg
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x, vec<T_> y) const	{ return avg(x, y); }
  };

  struct generic_sub_avg
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x, vec<T_> y) const	{return sub_avg(x, y);}
  };

  struct generic_abs
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x)		const	{return abs(x);}
  };

  struct generic_diff
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x, vec<T_> y) const	{ return diff(x, y); }
  };
}
    
template <class HEAD, class TAIL> inline auto
min(const boost::tuples::cons<HEAD, TAIL>& x,
    const boost::tuples::cons<HEAD, TAIL>& y)
    -> decltype(boost::tuples::cons_transform(x, y, detail::generic_min()))
{
    return boost::tuples::cons_transform(x, y, detail::generic_min());
}

template <class T, class HEAD, class TAIL,
	  class=typename std::enable_if<
		    !boost::tuples::is_tuple<T>::value>::type> inline auto
min(const T& c, const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(
		    x, std::bind(detail::generic_min(),
				 c, std::placeholders::_1)))
{
    return boost::tuples::cons_transform(
	       x, std::bind(detail::generic_min(),
			    c, std::placeholders::_1));
}

template <class HEAD, class TAIL, class T,
	  class=typename std::enable_if<
		    !boost::tuples::is_tuple<T>::value>::type> inline auto
min(const boost::tuples::cons<HEAD, TAIL>& x, const T& c) -> decltype(min(c, x))
{
    return min(c, x);
}

template <class HEAD, class TAIL> inline auto
max(const boost::tuples::cons<HEAD, TAIL>& x,
    const boost::tuples::cons<HEAD, TAIL>& y)
    -> decltype(boost::tuples::cons_transform(x, y, detail::generic_max()))
{
    return boost::tuples::cons_transform(x, y, detail::generic_max());
}

template <class T, class HEAD, class TAIL,
	  class=typename std::enable_if<
		    !boost::tuples::is_tuple<T>::value>::type> inline auto
max(const T& c, const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(
		    x, std::bind(detail::generic_max(),
				 c, std::placeholders::_1)))
{
    return boost::tuples::cons_transform(
	       x, std::bind(detail::generic_max(),
			    c, std::placeholders::_1));
}

template <class HEAD, class TAIL, class T,
	  class=typename std::enable_if<
		    !boost::tuples::is_tuple<T>::value>::type> inline auto
max(const boost::tuples::cons<HEAD, TAIL>& x, const T& c) -> decltype(max(c, x))
{
    return max(c, x);
}
    
template <class HEAD, class TAIL> inline auto
avg(const boost::tuples::cons<HEAD, TAIL>& x,
    const boost::tuples::cons<HEAD, TAIL>& y)
    -> decltype(boost::tuples::cons_transform(x, y, detail::generic_avg()))
{
    return boost::tuples::cons_transform(x, y, detail::generic_avg());
}
    
template <class HEAD, class TAIL> inline auto
sub_avg(const boost::tuples::cons<HEAD, TAIL>& x,
	const boost::tuples::cons<HEAD, TAIL>& y)
    -> decltype(boost::tuples::cons_transform(x, y, detail::generic_sub_avg()))
{
    return boost::tuples::cons_transform(x, y, detail::generic_sub_avg());
}

template <class HEAD, class TAIL> inline auto
abs(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(x, detail::generic_abs()))
{
    return boost::tuples::cons_transform(x, detail::generic_abs());
}
    
template <class HEAD, class TAIL> inline auto
diff(const boost::tuples::cons<HEAD, TAIL>& x,
     const boost::tuples::cons<HEAD, TAIL>& y)
    -> decltype(boost::tuples::cons_transform(x, y, detail::generic_diff()))
{
    return boost::tuples::cons_transform(x, y, detail::generic_diff());
}
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/arithmetic.h"
#elif defined(NEON)
#  include "TU/simd/arm/arithmetic.h"
#endif

#endif	// !__TU_SIMD_ARITHMETIC_H
