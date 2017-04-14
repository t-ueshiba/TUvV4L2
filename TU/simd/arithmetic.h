/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARITHMETIC_H)
#define __TU_SIMD_ARITHMETIC_H

#include "TU/tuple.h"
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

template <class T> inline vec<T>
operator *(T c, vec<T> x)
{
    return vec<T>(c) * x;
}

template <class T> inline vec<T>
operator *(vec<T> x, T c)
{
    return x * vec<T>(c);
}

template <class T> inline vec<T>
operator /(vec<T> x, T c)
{
    return x / vec<T>(c);
}

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
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
min(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return min(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
max(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return max(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
avg(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return avg(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
sub_avg(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return sub_avg(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
abs(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return abs(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
diff(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return diff(x, y); }, l, r);
}

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/arithmetic.h"
#elif defined(NEON)
#  include "TU/simd/arm/arithmetic.h"
#endif

#endif	// !__TU_SIMD_ARITHMETIC_H
