/*!
  \file		arithmetic.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトルに対する算術演算の定義
*/
#if !defined(TU_SIMD_ARITHMETIC_H)
#define TU_SIMD_ARITHMETIC_H

#include "TU/tuple.h"
#include "TU/simd/vec.h"
#include "TU/simd/cast.h"
#include "TU/simd/logical.h"
#include "TU/simd/zero.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class T> vec<T>	operator +(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator *(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator /(vec<T> x, vec<T> y)		;
template <class T> vec<T>	operator %(vec<T> x, vec<T> y)		;
template <class T> vec<T>	subs(vec<T> x, vec<T> y)		;
template <class T> vec<T>	mulhi(vec<T> x, vec<T> y)		;
template <class T> vec<T>	min(vec<T> x, vec<T> y)			;
template <class T> vec<T>	max(vec<T> x, vec<T> y)			;
template <class T> vec<T>	rcp(vec<T> x)				;
template <class T> vec<T>	sqrt(vec<T> x)				;
template <class T> vec<T>	rsqrt(vec<T> x)				;

template <class T> inline vec<signed_type<T> >
operator -(vec<T> x, vec<T> y)
{
    const vec<T>	mask(1 << (8*sizeof(T) - 1));
    
    return cast<signed_type<T> >(mask ^ x) - cast<signed_type<T> >(mask ^ y);
}

template <class T> inline vec<signed_type<T> >
operator -(vec<T> x)
{
    return zero<T>() - x;
}
    
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

template <class T> inline vec<T>
operator %(vec<T> x, T c)
{
    return x % vec<T>(c);
}

/************************************************************************
*  Average values							*
************************************************************************/
template <class T> inline vec<T>
avg(vec<T> x, vec<T> y)			{return (x + y) >> 1;}
template <class T> inline vec<signed_type<T> >
sub_avg(vec<T> x, vec<T> y)		{return (x - y) >> 1;}

/************************************************************************
*  Absolute values							*
************************************************************************/
template <class T> vec<T>		abs(vec<T> x)	;
template <> inline Iu8vec		abs(Iu8vec x)	{ return x; }
template <> inline Iu16vec		abs(Iu16vec x)	{ return x; }
template <> inline Iu32vec		abs(Iu32vec x)	{ return x; }
template <> inline Iu64vec		abs(Iu64vec x)	{ return x; }

/************************************************************************
*  Absolute differences							*
************************************************************************/
template <class T> vec<T>		diff(vec<T> x, vec<T> y)	;
  
/************************************************************************
*  Fused multiply-add							*
************************************************************************/
template <class T>
inline vec<T>	fma(vec<T> x, vec<T> y, vec<T> z)	{return x*y + z;}

/************************************************************************
*  Horizontal addition							*
************************************************************************/
template <class T> inline T		hadd(vec<T> x)			;

/************************************************************************
*  Arithmetic operators for vec tuples					*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
min(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return min(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
max(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return max(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
avg(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return avg(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
sub_avg(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return sub_avg(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
abs(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return abs(x, y); }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
diff(const L& l, const R& r)
{
    return tuple_transform([](auto x, auto y){ return diff(x, y); }, l, r);
}

template <class L, class C, class R,
	  std::enable_if_t<any<is_tuple, L, C, R>::value>* = nullptr>
inline auto
fma(const L& l, const C& c, const R& r)
{
    return tuple_transform([](auto x, auto y, auto z)
			   { return fma(x, y, z); }, l, c, r);
}

template <class... T> inline auto
hadd(const std::tuple<vec<T>...>& t)
{
    return tuple_transform([](auto x){ return hadd(x); }, t);
}

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/arithmetic.h"
#elif defined(NEON)
#  include "TU/simd/arm/arithmetic.h"
#endif

#endif	// !TU_SIMD_ARITHMETIC_H
