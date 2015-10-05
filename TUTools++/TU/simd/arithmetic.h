/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARITHMETIC_H)
#define __TU_SIMD_ARITHMETIC_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class T> static vec<T>	operator +(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator -(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator *(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator /(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator %(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator -(vec<T> x)		;
template <class T> static vec<T>	mulhi(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	min(vec<T> x, vec<T> y)		;
template <class T> static vec<T>	max(vec<T> x, vec<T> y)		;
template <class T> static vec<T>	rcp(vec<T> x)			;
template <class T> static vec<T>	sqrt(vec<T> x)			;
template <class T> static vec<T>	rsqrt(vec<T> x)			;
    
/************************************************************************
*  Average values							*
************************************************************************/
template <class T> static vec<T>	avg(vec<T> x, vec<T> y)		;
template <class T> static vec<T>	sub_avg(vec<T> x, vec<T> y)	;

/************************************************************************
*  Absolute values							*
************************************************************************/
template <class T> static vec<T>	abs(vec<T> x)	;
template <> inline Iu8vec		abs(Iu8vec x)	{return x;}
template <> inline Iu16vec		abs(Iu16vec x)	{return x;}
template <> inline Iu32vec		abs(Iu32vec x)	{return x;}
template <> inline Iu64vec		abs(Iu64vec x)	{return x;}

/************************************************************************
*  Absolute differences							*
************************************************************************/
template <class T> static vec<T>	diff(vec<T> x, vec<T> y)	;
  
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/arithmetic.h"
#elif defined(NEON)
#  include "TU/simd/arm/arithmetic.h"
#endif

#endif	// !__TU_SIMD_ARITHMETIC_H
