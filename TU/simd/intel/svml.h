/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_SVML_H)
#define __TU_SIMD_INTEL_SVML_H

#if defined(__INTEL_COMPILER)
#  include "TU/simd/arithmetic.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  SVML(Short Vector Math Library) functions				*
************************************************************************/
template <class T> vec<T>	erf(vec<T> x)				;
template <class T> vec<T>	erfc(vec<T> x)				;

template <class T> vec<T>	floor(vec<T> x)				;
template <class T> vec<T>	ceil(vec<T> x)				;

template <class T> vec<T>	exp(vec<T> x)				;
template <class T> vec<T>	cexp(vec<T> x)				;
template <class T> vec<T>	exp2(vec<T> x)				;
template <class T> vec<T>	pow(vec<T> x, vec<T> y)			;

template <class T> vec<T>	log(vec<T> x)				;
template <class T> vec<T>	log2(vec<T> x)				;
template <class T> vec<T>	log10(vec<T> x)				;
template <class T> vec<T>	clog(vec<T> x)				;

template <class T> vec<T>	invsqrt(vec<T> x)			;
template <class T> vec<T>	cbrt(vec<T> x)				;
template <class T> vec<T>	invcbrt(vec<T> x)			;
template <class T> vec<T>	csqrt(vec<T> x)				;

template <class T> vec<T>	cos(vec<T> x)				;
template <class T> vec<T>	sin(vec<T> x)				;
template <class T> vec<T>	tan(vec<T> x)				;
template <class T> vec<T>	sincos(typename vec<T>::base_type* pcos,
				       vec<T> x)			;
template <class T> vec<T>	acos(vec<T> x)				;
template <class T> vec<T>	asin(vec<T> x)				;
template <class T> vec<T>	atan(vec<T> x)				;
template <class T> vec<T>	atan2(vec<T> x, vec<T> y)		;
template <class T> vec<T>	cosh(vec<T> x)				;
template <class T> vec<T>	sinh(vec<T> x)				;
template <class T> vec<T>	tanh(vec<T> x)				;
template <class T> vec<T>	acosh(vec<T> x)				;
template <class T> vec<T>	asinh(vec<T> x)				;
template <class T> vec<T>	atanh(vec<T> x)				;

#if defined(SSE)
  SIMD_UNARY_FUNC(erf,     erf,		float)
  SIMD_UNARY_FUNC(erfc,    erfc,	float)

#  if defined(SSE4)
  SIMD_UNARY_FUNC(floor,   floor,	float)
  SIMD_UNARY_FUNC(ceil,    ceil,	float)
#  else
  SIMD_UNARY_FUNC(floor,   svml_floor,	float)
  SIMD_UNARY_FUNC(ceil,    svml_ceil,	float)
#  endif
  
  SIMD_UNARY_FUNC(exp,     exp,		float)
  SIMD_UNARY_FUNC(cexp,    cexp,	float)
  SIMD_UNARY_FUNC(exp2,    exp2,	float)
  SIMD_BINARY_FUNC(pow,    pow,		float)

  SIMD_UNARY_FUNC(log,     log,		float)
  SIMD_UNARY_FUNC(log2,    log2,	float)
  SIMD_UNARY_FUNC(log10,   log10,	float)
  SIMD_UNARY_FUNC(clog,    clog,	float)

  SIMD_UNARY_FUNC(invsqrt, invsqrt,	float)
  SIMD_UNARY_FUNC(cbrt,    cbrt,	float)
  SIMD_UNARY_FUNC(invcbrt, invcbrt,	float)
  SIMD_UNARY_FUNC(csqrt,   csqrt,	float)

  SIMD_UNARY_FUNC(cos,     cos,		float)
  SIMD_UNARY_FUNC(sin,     sin,		float)
  SIMD_UNARY_FUNC(tan,     tan,		float)
  SIMD_SPECIALIZED_FUNC(F32vec sincos(fvec_t* pcos, F32vec x),
			sincos, (pcos, x), void, float, SIMD_SUFFIX)
  SIMD_UNARY_FUNC(acos,    acos,	float)
  SIMD_UNARY_FUNC(asin,    asin,	float)
  SIMD_UNARY_FUNC(atan,    atan,	float)
  SIMD_BINARY_FUNC(atan2,  atan2,	float)
  SIMD_UNARY_FUNC(cosh,    cosh,	float)
  SIMD_UNARY_FUNC(sinh,    sinh,	float)
  SIMD_UNARY_FUNC(tanh,    tanh,	float)
  SIMD_UNARY_FUNC(acosh,   acosh,	float)
  SIMD_UNARY_FUNC(asinh,   asinh,	float)
  SIMD_UNARY_FUNC(atanh,   atanh,	float)
#endif

#if defined(SSE2)
#  if defined(_mm_idiv_epi32)
  // 整数除算
  SIMD_BINARY_FUNC(operator /, div, int8_t)
  SIMD_BINARY_FUNC(operator /, div, int16_t)
  SIMD_BINARY_FUNC(operator /, div, u_int8_t)
  SIMD_BINARY_FUNC(operator /, div, u_int16_t)

  // 剰余
  SIMD_BINARY_FUNC(operator %, rem, int8_t)
  SIMD_BINARY_FUNC(operator %, rem, int16_t)
  SIMD_BINARY_FUNC(operator %, rem, u_int8_t)
  SIMD_BINARY_FUNC(operator %, rem, u_int16_t)
#  endif
  
  // 整数除算
  SIMD_SIGNED_FUNC(operator /, idiv, int32_t)
  SIMD_SIGNED_FUNC(operator /, udiv, u_int32_t)

  // 剰余
  SIMD_SIGNED_FUNC(operator %, irem, int32_t)
  SIMD_SIGNED_FUNC(operator %, urem, u_int32_t)

  // 除算と剰余
  template <class T>  vec<T>	divrem(vec<T>& r,
					       vec<T> x, vec<T> y)	;

  SIMD_SPECIALIZED_FUNC(Is32vec divrem(Is32vec& r, Is32vec x, Is32vec y),
			idivrem, ((ivec_t*)&r, x, y),
			void, int32_t, SIMD_SIGNED)
  SIMD_SPECIALIZED_FUNC(Iu32vec divrem(Iu32vec& r, Iu32vec x, Iu32vec y),
			udivrem, ((ivec_t*)&r, x, y),
			void, u_int32_t, SIMD_SIGNED)

  SIMD_UNARY_FUNC(erf,     erf,		double)
  SIMD_UNARY_FUNC(erfc,    erfc,	double)

#  if defined(SSE4)
  SIMD_UNARY_FUNC(floor,   floor,	double)
  SIMD_UNARY_FUNC(ceil,    ceil,	double)
#  else
  SIMD_UNARY_FUNC(floor,   svml_floor,	double)
  SIMD_UNARY_FUNC(ceil,	   svml_ceil,	double)
#  endif
  SIMD_UNARY_FUNC(exp,     exp,		double)
  SIMD_UNARY_FUNC(exp2,    exp2,	double)
  SIMD_BINARY_FUNC(pow,    pow,		double)

  SIMD_UNARY_FUNC(log,     log,		double)
  SIMD_UNARY_FUNC(log2,    log2,	double)
  SIMD_UNARY_FUNC(log10,   log10,	double)

  SIMD_UNARY_FUNC(invsqrt, invsqrt,	double)
  SIMD_UNARY_FUNC(cbrt,    cbrt,	double)
  SIMD_UNARY_FUNC(invcbrt, invcbrt,	double)

  SIMD_UNARY_FUNC(cos,     cos,		double)
  SIMD_UNARY_FUNC(sin,     sin,		double)
  SIMD_UNARY_FUNC(tan,     tan,		double)
  SIMD_SPECIALIZED_FUNC(F64vec sincos(dvec_t* pcos, F64vec x),
			sincos, (pcos, x), void, double, SIMD_SUFFIX)
  SIMD_UNARY_FUNC(acos,    acos,	double)
  SIMD_UNARY_FUNC(asin,    asin,	double)
  SIMD_UNARY_FUNC(atan,    atan,	double)
  SIMD_BINARY_FUNC(atan2,  atan2,	double)
  SIMD_UNARY_FUNC(cosh,    cosh,	double)
  SIMD_UNARY_FUNC(sinh,    sinh,	double)
  SIMD_UNARY_FUNC(tanh,    tanh,	double)
  SIMD_UNARY_FUNC(acosh,   acosh,	double)
  SIMD_UNARY_FUNC(asinh,   asinh,	double)
  SIMD_UNARY_FUNC(atanh,   atanh,	double)
#endif
}	// namespace simd
}	// namespace TU
#endif	// __INTEL_COMPILER

#endif	// !__TU_SIMD_INTEL_SVML_H
