/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_CVT_H)
#define __TU_SIMD_ARM_CVT_H

namespace TU
{
namespace simd
{
namespace detail
{
  template <class T>
  static vec<T>		combine(half_type<T> x, half_type<T> y)		;
  template <size_t I, class T>
  static half_type<T>	split(vec<T> x)					;

#define SIMD_COMBINE_SPLIT(type)					\
    SIMD_SPECIALIZED_FUNC(						\
	vec<type> combine(half_type<type> x, half_type<type> y),	\
	combine, (x, y), void, type)					\
    SIMD_SPECIALIZED_FUNC(half_type<type> split<0>(vec<type> x),	\
			  get_low, (x), void, type)			\
    SIMD_SPECIALIZED_FUNC(half_type<type> split<1>(vec<type> x),	\
			  get_high, (x), void, type)

  SIMD_COMBINE_SPLIT(int8_t)
  SIMD_COMBINE_SPLIT(int16_t)
  SIMD_COMBINE_SPLIT(int32_t)
  SIMD_COMBINE_SPLIT(int64_t)
  SIMD_COMBINE_SPLIT(u_int8_t)
  SIMD_COMBINE_SPLIT(u_int16_t)
  SIMD_COMBINE_SPLIT(u_int32_t)
  SIMD_COMBINE_SPLIT(u_int64_t)
  SIMD_COMBINE_SPLIT(float)

#undef SIMD_COMBINE_SPLIT
	
#define SIMD_CVTUP(type)						\
    SIMD_FUNC(vec<upper_type<type> > cvtup(half_type<type> x),		\
	      movl, (x), void, type)
#define SIMD_CVTDOWN(type)						\
    SIMD_FUNC(half_type<lower_type<type> > cvtdown(vec<type> x),	\
	      movn, (x), void, type)

SIMD_CVTUP(int8_t)
SIMD_CVTUP(int16_t)
SIMD_CVTUP(int32_t)
SIMD_CVTUP(u_int8_t)
SIMD_CVTUP(u_int16_t)
SIMD_CVTUP(u_int32_t)
SIMD_CVTDOWN(int16_t)
SIMD_CVTDOWN(int32_t)
SIMD_CVTDOWN(int64_t)
SIMD_CVTDOWN(u_int16_t)
SIMD_CVTDOWN(u_int32_t)
SIMD_CVTDOWN(u_int64_t)

#undef SIMD_CVTUP
#undef SIMD_CVTDOWN

}	// namespace detail
    
#define SIMD_CVTEQ(type)						\
    template <> inline vec<type>					\
    cvt<type, 0>(vec<type> x)						\
    {									\
	return x;							\
    }
#define SIMD_CVT(from, to)						\
    template <> inline vec<to>						\
    cvt<to, 0>(vec<from> x)						\
    {									\
	return cast<to>(detail::cvtup(detail::split<0>(x)));		\
    }									\
    template <> inline vec<to>						\
    cvt<to, 1>(vec<from> x)						\
    {									\
	return cast<to>(detail::cvtup(detail::split<1>(x)));		\
    }
#define SIMD_CVTF(from, to)						\
    SIMD_SPECIALIZED_FUNC(vec<to> cvt<to>(vec<from> x), cvtq, (x), to, from)

SIMD_CVTEQ(int8_t)
SIMD_CVTEQ(int16_t)
SIMD_CVTEQ(int32_t)
SIMD_CVTEQ(int64_t)
SIMD_CVTEQ(u_int8_t)
SIMD_CVTEQ(u_int16_t)
SIMD_CVTEQ(u_int32_t)
SIMD_CVTEQ(u_int64_t)
SIMD_CVTEQ(float)

SIMD_CVT(int8_t,    int16_t)
SIMD_CVT(int16_t,   int32_t)
SIMD_CVT(int32_t,   int64_t)
SIMD_CVT(u_int8_t,  int16_t)
SIMD_CVT(u_int8_t,  u_int16_t)
SIMD_CVT(u_int16_t, int32_t)
SIMD_CVT(u_int16_t, u_int32_t)
SIMD_CVT(u_int32_t, int64_t)
SIMD_CVT(u_int32_t, u_int64_t)

SIMD_CVTF(int32_t,   float)
SIMD_CVTF(float,     int32_t)
SIMD_CVTF(u_int32_t, float)

#undef SIMD_CVTEQ
#undef SIMD_CVT
#undef SIMD_CVTF

template <class S, class T> inline vec<S>
cvt(vec<T> x, vec<T> y)
{
    return cast<S>(detail::combine(detail::cvtdown(x), detail::cvtdown(y)));
}

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_CVT_H
