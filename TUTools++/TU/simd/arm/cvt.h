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
  vec<T>	combine(half_type<T> x, half_type<T> y)			;
  template <bool HI, class T>
  half_type<T>	split(vec<T> x)						;

#define SIMD_COMBINE_SPLIT(type)					\
    SIMD_SPECIALIZED_FUNC(						\
	vec<type> combine<type>(half_type<type> x, half_type<type> y),	\
	combine, (x, y), void, type)					\
    SIMD_SPECIALIZED_FUNC(half_type<type> split<false>(vec<type> x),	\
			  get_low, (x), void, type)			\
    SIMD_SPECIALIZED_FUNC(half_type<type> split<true>(vec<type> x),	\
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

template <class T, bool HI, bool MASK, class S> inline vec<T>
cvt(vec<S> x)
{
    using F = typename std::conditional<MASK, signed_type<S>, S>::type;
    
    return cast<T>(detail::cvtup(detail::split<HI>(cast<F>(x))));
}

template <class T, bool MASK, class S> inline vec<T>
cvt(vec<S> x, vec<S> y)
{
    using F = typename std::conditional<MASK, signed_type<S>, S>::type;
    
    return cast<T>(detail::combine<lower_type<F> >(
		       detail::cvtdown(cast<F>(x)),
		       detail::cvtdown(cast<F>(y))));
}

#define SIMD_CVTEQ(type)						\
    template <> inline vec<type>					\
    cvt<type>(vec<type> x)						\
    {									\
	return x;							\
    }									\
    template <> inline vec<type>					\
    cvt<type, false, true>(vec<type> x)					\
    {									\
	return x;							\
    }
    
SIMD_CVTEQ(int8_t)		// int8_t    -> int8_t
SIMD_CVTEQ(int16_t)		// int16_t   -> int16_t
SIMD_CVTEQ(int32_t)		// int32_t   -> int32_t
SIMD_CVTEQ(int64_t)		// int64_t   -> int64_t
SIMD_CVTEQ(u_int8_t)		// u_int8_t  -> u_int8_t
SIMD_CVTEQ(u_int16_t)		// u_int16_t -> u_int16_t
SIMD_CVTEQ(u_int32_t)		// u_int32_t -> u_int32_t
SIMD_CVTEQ(u_int64_t)		// u_int64_t -> u_int64_t
SIMD_CVTEQ(float)		// float     -> float

#define SIMD_CVTF(from, to)						\
    SIMD_SPECIALIZED_FUNC(vec<to> cvt<to>(vec<from> x), cvtq, (x), to, from)

SIMD_CVTF(int32_t,   float)	// int32_t   -> float
SIMD_CVTF(float,     int32_t)	// float     -> int32_t
SIMD_CVTF(u_int32_t, float)	// u_int32_t -> float

#define SIMD_CVT_MASK(from, to)						\
    template <> inline vec<to>						\
    cvt<to, false, true>(vec<from> x)					\
    {									\
	return cast<to>(x);						\
    }

SIMD_CVT_MASK(u_int32_t, float)	// u_intew_t -> float
SIMD_CVT_MASK(float, u_int32_t)	// float     -> u_int32_t

#undef SIMD_CVTEQ
#undef SIMD_CVTF
#undef SIMD_CVT_MASK

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_CVT_H
