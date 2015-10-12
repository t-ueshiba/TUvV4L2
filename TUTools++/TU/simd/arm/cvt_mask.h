/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_CVT_MASK_H)
#define __TU_SIMD_ARM_CVT_MASK_H

namespace TU
{
namespace simd
{
template <class S, size_t I, class T> inline vec<S>
cvt_mask(vec<T> x)
{
    return cast<S>(cvt<signed_type<S>, I>(cast<signed_type<T> >(x)));
}

template <class S, class T> inline vec<S>
cvt_mask(vec<T> x, vec<T> y)
{
    return cast<S>(cvt<signed_type<S> >(cast<signed_type<T> >(x),
					cast<signed_type<T> >(y)));
}

#define SIMD_CVT_MASK(from, to)						\
    template <> inline vec<to>						\
    cvt_mask<to, 0>(vec<from> x)					\
    {									\
	return cast<to>(x);						\
    }

SIMD_CVT_MASK(u_int32_t, float)
SIMD_CVT_MASK(float, u_int32_t)

#undef SIMD_CVT_MASK

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_CVT_MASK_H
