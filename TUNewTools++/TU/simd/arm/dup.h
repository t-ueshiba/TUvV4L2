/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_DUP_H)
#define __TU_SIMD_ARM_DUP_H

namespace TU
{
namespace simd
{
/************************************************************************
*  Duplication operators						*
************************************************************************/
template <bool HI, class T> inline vec<T>
dup(vec<T> x)
{
    using I = typename std::conditional<
		  std::is_integral<T>::value,
		  unsigned_type<T>,
		  unsigned_type<complementary_type<T> > >::type;
    using U = upper_type<I>;
    
    const auto	y = cvt<U, HI>(cast<I>(x));
    return cast<T>(y | (y << 8*sizeof(T)));
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_DUP_H
