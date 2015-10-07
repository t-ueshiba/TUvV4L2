/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_SELECT_H)
#define __TU_SIMD_INTEL_SELECT_H

namespace TU
{
namespace simd
{
template <class T> inline vec<T>
select(vec<T> mask, vec<T> x, vec<T> y)
{
    return (mask & x) | andnot(mask, y);
}
    
#if defined(SSE4)
#  define SIMD_SELECT(type)						\
    template <> inline vec<type>					\
    select(vec<mask_type<type> > mask, vec<type> x, vec<type> y)	\
    {									\
	   return SIMD_MNEMONIC(blendv, SIMD_PREFIX(type), ,		\
				SIMD_SIGNED(int8_t))(y, x, mask);	\
    }
#  define SIMD_SELECT_F(type)						\
    SIMD_SPECIALIZED_FUNC(						\
	vec<type> select(vec<mask_type<type> > mask,			\
			 vec<type> x, vec<type> y),			\
	blendv, (y, x, mask), void, type, SIMD_BASE)

  SIMD_SELECT(int8_t)
  SIMD_SELECT(int16_t)
  SIMD_SELECT(int32_t)
  SIMD_SELECT(int64_t)
  SIMD_SELECT(u_int8_t)
  SIMD_SELECT(u_int16_t)
  SIMD_SELECT(u_int32_t)
  SIMD_SELECT(u_int64_t)

  SIMD_SELECT_F(float)
  SIMD_SELECT_F(double)
  
#  undef SIMD_SELECT
#  undef SIMD_SELECT_F
#endif

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_SELECT_H
