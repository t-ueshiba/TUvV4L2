/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_UNPACK_H)
#define __TU_SIMD_INTEL_UNPACK_H

namespace TU
{
namespace simd
{
/************************************************************************
*  Unpack operators							*
************************************************************************/
//! 2つのベクトルの下位半分の成分を交互に混合する．
/*!
  \param x	その成分を偶数番目に配置するベクトル
  \param y	その成分を奇数番目に配置するベクトル
  \return	生成されたベクトル
*/
template <class T> vec<T>	unpack_low(vec<T> x, vec<T> y)		;

//! 2つのベクトルの上位半分の成分を交互に混合する．
/*!
  \param x	その成分を偶数番目に配置するベクトル
  \param y	その成分を奇数番目に配置するベクトル
  \return	生成されたベクトル
*/
template <class T> vec<T>	unpack_high(vec<T> x, vec<T> y)		;

#define SIMD_UNPACK_LOW_HIGH(type)					\
    SIMD_SIGNED_FUNC(unpack_low,  unpacklo, type)			\
    SIMD_SIGNED_FUNC(unpack_high, unpackhi, type)

SIMD_UNPACK_LOW_HIGH(int8_t)
SIMD_UNPACK_LOW_HIGH(int16_t)
SIMD_UNPACK_LOW_HIGH(int32_t)
SIMD_UNPACK_LOW_HIGH(u_int8_t)
SIMD_UNPACK_LOW_HIGH(u_int16_t)
SIMD_UNPACK_LOW_HIGH(u_int32_t)
#if defined(SSE)
  SIMD_UNPACK_LOW_HIGH(float)
#  if defined(SSE2)
  SIMD_UNPACK_LOW_HIGH(int64_t)
  SIMD_UNPACK_LOW_HIGH(u_int64_t)
  SIMD_UNPACK_LOW_HIGH(double)
#  endif
#endif

#undef SIMD_UNPACK_LOW_HIGH
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_UNPACK_H
