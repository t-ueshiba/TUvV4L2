/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_UNPACK_H)
#define TU_SIMD_X86_UNPACK_H

namespace TU
{
namespace simd
{
namespace detail
{
#if defined(AVX)
#  define SIMD_PERMUTE(type)						\
    template <size_t I>							\
    SIMD_FUNC(type permute(type x, type y),				\
	      permute2f128, (x, y, I), void, type, SIMD_BASE)

#  if defined(AVX2)
    SIMD_PERMUTE(ivec_t)
#  endif
  SIMD_PERMUTE(fvec_t)
  SIMD_PERMUTE(dvec_t)

#  undef SIMD_PERMUTE
#endif

template <bool HI, class T> base_type<T> unpack(vec<T> x, vec<T> y)	;

#define SIMD_UNPACK_LOW_HIGH(type)					\
    SIMD_SPECIALIZED_FUNC(base_type<type>				\
			  unpack<false>(vec<type> x, vec<type> y),	\
			  unpacklo, (x, y), void, type, SIMD_SIGNED)	\
    SIMD_SPECIALIZED_FUNC(base_type<type>				\
			  unpack<true>(vec<type> x, vec<type> y),	\
			  unpackhi, (x, y), void, type, SIMD_SIGNED)

SIMD_UNPACK_LOW_HIGH(int8_t)
SIMD_UNPACK_LOW_HIGH(int16_t)
SIMD_UNPACK_LOW_HIGH(int32_t)
SIMD_UNPACK_LOW_HIGH(uint8_t)
SIMD_UNPACK_LOW_HIGH(uint16_t)
SIMD_UNPACK_LOW_HIGH(uint32_t)
#if defined(SSE)
  SIMD_UNPACK_LOW_HIGH(float)
#  if defined(SSE2)
    SIMD_UNPACK_LOW_HIGH(int64_t)
    SIMD_UNPACK_LOW_HIGH(uint64_t)
    SIMD_UNPACK_LOW_HIGH(double)
#  endif
#endif

#undef SIMD_UNPACK_LOW_HIGH
}	// namespace detail
    
/************************************************************************
*  Unpack operators							*
************************************************************************/
//! 2つのベクトルの下位または上位半分の成分を交互に混合する．
/*!
  \param x	その成分を偶数番目に配置するベクトル
  \param y	その成分を奇数番目に配置するベクトル
  \return	生成されたベクトル
*/
template <bool HI, class T>
inline typename std::enable_if<(sizeof(vec<T>) < 32), base_type<T> >::type
unpack(vec<T> x, vec<T> y)
{
    return detail::unpack<HI>(x, y);
}

#if defined(AVX)
template <bool HI, class T>
inline typename std::enable_if<(sizeof(vec<T>) >= 32), base_type<T> >::type
unpack(vec<T> x, vec<T> y)
{
    return detail::permute<(HI ? 0x31 : 0x20)>(detail::unpack<false>(x, y),
					       detail::unpack<true >(x, y));
}
#endif    
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_X86_UNPACK_H
