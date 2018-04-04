/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_TYPE_TRAITS_H)
#define TU_SIMD_X86_TYPE_TRAITS_H

#include <x86intrin.h>

namespace TU
{
namespace simd
{
/************************************************************************
*  SIMD vector types							*
************************************************************************/
#if defined(AVX512)
  using ivec_t	= __m512i;		//!< 整数ベクトルのSIMD型
#elif defined(AVX2)
  using ivec_t	= __m256i;		//!< 整数ベクトルのSIMD型
#elif defined(SSE2)  
  using ivec_t	= __m128i;		//!< 整数ベクトルのSIMD型
#else
  using ivec_t	= __m64;		//!< 整数ベクトルのSIMD型
#endif
    
#if defined(AVX512)
  using fvec_t	= __m512;		//!< floatベクトルのSIMD型
#elif defined(AVX)
  using fvec_t	= __m256;		//!< floatベクトルのSIMD型
#elif defined(SSE)
  using fvec_t	= __m128;		//!< floatベクトルのSIMD型
#else
  using fvec_t	= char;			//!< ダミー
#endif

#if defined(AVX512)
  using dvec_t	= __m512d;		//!< doubleベクトルのSIMD型
#elif defined(AVX)
  using dvec_t	= __m256d;		//!< doubleベクトルのSIMD型
#elif defined(SSE2)
  using dvec_t	= __m128d;		//!< doubleベクトルのSIMD型
#else
  using dvec_t	= char;			//!< ダミー
#endif

/************************************************************************
*  type traits								*
************************************************************************/
template <class T>
struct type_traits : type_traits_base<T>
{
    using complementary_type		= void;	//!< 相互変換可能な浮動小数点数
    using complementary_mask_type	= void;
    using base_type			= ivec_t;
};

template <>
struct type_traits<int16_t> : type_traits_base<int16_t>
{
    using complementary_type
	= std::conditional_t<sizeof(fvec_t) == sizeof(ivec_t), void, float>;
    using complementary_mask_type	= void;
    using base_type			= ivec_t;
};

template <>
struct type_traits<int32_t> : type_traits_base<int32_t>
{
    using complementary_type
	= std::conditional_t<sizeof(fvec_t) == sizeof(ivec_t) ||
			     std::is_void<dvec_t>::value, float, double>;
    using complementary_mask_type	= void;
    using base_type			= ivec_t;
};

template <>
struct type_traits<uint16_t> : type_traits_base<uint16_t>
{
    using complementary_type
	= std::conditional_t<sizeof(fvec_t) == sizeof(ivec_t), void, float>;
    using complementary_mask_type	= complementary_type;
    using base_type			= ivec_t;
};
    
template <>
struct type_traits<uint32_t> : type_traits_base<uint32_t>
{
    using complementary_type		= void;
    using complementary_mask_type
	= std::conditional_t<sizeof(fvec_t) == sizeof(ivec_t), float, double>;
    using base_type			= ivec_t;
};
    
template <>
struct type_traits<uint64_t> : type_traits_base<uint64_t>
{
    using complementary_type		= void;
    using complementary_mask_type
	= std::conditional_t<sizeof(fvec_t) == sizeof(ivec_t), double, void>;
    using base_type			= ivec_t;
};

template <>
struct type_traits<float> : type_traits_base<float>
{
    using mask_type			= float;
    using complementary_type
	= std::conditional_t<sizeof(ivec_t) == sizeof(fvec_t), int32_t,
							       int16_t>;
    using complementary_mask_type
	= std::conditional_t<sizeof(ivec_t) == sizeof(fvec_t), uint32_t,
							       uint16_t>;
    using base_type			= fvec_t;
};

template <>
struct type_traits<double> : type_traits_base<double>
{
    using mask_type			= double;
    using complementary_type		= int32_t;
    using complementary_mask_type
	= std::conditional_t<sizeof(ivec_t) == sizeof(dvec_t), uint64_t,
							        int32_t>;
    using base_type			= dvec_t;
};
    
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_X86_TYPE_TRAITS_H
