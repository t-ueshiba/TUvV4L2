/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_TYPE_TRAITS_H)
#define TU_SIMD_X86_TYPE_TRAITS_H

#include <immintrin.h>

namespace TU
{
namespace simd
{
/************************************************************************
*  SIMD vector types							*
************************************************************************/
#if defined(AVX2)
  typedef __m256i	ivec_t;		//!< 整数ベクトルのSIMD型
#elif defined(SSE2)  
  typedef __m128i	ivec_t;		//!< 整数ベクトルのSIMD型
#else
  typedef __m64		ivec_t;		//!< 整数ベクトルのSIMD型
#endif
    
#if defined(AVX)
  typedef __m256	fvec_t;		//!< floatベクトルのSIMD型
#elif defined(SSE)
  typedef __m128	fvec_t;		//!< floatベクトルのSIMD型
#else
  typedef char		fvec_t;		//!< ダミー
#endif

#if defined(AVX)
  typedef __m256d	dvec_t;		//!< doubleベクトルのSIMD型
#elif defined(SSE2)
  typedef __m128d	dvec_t;		//!< doubleベクトルのSIMD型
#else
  typedef char		dvec_t;		//!< ダミー
#endif

/************************************************************************
*  type traits								*
************************************************************************/
template <class T>
struct type_traits : type_traits_base<T>
{
    typedef void		complementary_type;
    typedef void		complementary_mask_type;
    typedef ivec_t		base_type;
};

template <>
struct type_traits<int16_t> : type_traits_base<int16_t>
{
    typedef typename std::conditional<
	(sizeof(fvec_t) == sizeof(ivec_t)),	// fvec_t と ivec_tが同サイズ
	void, float>::type	complementary_type;  //相互変換可能な浮動小数点数
    typedef void		complementary_mask_type;
    typedef ivec_t		base_type;
};

template <>
struct type_traits<int32_t> : type_traits_base<int32_t>
{
    typedef typename std::conditional<
	(sizeof(fvec_t) == sizeof(ivec_t)) ||	// fvec_t と ivec_tが同サイズ
	(sizeof(dvec_t) == sizeof(char)),	// または dvec_t が未定義なら...
	float, double>::type	complementary_type;  //相互変換可能な浮動小数点数
    typedef void		complementary_mask_type;
    typedef ivec_t		base_type;
};

template <>
struct type_traits<uint16_t> : type_traits_base<uint16_t>
{
    typedef typename std::conditional<
	(sizeof(fvec_t) == sizeof(ivec_t)),	// fvec_t と ivec_tが同サイズ
	void, float>::type	complementary_type;
    typedef complementary_type	complementary_mask_type;
    typedef ivec_t		base_type;
};
    
template <>
struct type_traits<uint32_t> : type_traits_base<uint32_t>
{
    typedef void		complementary_type;
    typedef typename std::conditional<
	(sizeof(fvec_t) == sizeof(ivec_t)),	// fvec_t と ivec_tが同サイズ
	float, double>::type	complementary_mask_type;
    typedef ivec_t		base_type;
};
    
template <>
struct type_traits<uint64_t> : type_traits_base<uint64_t>
{
    typedef void		complementary_type;
    typedef typename std::conditional<
	(sizeof(fvec_t) == sizeof(ivec_t)),	// fvec_t と ivec_tが同サイズ
	double, void>::type	complementary_mask_type;
    typedef ivec_t		base_type;
};

template <>
struct type_traits<float> : type_traits_base<float>
{
    typedef float		mask_type;
    typedef typename std::conditional<
	sizeof(ivec_t) == sizeof(fvec_t),
	int32_t,
	int16_t>::type		complementary_type;
    typedef typename std::conditional<
	sizeof(ivec_t) == sizeof(fvec_t),
	uint32_t,
	uint16_t>::type		complementary_mask_type;
    typedef fvec_t		base_type;
};

template <>
struct type_traits<double> : type_traits_base<double>
{
    typedef double		mask_type;
    typedef int32_t		complementary_type;
    typedef typename std::conditional<
	sizeof(ivec_t) == sizeof(dvec_t),
	uint64_t,
	int32_t>::type		complementary_mask_type;
    typedef dvec_t		base_type;
};
    
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_X86_TYPE_TRAITS_H
