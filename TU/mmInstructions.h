/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id$
 */
/*!
  \file		mmInstructions.h
  \brief	Intel CPUのマルチメディア命令に関連するクラスと関数の定義と実装
*/
#if !defined(__mmInstructions_h) && defined(__INTEL_COMPILER)
#define __mmInstructions_h

#if defined(AVX2)		// Core-i7 Haswell (2013)
#  define AVX
#endif
#if defined(AVX)		// Core-i7 Sandy-Bridge (2011)
#  define SSE4
#endif
#if defined(SSE4)		// Core2 with Penryn core(45nm)
#  define SSSE3
#endif
#if defined(SSSE3)		// Core2 (Jun. 2006)
#  define SSE3
#endif
#if defined(SSE3)		// Pentium-4 with Prescott core (Feb. 2004)
#  define SSE2
#endif
#if defined(SSE2)		// Pentium-4 (Nov. 2000)
#  define SSE
#endif
#if defined(SSE)		// Pentium-3 (Feb. 1999)
#  define MMX
#endif

#if defined(MMX)
#include <sys/types.h>
#include <immintrin.h>
#include <iostream>
#include <cassert>
#include <boost/utility/enable_if.hpp>
#include "TU/iterator.h"

/************************************************************************
*  Emulations								*
************************************************************************/
// alignr はSSSE3以降でのみサポートされるが，至便なのでemulationバージョンを定義
#if !defined(SSSE3)
  static inline __m64
  _mm_alignr_pi8(__m64 y, __m64 x, const int count)
  {
      return _mm_or_si64(_mm_slli_si64(y, 8*(8 - count)),
			 _mm_srli_si64(x, 8*count));
  }
#  if defined(SSE2)
    static inline __m128i
    _mm_alignr_epi8(__m128i y, __m128i x, const int count)
    {
	return _mm_or_si128(_mm_slli_si128(y, 16 - count),
			    _mm_srli_si128(x, count));
    }
#  endif
#endif

// AVX以降では alignr が上下のlaneに分断されて使いにくいので，自然なバージョンを定義
#if defined(AVX2)
  template <size_t N> static inline __m256i
  _mm256_emu_alignr_epi8(__m256i y, __m256i x)
  {
      return (N < 16 ?
	      _mm256_alignr_epi8(_mm256_permute2f128_si256(x, y, 0x21), x, N) :
	      _mm256_alignr_epi8(y, _mm256_permute2f128_si256(x, y, 0x21),
				 N - 16));
  }
#elif defined(AVX)
  template <size_t N> static inline __m256i
  _mm256_emu_alignr_epi8(__m256i y, __m256i x)
  {
      return (N < 16 ?
	      _mm256_insertf128_si256(
		  _mm256_insertf128_si256(
		      _mm256_undefined_si256(),
		      _mm_alignr_epi8(_mm256_extractf128_si256(x, 0x1),
				      _mm256_extractf128_si256(x, 0x0),
				      N),
		      0x0),
		  _mm_alignr_epi8(_mm256_extractf128_si256(y, 0x0),
				  _mm256_extractf128_si256(x, 0x1),
				  N),
		  0x1) :
	      _mm256_insertf128_si256(
		  _mm256_insertf128_si256(
		      _mm256_undefined_si256(),
		      _mm_alignr_epi8(_mm256_extractf128_si256(y, 0x0),
				      _mm256_extractf128_si256(x, 0x1),
				      N - 16),
		      0x0),
		  _mm_alignr_epi8(_mm256_extractf128_si256(y, 0x1),
				  _mm256_extractf128_si256(y, 0x0),
				  N - 16),
		  0x1));
  }
#endif

// AVX以降では srli_si256, slli_si256 が上下のlaneに分断されて使いにくいので，
// 自然なバージョンを定義
#if defined(AVX2)
  template <size_t N> static inline __m256i
  _mm256_emu_srli_si256(__m256i x)
  {
      return _mm256_alignr_epi8(_mm256_permute2f128_si256(x, x, 0x81), x, N);
  }

  template <size_t N> static inline __m256i
  _mm256_emu_slli_si256(__m256i x)
  {
      return (N < 16 ?
	      _mm256_alignr_epi8(x, _mm256_permute2f128_si256(x, x, 0x08),
				 16 - N) :
	      _mm256_alignr_epi8(_mm256_permute2f128_si256(x, x, 0x08),
				 _mm256_setzero_si256(),
				 32 - N));
  }
#elif defined(AVX)
  template <size_t N> static inline __m256i
  _mm256_emu_srli_si256(__m256i x)
  {
      __m128i	y = _mm256_extractf128_si256(x, 0x1);
      return _mm256_insertf128_si256(
		 _mm256_insertf128_si256(
		     _mm256_undefined_si256(),
		     _mm_alignr_epi8(y, _mm256_extractf128_si256(x, 0x0), N),
		     0x0),
		 _mm_srli_si128(y, N),
		 0x1);
  }

  template <size_t N> static inline __m256i
  _mm256_emu_slli_si256(__m256i x)
  {
      __m128i	y = _mm256_extractf128_si256(x, 0x0);
      return (N < 16 ?
	      _mm256_insertf128_si256(
		  _mm256_insertf128_si256(_mm256_undefined_si256(),
					  _mm_slli_si128(y, N),
					  0x0),
		  _mm_alignr_epi8(_mm256_extractf128_si256(x, 0x1), y, 16 - N),
		  0x1) :
	      _mm256_insertf128_si256(
		  _mm256_setzero_si256(),
		  _mm_alignr_epi8(y, _mm_setzero_si128(), 32 - N),
		  0x1));
  }
#endif

namespace TU
{
/*!
  \namespace	mm
  \brief	Intel SIMD命令を利用するためのクラスおよび関数を納める名前空間
*/
namespace mm
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
template <class T>	struct type_traits;

template <>
struct type_traits<int8_t>
{
    typedef u_int8_t	mask_type;		  //!< マスク
    typedef int8_t	signed_type;		  //!< 同サイズの符号付き整数
    typedef u_int8_t	unsigned_type;		  //!< 同サイズの符号なし整数
    typedef void	lower_type;		  //!< 半サイズの整数
    typedef int16_t	upper_type;		  //!< 倍サイズの整数
    typedef float	complementary_type;	  //!< ダミー
    typedef void	complementary_mask_type;  //!< ダミー
};
    
template <>
struct type_traits<int16_t>
{
    typedef u_int16_t	mask_type;
    typedef int16_t	signed_type;
    typedef u_int16_t	unsigned_type;
    typedef int8_t	lower_type;
    typedef int32_t	upper_type;
    typedef float	complementary_type;
    typedef void	complementary_mask_type;
};
    
template <>
struct type_traits<int32_t>
{
    typedef u_int32_t	mask_type;
    typedef int32_t	signed_type;
    typedef u_int32_t	unsigned_type;
    typedef int16_t	lower_type;
    typedef int64_t	upper_type;
    typedef typename boost::mpl::if_c<
	(sizeof(fvec_t) == sizeof(ivec_t)) ||	// fvec_t と ivec_tが同サイズ
	(sizeof(dvec_t) == sizeof(char)),	// または dvec_t が未定義なら...
	float, double>::type
			complementary_type;	//!< 相互変換可能な浮動小数点数
    typedef void	complementary_mask_type;
};
    
template <>
struct type_traits<int64_t>
{
    typedef u_int64_t	mask_type;
    typedef int64_t	signed_type;
    typedef u_int64_t	unsigned_type;
    typedef int32_t	lower_type;
    typedef void	upper_type;
    typedef double	complementary_type;
    typedef void	complementary_mask_type;
};
    
template <>
struct type_traits<u_int8_t>
{
    typedef u_int8_t	mask_type;
    typedef int8_t	signed_type;
    typedef u_int8_t	unsigned_type;
    typedef void	lower_type;
    typedef u_int16_t	upper_type;
    typedef float	complementary_type;
    typedef float	complementary_mask_type;
};
    
template <>
struct type_traits<u_int16_t>
{
    typedef u_int16_t	mask_type;
    typedef int16_t	signed_type;
    typedef u_int16_t	unsigned_type;
    typedef u_int8_t	lower_type;
    typedef u_int32_t	upper_type;
    typedef float	complementary_type;
    typedef float	complementary_mask_type;
};
    
template <>
struct type_traits<u_int32_t>
{
    typedef u_int32_t	mask_type;
    typedef int32_t	signed_type;
    typedef u_int32_t	unsigned_type;
    typedef u_int16_t	lower_type;
    typedef u_int64_t	upper_type;
    typedef float	complementary_type;
    typedef typename boost::mpl::if_c<
	sizeof(ivec_t) == sizeof(fvec_t),
	float, double>::type
			complementary_mask_type;
};
    
template <>
struct type_traits<u_int64_t>
{
    typedef u_int64_t	mask_type;
    typedef int64_t	signed_type;
    typedef u_int64_t	unsigned_type;
    typedef u_int32_t	lower_type;
    typedef void	upper_type;
    typedef double	complementary_type;
    typedef double	complementary_mask_type;
};

template <>
struct type_traits<float>
{
    typedef float	mask_type;
    typedef int32_t	signed_type;
    typedef u_int32_t	unsigned_type;
    typedef void	lower_type;
    typedef double	upper_type;
    typedef typename boost::mpl::if_c<
	sizeof(ivec_t) == sizeof(fvec_t),
	int32_t, int16_t>::type
			complementary_type;
    typedef typename boost::mpl::if_c<
	sizeof(ivec_t) == sizeof(fvec_t),
	u_int32_t, u_int16_t>::type
			complementary_mask_type;
};

template <>
struct type_traits<double>
{
    typedef double	mask_type;
    typedef int64_t	signed_type;
    typedef u_int64_t	unsigned_type;
    typedef float	lower_type;
    typedef void	upper_type;
    typedef int32_t	complementary_type;
    typedef typename boost::mpl::if_c<
	sizeof(ivec_t) == sizeof(dvec_t),
	u_int64_t, u_int32_t>::type
			complementary_mask_type;
};

/************************************************************************
*  class vec<T>								*
************************************************************************/
//! T型整数の成分を持つSIMDベクトルを表すクラス
template <class T>
class vec
{
  public:
  //! 成分の型    
    typedef T							element_type;
  //! ベースとなるSIMDデータ型
    typedef typename boost::mpl::if_<
	boost::is_same<T, double>, dvec_t,
	typename boost::mpl::if_<
	    boost::is_same<T, float>,
	    fvec_t, ivec_t>::type>::type			base_type;
    typedef vec<typename type_traits<T>::mask_type>		mask_type;
    
    enum	{element_size = sizeof(element_type),
		 size	      = sizeof(base_type)/sizeof(element_type),
		 lane_size    = (sizeof(base_type) > 16 ?
				 16/sizeof(element_type) : size)};

    vec()					{}
    vec(element_type a)				;
    vec(element_type a1,  element_type a0)	;
    vec(element_type a3,  element_type a2,
	element_type a1,  element_type a0)	;
    vec(element_type a7,  element_type a6,
	element_type a5,  element_type a4,
	element_type a3,  element_type a2,
	element_type a1,  element_type a0)	;
    vec(element_type a15, element_type a14,
	element_type a13, element_type a12,
	element_type a11, element_type a10,
	element_type a9,  element_type a8,
	element_type a7,  element_type a6,
	element_type a5,  element_type a4,
	element_type a3,  element_type a2,
	element_type a1,  element_type a0)	;
    vec(element_type a31, element_type a30,
	element_type a29, element_type a28,
	element_type a27, element_type a26,
	element_type a25, element_type a24,
	element_type a23, element_type a22,
	element_type a21, element_type a20,
	element_type a19, element_type a18,
	element_type a17, element_type a16,
	element_type a15, element_type a14,
	element_type a13, element_type a12,
	element_type a11, element_type a10,
	element_type a9,  element_type a8,
	element_type a7,  element_type a6,
	element_type a5,  element_type a4,
	element_type a3,  element_type a2,
	element_type a1,  element_type a0)	;
    
  // ベース型との間の型変換
    vec(base_type m)	:_base(m)	{}
		operator base_type()	{ return _base; }

    vec&	flip_sign()		{ return *this = -*this; }
    vec&	operator +=(vec x)	{ return *this = *this + x; }
    vec&	operator -=(vec x)	{ return *this = *this - x; }
    vec&	operator *=(vec x)	{ return *this = *this * x; }
    vec&	operator &=(vec x)	{ return *this = *this & x; }
    vec&	operator |=(vec x)	{ return *this = *this | x; }
    vec&	operator ^=(vec x)	{ return *this = *this ^ x; }
    vec&	andnot(vec x)		{ return *this = mm::andnot(x, *this); }
    template <class S>
    vec&	operator &=(typename boost::enable_if<
				boost::is_same<vec<S>, mask_type>,
				mask_type>::type x)
		{
		    return *this = *this & x;
		}
    template <class S>
    vec&	operator |=(typename boost::enable_if<
				boost::is_same<vec<S>, mask_type>,
				mask_type>::type x)
		{
		    return *this = *this | x;
		}
    template <class S>
    vec&	operator ^=(typename boost::enable_if<
				boost::is_same<vec<S>, mask_type>,
				mask_type>::type x)
		{
		    return *this = *this ^ x;
		}
    template <class S>
    vec&	andnot(typename boost::enable_if<
			   boost::is_same<vec<S>, mask_type>,
			   mask_type>::type x)
		{
		    return *this = mm::andnot(x, *this);
		}
    element_type	operator [](size_t i) const
		{
		    assert(i < size);
		    return *((element_type*)&_base + i);
		}
    element_type&	operator [](size_t i)
		{
		    assert(i < size);
		    return *((element_type*)&_base + i);
		}
    
    static size_t	floor(size_t n)	{ return size*(n/size); }
    static size_t	ceil(size_t n)	{ return (n == 0 ? 0 :
						  size*((n - 1)/size + 1)); }

  private:
    base_type		_base;
};

template <class T> static inline vec<T>&
operator &=(vec<T>& x,
	    typename boost::disable_if<
		boost::is_same<typename vec<T>::mask_type, vec<T> >,
		typename vec<T>::mask_type>::type y)
{
    return x = x & y;
}

template <class T> static inline vec<T>&
operator |=(vec<T>& x,
	    typename boost::disable_if<
		boost::is_same<typename vec<T>::mask_type, vec<T> >,
		typename vec<T>::mask_type>::type y)
{
    return x = x | y;
}

template <class T> static inline vec<T>&
operator ^=(vec<T>& x,
	    typename boost::disable_if<
		boost::is_same<typename vec<T>::mask_type, vec<T> >,
		typename vec<T>::mask_type>::type y)
{
    return x = x ^ y;
}

typedef vec<int8_t>	Is8vec;		//!< 符号付き8bit整数ベクトル
typedef vec<int16_t>	Is16vec;	//!< 符号付き16bit整数ベクトル
typedef vec<int32_t>	Is32vec;	//!< 符号付き32bit整数ベクトル
typedef vec<int64_t>	Is64vec;	//!< 符号付き64bit整数ベクトル
typedef vec<u_int8_t>	Iu8vec;		//!< 符号なし8bit整数ベクトル
typedef vec<u_int16_t>	Iu16vec;	//!< 符号なし16bit整数ベクトル
typedef vec<u_int32_t>	Iu32vec;	//!< 符号なし32bit整数ベクトル
typedef vec<u_int64_t>	Iu64vec;	//!< 符号なし64bit整数ベクトル
#if defined(SSE)
typedef vec<float>	F32vec;		//!< 32bit浮動小数点数ベクトル
#  if defined(SSE2)
typedef vec<double>	F64vec;		//!< 64bit浮動小数点数ベクトル
#  endif
#endif

//! SIMDベクトルの内容をストリームに出力する．
/*!
  \param out	出力ストリーム
  \param vec	SIMDベクトル
  \return	outで指定した出力ストリーム
*/
template <class T> std::ostream&
operator <<(std::ostream& out, const vec<T>& x)
{
    typedef typename boost::mpl::if_c<
	(boost::is_same<T, int8_t  >::value ||
	 boost::is_same<T, u_int8_t>::value), int32_t, T>::type	element_type;

    for (size_t i = 0; i < vec<T>::size; ++i)
	out << ' ' << element_type(x[i]);

    return out;
}

/************************************************************************
*  Predicates for boost::mpl						*
************************************************************************/
//! 与えられた型が何らかの mm::vec であるかを判定する boost::mpl 用の predicate
template <class T> struct is_vec		{ enum { value = false }; };
template <class T> struct is_vec<vec<T> >	{ enum { value = true  }; };

/************************************************************************
*  Macros for constructing mnemonics of intrinsics			*
************************************************************************/
#define MM_PREFIX(type)		MM_PREFIX_##type
#define MM_SUFFIX(type)		MM_SUFFIX_##type
#define MM_SIGNED(type)		MM_SIGNED_##type
#define MM_BASE(type)		MM_BASE_##type

#if defined(AVX2)
#  define MM_PREFIX_int8_t	_mm256_
#  define MM_PREFIX_int16_t	_mm256_
#  define MM_PREFIX_int32_t	_mm256_
#  define MM_PREFIX_int64_t	_mm256_
#  define MM_PREFIX_u_int8_t	_mm256_
#  define MM_PREFIX_u_int16_t	_mm256_
#  define MM_PREFIX_u_int32_t	_mm256_
#  define MM_PREFIX_u_int64_t	_mm256_
#  define MM_PREFIX_ivec_t	_mm256_
#else
#  define MM_PREFIX_int8_t	_mm_
#  define MM_PREFIX_int16_t	_mm_
#  define MM_PREFIX_int32_t	_mm_
#  define MM_PREFIX_int64_t	_mm_
#  define MM_PREFIX_u_int8_t	_mm_
#  define MM_PREFIX_u_int16_t	_mm_
#  define MM_PREFIX_u_int32_t	_mm_
#  define MM_PREFIX_u_int64_t	_mm_
#  define MM_PREFIX_ivec_t	_mm_
#endif

#if defined(SSE2)
#  define MM_SUFFIX_int8_t	epi8
#  define MM_SUFFIX_int16_t	epi16
#  define MM_SUFFIX_int32_t	epi32
#  define MM_SUFFIX_int64_t	epi64
#  define MM_SUFFIX_u_int8_t	epu8
#  define MM_SUFFIX_u_int16_t	epu16
#  define MM_SUFFIX_u_int32_t	epu32
#  define MM_SUFFIX_u_int64_t	epi64
#  if defined(AVX2)
#    define MM_SUFFIX_ivec_t	si256
#  else
#    define MM_SUFFIX_ivec_t	si128
#  endif
#else
#  define MM_SUFFIX_int8_t	pi8
#  define MM_SUFFIX_int16_t	pi16
#  define MM_SUFFIX_int32_t	pi32
#  define MM_SUFFIX_int64_t	si64
#  define MM_SUFFIX_u_int8_t	pu8
#  define MM_SUFFIX_u_int16_t	pu16
#  define MM_SUFFIX_u_int32_t	pu32
#  define MM_SUFFIX_u_int64_t	si64
#  define MM_SUFFIX_ivec_t	si64
#endif
#define MM_SUFFIX_void

#define MM_SIGNED_int8_t	MM_SUFFIX_int8_t
#define MM_SIGNED_int16_t	MM_SUFFIX_int16_t
#define MM_SIGNED_int32_t	MM_SUFFIX_int32_t
#define MM_SIGNED_int64_t	MM_SUFFIX_int64_t
#define MM_SIGNED_u_int8_t	MM_SUFFIX_int8_t
#define MM_SIGNED_u_int16_t	MM_SUFFIX_int16_t
#define MM_SIGNED_u_int32_t	MM_SUFFIX_int32_t
#define MM_SIGNED_u_int64_t	MM_SUFFIX_int64_t
    
#define MM_BASE_int8_t		MM_SUFFIX_ivec_t
#define MM_BASE_int16_t		MM_SUFFIX_ivec_t
#define MM_BASE_int32_t		MM_SUFFIX_ivec_t
#define MM_BASE_int64_t		MM_SUFFIX_ivec_t
#define MM_BASE_u_int8_t	MM_SUFFIX_ivec_t
#define MM_BASE_u_int16_t	MM_SUFFIX_ivec_t
#define MM_BASE_u_int32_t	MM_SUFFIX_ivec_t
#define MM_BASE_u_int64_t	MM_SUFFIX_ivec_t
#define MM_BASE_ivec_t		MM_SUFFIX_ivec_t

#if defined(AVX)
#  define MM_PREFIX_float	_mm256_
#  define MM_PREFIX_fvec_t	_mm256_
#  define MM_PREFIX_double	_mm256_
#  define MM_PREFIX_dvec_t	_mm256_
#elif defined(SSE)
#  define MM_PREFIX_float	_mm_
#  define MM_PREFIX_fvec_t	_mm_
#  if defined(SSE2)
#    define MM_PREFIX_double	_mm_
#    define MM_PREFIX_dvec_t	_mm_
#  endif
#endif

#if defined(SSE)
#  define MM_SUFFIX_float	ps
#  define MM_SUFFIX_fvec_t	ps
#  define MM_SIGNED_float	ps
#  define MM_BASE_float		ps
#  define MM_BASE_fvec_t	ps
#endif    
#if defined(SSE2)
#  define MM_SUFFIX_double	pd
#  define MM_SUFFIX_dvec_t	pd
#  define MM_SIGNED_double	pd
#  define MM_BASE_double	pd
#  define MM_BASE_dvec_t	pd
#endif    

#define MM_CAT(op, prefix, from, suffix)	prefix##op##from##_##suffix
#define MM_MNEMONIC(op, prefix, from, suffix)	MM_CAT(op, prefix, from, suffix)

#define MM_TMPL_FUNC(signature, op, args, from, to, suffix)		\
    inline signature							\
    {									\
	return MM_MNEMONIC(op, MM_PREFIX(to),				\
			   MM_SUFFIX(from), suffix(to))args;		\
    }

#define MM_FUNC(signature, op, args, from, to, suffix)			\
    template <> MM_TMPL_FUNC(signature, op, args, from, to, suffix)

#define MM_FUNC_2(func, op, type)					\
    MM_FUNC(vec<type> func(vec<type> x, vec<type> y),			\
	    op, (x, y), void, type, MM_SIGNED)

#define MM_BASE_FUNC_2(func, op, type)					\
    MM_FUNC(vec<type> func(vec<type> x, vec<type> y),			\
	    op, (x, y), void, type, MM_BASE)

#define MM_NUMERIC_FUNC_1(func, op, type)				\
    MM_FUNC(vec<type> func(vec<type> x),				\
	    op, (x), void, type, MM_SUFFIX)

#define MM_NUMERIC_FUNC_2(func, op, type)				\
    MM_FUNC(vec<type> func(vec<type> x, vec<type> y),			\
	    op, (x, y), void, type, MM_SUFFIX)

/************************************************************************
*  Constructors of vec<T>						*
************************************************************************/
#define MM_CONSTRUCTOR_1(type)						\
    inline								\
    vec<type>::vec(element_type a)					\
	:_base(MM_MNEMONIC(set1, MM_PREFIX(type), , MM_SIGNED(type))	\
	       (a))							\
    {									\
    }
#define MM_CONSTRUCTOR_2(type)						\
    inline								\
    vec<type>::vec(element_type a1, element_type a0)			\
	:_base(MM_MNEMONIC(set, MM_PREFIX(type), , MM_SIGNED(type))	\
	       (a1, a0))						\
    {									\
    }
#define MM_CONSTRUCTOR_4(type)						\
    inline								\
    vec<type>::vec(element_type a3, element_type a2,			\
		   element_type a1, element_type a0)			\
	:_base(MM_MNEMONIC(set,  MM_PREFIX(type), , MM_SIGNED(type))	\
	       (a3, a2, a1, a0))					\
    {									\
    }
#define MM_CONSTRUCTOR_8(type)						\
    inline								\
    vec<type>::vec(element_type a7, element_type a6,			\
		   element_type a5, element_type a4,			\
		   element_type a3, element_type a2,			\
		   element_type a1, element_type a0)			\
	:_base(MM_MNEMONIC(set,  MM_PREFIX(type), , MM_SIGNED(type))	\
	       (a7, a6, a5, a4,	a3, a2, a1, a0))			\
    {									\
    }
#define MM_CONSTRUCTOR_16(type)						\
    inline								\
    vec<type>::vec(element_type a15, element_type a14,			\
		   element_type a13, element_type a12,			\
		   element_type a11, element_type a10,			\
		   element_type a9,  element_type a8,			\
		   element_type a7,  element_type a6,			\
		   element_type a5,  element_type a4,			\
		   element_type a3,  element_type a2,			\
		   element_type a1,  element_type a0)			\
	:_base(MM_MNEMONIC(set,  MM_PREFIX(type), , MM_SIGNED(type))	\
	       (a15, a14, a13, a12, a11, a10, a9, a8,			\
		a7,  a6,  a5,  a4,  a3,  a2,  a1, a0))			\
    {									\
    }
#define MM_CONSTRUCTOR_32(type)						\
    inline								\
    vec<type>::vec(element_type a31, element_type a30,			\
		   element_type a29, element_type a28,			\
		   element_type a27, element_type a26,			\
		   element_type a25, element_type a24,			\
		   element_type a23, element_type a22,			\
		   element_type a21, element_type a20,			\
		   element_type a19, element_type a18,			\
		   element_type a17, element_type a16,			\
		   element_type a15, element_type a14,			\
		   element_type a13, element_type a12,			\
		   element_type a11, element_type a10,			\
		   element_type a9,  element_type a8,			\
		   element_type a7,  element_type a6,			\
		   element_type a5,  element_type a4,			\
		   element_type a3,  element_type a2,			\
		   element_type a1,  element_type a0)			\
	:_base(MM_MNEMONIC(set,  MM_PREFIX(type), , MM_SIGNED(type))	\
	       (a31, a30, a29, a28, a27, a26, a25, a24,			\
		a23, a22, a21, a20, a19, a18, a17, a16,			\
		a15, a14, a13, a12, a11, a10, a9,  a8,			\
		a7,  a6,  a5,  a4,  a3,  a2,  a1,  a0))			\
    {									\
    }

MM_CONSTRUCTOR_1(int8_t)
MM_CONSTRUCTOR_1(int16_t)
MM_CONSTRUCTOR_1(int32_t)
MM_CONSTRUCTOR_1(u_int8_t)
MM_CONSTRUCTOR_1(u_int16_t)
MM_CONSTRUCTOR_1(u_int32_t)

#if defined(AVX2)
  MM_CONSTRUCTOR_32(int8_t)	
  MM_CONSTRUCTOR_16(int16_t)	
  MM_CONSTRUCTOR_8(int32_t)
  MM_CONSTRUCTOR_32(u_int8_t)	
  MM_CONSTRUCTOR_16(u_int16_t)	
  MM_CONSTRUCTOR_8(u_int32_t)
#elif defined(SSE2)
  MM_CONSTRUCTOR_16(int8_t)
  MM_CONSTRUCTOR_8(int16_t)
  MM_CONSTRUCTOR_4(int32_t)
  MM_CONSTRUCTOR_16(u_int8_t)
  MM_CONSTRUCTOR_8(u_int16_t)
  MM_CONSTRUCTOR_4(u_int32_t)
#else
  MM_CONSTRUCTOR_8(int8_t)
  MM_CONSTRUCTOR_4(int16_t)
  MM_CONSTRUCTOR_2(int32_t)
  MM_CONSTRUCTOR_8(u_int8_t)
  MM_CONSTRUCTOR_4(u_int16_t)
  MM_CONSTRUCTOR_2(u_int32_t)
#endif

#if defined(SSE)
  MM_CONSTRUCTOR_1(float)
#  if defined(AVX)
  MM_CONSTRUCTOR_8(float)
#  else
  MM_CONSTRUCTOR_4(float)
#  endif
#endif

#if defined(SSE2)
  MM_CONSTRUCTOR_1(double)
#  if defined(AVX) 
  MM_CONSTRUCTOR_4(double)
#  else
  MM_CONSTRUCTOR_2(double)
#  endif
#endif

#undef MM_CONSTRUCTOR_1
#undef MM_CONSTRUCTOR_2
#undef MM_CONSTRUCTOR_4
#undef MM_CONSTRUCTOR_8
#undef MM_CONSTRUCTOR_16
#undef MM_CONSTRUCTOR_32

/************************************************************************
*  Load/Store								*
************************************************************************/
//! メモリからベクトルをロードする．
/*!
  \param p	ロード元のメモリアドレス
  \return	ロードされたベクトル
*/
template <bool ALIGNED=false, class T>
static vec<T>	load(const T* p)					;

//! メモリにベクトルをストアする．
/*!
  \param p	ストア先のメモリアドレス
  \param x	ストアされるベクトル
*/
template <bool ALIGNED=false, class T>
static void	store(T* p, vec<T> x)					;

#if defined(SSE2)
#  if defined(SSE3)
#    define MM_LOAD_STORE(type)						\
      MM_FUNC(vec<type> load<true>(const type* p), load,		\
	      ((const vec<type>::base_type*)p), void, type, MM_BASE)	\
      MM_FUNC(vec<type> load<false>(const type* p), lddqu,		\
	      ((const vec<type>::base_type*)p), void, type, MM_BASE)	\
      MM_FUNC(void store<true>(type* p, vec<type> x), store,		\
	      ((vec<type>::base_type*)p, x), void, type, MM_BASE)	\
      MM_FUNC(void store<false>(type* p, vec<type> x), storeu,		\
	      ((vec<type>::base_type*)p, x), void, type, MM_BASE)
#  else
#    define MM_LOAD_STORE(type)						\
      MM_FUNC(vec<type> load<true>(const type* p), load,		\
	      ((const vec<type>::base_type*)p), void, type, MM_BASE)	\
      MM_FUNC(vec<type> load<false>(const type* p), loadu,		\
	      ((const vec<type>::base_type*)p), void, type, MM_BASE)	\
      MM_FUNC(void store<true>(type* p, vec<type> x), store,		\
	      ((vec<type>::base_type*)p, x), void, type, MM_BASE)	\
      MM_FUNC(void store<false>(type* p, vec<type> x), storeu,		\
	      ((vec<type>::base_type*)p, x), void, type, MM_BASE)
#  endif
  MM_LOAD_STORE(int8_t)
  MM_LOAD_STORE(int16_t)
  MM_LOAD_STORE(int32_t)
  MM_LOAD_STORE(int64_t)
  MM_LOAD_STORE(u_int8_t)
  MM_LOAD_STORE(u_int16_t)
  MM_LOAD_STORE(u_int32_t)
  MM_LOAD_STORE(u_int64_t)

#  undef MM_LOAD_STORE  
#else
  template <bool ALIGNED, class T> inline vec<T>
  load(const T* p)
  {
      return *((const typename vec<T>::base_type*)p);
  }
  template <bool ALIGNED, class T> inline void
  store(T* p, vec<T> x)
  {
      *((typename vec<T>::base_type*)p) = x;
  }
#endif

#if defined(SSE)
#  define MM_LOAD_STORE(type)						\
    MM_FUNC(vec<type> load<true>(const type* p), load,			\
	    (p), void, type, MM_BASE)					\
    MM_FUNC(vec<type> load<false>(const type* p), loadu,		\
	    (p), void, type, MM_BASE)					\
    MM_FUNC(void store<true>(type* p, vec<type> x), store,		\
	    (p, x), void, type, MM_BASE)				\
    MM_FUNC(void store<false>(type* p, vec<type> x), storeu,		\
	    (p, x), void, type, MM_BASE)

  MM_LOAD_STORE(float)
#  if defined(SSE2)
  MM_LOAD_STORE(double)
#  endif
#  undef MM_LOAD_STORE
#endif
  
/************************************************************************
*  Zero-vector generators						*
************************************************************************/
//! 全成分が0であるベクトルを生成する．
template <class T> static vec<T>	zero()				;

#define MM_ZERO(type)							\
    MM_FUNC(vec<type> zero<type>(), setzero, (), void, type, MM_BASE)

MM_ZERO(int8_t)
MM_ZERO(int16_t)
MM_ZERO(int32_t)
MM_ZERO(int64_t)
MM_ZERO(u_int8_t)
MM_ZERO(u_int16_t)
MM_ZERO(u_int32_t)
MM_ZERO(u_int64_t)
    
#if defined(SSE)
  MM_ZERO(float)
#endif
#if defined(SSE2)
  MM_ZERO(double)
#endif

#undef MM_ZERO

/************************************************************************
*  Cast operators							*
************************************************************************/
template <class S, class T> static S	cast_base(T x)			;

//! T型の成分を持つベクトルからS型の成分を持つベクトルへのキャストを行なう．
template <class S, class T> static inline vec<S>
cast(vec<T> x)
{
    return
	cast_base<typename vec<S>::base_type>(typename vec<T>::base_type(x));
}

/*
 *  cast_base() の実装
 */
// 整数 <-> 整数
template <> inline ivec_t
cast_base(ivec_t x)
{
    return x;
}

// 整数 <-> float, double
#if !defined(AVX2) && defined(AVX)
  template <> fvec_t
  cast_base<fvec_t>(ivec_t x)
  {
      return _mm256_castsi256_ps(_mm256_castsi128_si256(x));
  }

  template <> ivec_t
  cast_base<ivec_t>(fvec_t x)
  {
      return _mm256_castsi256_si128(_mm256_castps_si256(x));
  }

  template <> dvec_t
  cast_base<dvec_t>(ivec_t x)
  {
      return _mm256_castsi256_pd(_mm256_castsi128_si256(x));
  }

  template <> ivec_t
  cast_base<ivec_t>(dvec_t x)
  {
      return _mm256_castsi256_si128(_mm256_castpd_si256(x));
  }
#elif defined(SSE2)
  MM_FUNC(fvec_t cast_base<fvec_t>(ivec_t x),
	  cast, (x), ivec_t, fvec_t, MM_BASE)
  MM_FUNC(ivec_t cast_base<ivec_t>(fvec_t x),
	  cast, (x), fvec_t, ivec_t, MM_BASE)
  MM_FUNC(dvec_t cast_base<dvec_t>(ivec_t x),
	  cast, (x), ivec_t, dvec_t, MM_BASE)
  MM_FUNC(ivec_t cast_base<ivec_t>(dvec_t x),
	  cast, (x), dvec_t, ivec_t, MM_BASE)
#endif

// float <-> double
#if defined(SSE2)
  MM_FUNC(dvec_t cast_base<dvec_t>(fvec_t x),
	  cast, (x), fvec_t, dvec_t, MM_BASE)
  MM_FUNC(fvec_t cast_base<fvec_t>(dvec_t x),
	  cast, (x), dvec_t, fvec_t, MM_BASE)
#endif
  
/************************************************************************
*  Shuffle operators							*
************************************************************************/
//! 8つの成分を持つ整数ベクトルの下位4成分をシャッフルする．
/*!
  上位4成分は変化しない．
  \param I0	最下位に来る成分のindex (0 <= I0 < 4)
  \param I1	下から2番目に来る成分のindex (0 <= I1 < 4)
  \param I2	下から3番目に来る成分のindex (0 <= I2 < 4)
  \param I3	下から4番目に来る成分のindex (0 <= I3 < 4)
  \param x	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t I3, size_t I2, size_t I1, size_t I0, class T> static vec<T>
shuffle_low(vec<T> x)							;

//! 8つの成分を持つ整数ベクトルの上位4成分をシャッフルする．
/*!
  下位4成分は変化しない．
  \param I0	下から5番目に来る成分のindex (0 <= I0 < 4)
  \param I1	下から6番目に来る成分のindex (0 <= I1 < 4)
  \param I2	下から7番目に来る成分のindex (0 <= I2 < 4)
  \param I3	最上位に来る成分のindex (0 <= I3 < 4)
  \param x	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t I3, size_t I2, size_t I1, size_t I0, class T> static vec<T>
shuffle_high(vec<T> x)							;

//! 4つの成分を持つ整数ベクトルの成分をシャッフルする．
/*!
  \param I0	最下位に来る成分のindex (0 <= I0 < 4)
  \param I1	下から2番目に来る成分のindex (0 <= I1 < 4)
  \param I2	下から3番目に来る成分のindex (0 <= I2 < 4)
  \param I3	最上位に来る成分のindex (0 <= I3 < 4)
  \param x	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t I3, size_t I2, size_t I1, size_t I0, class T> static vec<T>
shuffle(vec<T> x)							;

#define MM_SHUFFLE_LOW_HIGH_I4(type)					\
    template <size_t I3, size_t I2, size_t I1, size_t I0>		\
    MM_TMPL_FUNC(vec<type> shuffle_low(vec<type> x),			\
		 shufflelo, (x, _MM_SHUFFLE(I3, I2, I1, I0)),		\
		 void, type, MM_SIGNED)					\
    template <size_t I3, size_t I2, size_t I1, size_t I0>		\
    MM_TMPL_FUNC(vec<type> shuffle_high(vec<type> x),			\
		 shufflehi, (x, _MM_SHUFFLE(I3, I2, I1, I0)),		\
		 void, type, MM_SIGNED)
#define MM_SHUFFLE_I4(type)						\
    template <size_t I3, size_t I2, size_t I1, size_t I0>		\
    MM_TMPL_FUNC(vec<type> shuffle(vec<type> x),			\
		 shuffle, (x, _MM_SHUFFLE(I3, I2, I1, I0)),		\
		 void, type, MM_SIGNED)

#if defined(SSE2)
  MM_SHUFFLE_I4(int32_t)
  MM_SHUFFLE_I4(u_int32_t)
  MM_SHUFFLE_LOW_HIGH_I4(int16_t)
  MM_SHUFFLE_LOW_HIGH_I4(u_int16_t)
#elif defined(SSE)
  MM_SHUFFLE_I4(int16_t)
  MM_SHUFFLE_I4(u_int16_t)
#endif
  
#undef MM_SHUFFLE_LOW_HIGH_I4
#undef MM_SHUFFLE_I4

//! 4つの成分を持つ2つの浮動小数点数ベクトルの成分をシャッフルする．
/*!
  下位2成分はxから，上位2成分はyからそれぞれ選択される．
  \param Xl	最下位に来るベクトルxの成分のindex (0 <= Xl < 4)
  \param Xh	下から2番目に来るベクトルxの成分のindex (0 <= Xh < 4)
  \param Yl	下から3番目に来るベクトルyの成分のindex (0 <= Yl < 4)
  \param Yh	最上位に来るベクトルyの成分のindex (0 <= Yh < 4)
  \param x	シャッフルされるベクトル
  \param y	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t Yh, size_t Yl, size_t Xh, size_t Xl, class T> static vec<T>
shuffle(vec<T> x, vec<T> y)						;

//! 2つの成分を持つ2つの浮動小数点数ベクトルの成分をシャッフルする．
/*!
  下位成分はxから，上位成分はyからそれぞれ選択される．
  \param X	下位に来るベクトルxの成分のindex (0 <= I0 < 2)
  \param Y	上位に来るベクトルyの成分のindex (0 <= I3 < 2)
  \param x	シャッフルされるベクトル
  \param y	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t Y, size_t X, class T> static vec<T>
shuffle(vec<T> x, vec<T> y)						;

#define _MM_SHUFFLE4(i3, i2, i1, i0)					\
    (((i3) << 3) | ((i2) << 2) | ((i1) << 1) | (i0))
#define MM_SHUFFLE_F4(type)						\
    template <size_t Yh, size_t Yl, size_t Xh, size_t Xl>		\
    MM_TMPL_FUNC(vec<type> shuffle(vec<type> x, vec<type> y),		\
		 shuffle, (x, y, _MM_SHUFFLE(Yh, Yl, Xh, Xl)),		\
		 void, type, MM_SUFFIX)
#define MM_SHUFFLE_D4(type)						\
    template <size_t Yh, size_t Yl, size_t Xh, size_t Xl>		\
    MM_TMPL_FUNC(vec<type> shuffle(vec<type> x, vec<type> y),		\
		 shuffle, (x, y, _MM_SHUFFLE4(Yh, Yl, Xh, Xl)),		\
		 void, type, MM_SUFFIX)
#define MM_SHUFFLE_D2(type)						\
    template <size_t Y, size_t X>					\
    MM_TMPL_FUNC(vec<type> shuffle(vec<type> x, vec<type> y),		\
		 shuffle, (x, y, _MM_SHUFFLE2(Y, X)),			\
		 void, type, MM_SUFFIX)

#if defined(SSE)
  MM_SHUFFLE_F4(float)
#  if defined(AVX)
  MM_SHUFFLE_D4(double)
#  elif defined(SSE2)
  MM_SHUFFLE_D2(double)
#  endif
#endif

#undef MM_SHUFFLE_D2
#undef MM_SHUFFLE_F4

/************************************************************************
*  全成分にN番目の要素をセット						*
************************************************************************/
//! 与えられたベクトルの指定された成分を全成分にセットしたベクトルを生成する．
/*!
  与えられたベクトルの成分数は2または4でなければならない．
  \param N	セットするxのベクトルの成分を指定するindex (0 <= N < 4)
  \param x	2つ，または4つの成分を持つベクトル
  \return	生成されたベクトル
*/
template <size_t N, class T> static inline vec<T>
set1(vec<T> x)
{
      return shuffle<N, N, N, N>(x);
}

#if defined(SSE)
  template <size_t N> static inline F32vec
  set1(F32vec x)		{return shuffle<N, N, N, N>(x, x);}
#  if defined(AVX)
  template <size_t N> static inline F64vec
  set1(F64vec x)		{return shuffle<N, N, N, N>(x, x);}
#  elif defined(SSE2)
  template <size_t N> static inline F64vec
  set1(F64vec x)		{return shuffle<N, N>(x, x);}
#  endif
#endif

/************************************************************************
*  Unpack operators							*
************************************************************************/
//! 2つのベクトルの下位半分の成分を交互に混合する．
/*!
  \param x	その成分を偶数番目に配置するベクトル
  \param y	その成分を奇数番目に配置するベクトル
  \return	生成されたベクトル
*/
template <class T> static vec<T>	unpack_low(vec<T> x, vec<T> y)	;

//! 2つのベクトルの上位半分の成分を交互に混合する．
/*!
  \param x	その成分を偶数番目に配置するベクトル
  \param y	その成分を奇数番目に配置するベクトル
  \return	生成されたベクトル
*/
template <class T> static vec<T>	unpack_high(vec<T> x, vec<T> y)	;

#define MM_UNPACK_LOW_HIGH(type)					\
    MM_FUNC_2(unpack_low,  unpacklo, type)				\
    MM_FUNC_2(unpack_high, unpackhi, type)

MM_UNPACK_LOW_HIGH(int8_t)
MM_UNPACK_LOW_HIGH(int16_t)
MM_UNPACK_LOW_HIGH(int32_t)
MM_UNPACK_LOW_HIGH(u_int8_t)
MM_UNPACK_LOW_HIGH(u_int16_t)
MM_UNPACK_LOW_HIGH(u_int32_t)
#if defined(SSE)
  MM_UNPACK_LOW_HIGH(float)
#  if defined(SSE2)
  MM_UNPACK_LOW_HIGH(int64_t)
  MM_UNPACK_LOW_HIGH(u_int64_t)
  MM_UNPACK_LOW_HIGH(double)
#  endif
#endif

#undef MM_UNPACK_LOW_HIGH

/************************************************************************
*  N-tuple generators							*
************************************************************************/
// 複製数：N = 2, 4, 8, 16,...;
// 全体をN個の部分に分けたときの複製区間：0 <= I < N
template <size_t N, size_t I, class T> static vec<T>	n_tuple(vec<T> x);

template <size_t I, class T> static inline vec<T>
dup(vec<T> x)
{
    return n_tuple<2, I>(x);
}

template <size_t I, class T> static inline vec<T>
quadup(vec<T> x)
{
    return n_tuple<4, I>(x);
}
    
template <size_t I, class T> static inline vec<T>
octup(vec<T> x)
{
    return n_tuple<8, I>(x);
}
    
#define MM_N_TUPLE(type)						\
    template <> inline vec<type>					\
    n_tuple<2, 0>(vec<type> x)		{return unpack_low(x, x);}	\
    template <> inline vec<type>					\
    n_tuple<2, 1>(vec<type> x)		{return unpack_high(x, x);}

template <size_t N, size_t I, class T> inline vec<T>
n_tuple(vec<T> x)
{
    return n_tuple<2, (I&0x1)>(n_tuple<(N>>1), (I>>1)>(x));
}

MM_N_TUPLE(int8_t)
MM_N_TUPLE(int16_t)
MM_N_TUPLE(int32_t)
MM_N_TUPLE(u_int8_t)
MM_N_TUPLE(u_int16_t)
MM_N_TUPLE(u_int32_t)
#if defined(SSE)
  MM_N_TUPLE(float)
#  if defined(SSE2)
  MM_N_TUPLE(int64_t)
  MM_N_TUPLE(u_int64_t)
  MM_N_TUPLE(double)
#  endif
#endif

#undef MM_N_TUPLE
    
#if defined(SSE)
/************************************************************************
*  Inserting/Extracting elements					*
************************************************************************/
//! ベクトルの指定された位置に成分を挿入する．
/*!
  \param I	挿入する位置を指定するindex
  \param x	ベクトル
  \return	成分を挿入されたベクトル
*/
template <size_t I, class T> static vec<T>   insert(vec<T> x, int val)	;

//! ベクトルから指定された位置から成分を取り出す．
/*!
  \param I	取り出す位置を指定するindex
  \param x	ベクトル
  \return	取り出された成分
*/
template <size_t I, class T> static int	    extract(vec<T> x)		;

#  if defined(AVX2)
#    define MM_INSERT_EXTRACT(type)					\
      template <size_t I> inline vec<type>				\
      insert(vec<type> x, int val)					\
      {									\
	  return _mm256_insertf128_si256(				\
		     x,							\
		     MM_MNEMONIC(insert, _mm_, , MM_SIGNED(type))(	\
			 _mm256_extractf128_si256(			\
			     x, I / vec<type>::lane_size),		\
			 val, I % vec<type>::lane_size),		\
		     I / vec<type>::lane_size);				\
      }									\
      template <size_t I> inline int					\
      extract(vec<type> x)						\
      {									\
	  return MM_MNEMONIC(extract, _mm_, , MM_SIGNED(type))(		\
		     _mm256_extractf128_si256(				\
			 x, I / vec<type>::lane_size),			\
		     I / vec<type>::lane_size);				\
      }

#  else
#    define MM_INSERT_EXTRACT(type)					\
      template <size_t I>		 				\
      MM_TMPL_FUNC(vec<type> insert(vec<type> x, int val), insert,	\
		   (x, val, I), void, type, MM_SIGNED)			\
      template <size_t I>						\
      MM_TMPL_FUNC(int extract(vec<type> x), extract,			\
		   (x, I), void, type, MM_SIGNED)
#  endif

  MM_INSERT_EXTRACT(int16_t)
  MM_INSERT_EXTRACT(u_int16_t)
#  if defined(SSE4)
    MM_INSERT_EXTRACT(int8_t)
    MM_INSERT_EXTRACT(u_int8_t)
    MM_INSERT_EXTRACT(int32_t)
    MM_INSERT_EXTRACT(u_int32_t)
#  endif

#  undef MM_INSERT_EXTRACT

#  if defined(AVX)
    template <size_t I> inline F32vec
    insert(F32vec x, float val)
    {
	return _mm256_insertf128_ps(x,
				    _mm_insert_ps(
					_mm256_extractf128_ps(
					    x, I / F32vec::lane_size),
					_mm_set_ss(val),
					(I % F32vec::lane_size) << 4),
				    I / F32vec::lane_size);
    }
#  elif defined(SSE4)
    template <size_t I> inline F32vec
    insert(F32vec x, float val)
    {
	return _mm_insert_ps(x, _mm_set_ss(val), I << 4);
    }
#  endif

template <size_t I> float	extract(F32vec x)			;
#  if defined(AVX)
    template <> inline float
    extract<0>(F32vec x)
    {
	return _mm_cvtss_f32(_mm256_extractf128_ps(x, 0x0));
    }

    template <> inline float
    extract<4>(F32vec x)
    {
	return _mm_cvtss_f32(_mm256_extractf128_ps(x, 0x1));
    }
#  else
    template <> inline float
    extract<0>(F32vec x)
    {
	return _mm_cvtss_f32(x);
    }
#  endif

#  if defined(SSE2)
    template <size_t I> double	extract(F64vec x)			;
#    if defined(AVX)
      template <> inline double
      extract<0>(F64vec x)
      {
	  return _mm_cvtsd_f64(_mm256_extractf128_pd(x, 0x0));
      }

      template <> inline double
      extract<2>(F64vec x)
      {
	  return _mm_cvtsd_f64(_mm256_extractf128_pd(x, 0x1));
      }
#    else
      template <> inline double
      extract<0>(F64vec x)
      {
	  return _mm_cvtsd_f64(x);
      }
#    endif
#  endif
#endif  

/************************************************************************
*  Elementwise shift operators						*
************************************************************************/
//! ベクトルの要素を左シフトする．
/*!
  シフト後の下位には0が入る．
  \param N	シフト数(成分単位)
  \param x	シフトされるベクトル
  \return	シフトされたベクトル
*/
template <size_t N, class T> static vec<T>	shift_l(vec<T> x)	;

//! ベクトルの要素を右シフトする．
/*!
  シフト後の上位には0が入る．
  \param N	シフト数(成分単位)
  \param x	シフトされるベクトル
  \return	シフトされたベクトル
*/
template <size_t N, class T> static vec<T>	shift_r(vec<T> x)	;

// 整数ベクトルの要素シフト（実装上の注意：MMXでは64bit整数のシフトは
// bit単位だが，SSE2以上の128bit整数ではbyte単位である．また，AVX2では
// 上下のlaneに分断されないemulationバージョンを使う．）
#if defined(AVX2)
#  define MM_ELM_SHIFTS_I(type)						\
    template <> inline vec<type>					\
    shift_l<0>(vec<type> x)				{return x;}	\
    template <> inline vec<type>					\
    shift_r<0>(vec<type> x)				{return x;}	\
    template <size_t N> vec<type>					\
    shift_l(vec<type> x)						\
    {									\
	return _mm256_emu_slli_si256<N*vec<type>::element_size>(x);	\
    }									\
    template <size_t N> vec<type>					\
    shift_r(vec<type> x)						\
    {									\
	return _mm256_emu_srli_si256<N*vec<type>::element_size>(x);	\
    }
#elif defined(SSE2)
#  define MM_ELM_SHIFTS_I(type)						\
    template <> inline vec<type>					\
    shift_l<0>(vec<type> x)				{return x;}	\
    template <> inline vec<type>					\
    shift_r<0>(vec<type> x)				{return x;}	\
    template <size_t N>							\
    MM_TMPL_FUNC(vec<type> shift_l(vec<type> x), slli,			\
		 (x, N*vec<type>::element_size), void, type, MM_BASE)	\
    template <size_t N>							\
    MM_TMPL_FUNC(vec<type> shift_r(vec<type> x), srli,			\
		 (x, N*vec<type>::element_size), void, type, MM_BASE)
#else
#  define MM_ELM_SHIFTS_I(type)						\
    template <> inline vec<type>					\
    shift_l<0>(vec<type> x)				{return x;}	\
    template <> inline vec<type>					\
    shift_r<0>(vec<type> x)				{return x;}	\
    template <size_t N>							\
    MM_TMPL_FUNC(vec<type> shift_l(vec<type> x), slli,			\
		 (x, 8*N*vec<type>::element_size), void, type, MM_BASE)	\
    template <size_t N>							\
    MM_TMPL_FUNC(vec<type> shift_r(vec<type> x), srli,			\
		 (x, 8*N*vec<type>::element_size), void, type, MM_BASE)
#endif

MM_ELM_SHIFTS_I(int8_t)
MM_ELM_SHIFTS_I(int16_t)
MM_ELM_SHIFTS_I(int32_t)
MM_ELM_SHIFTS_I(int64_t)
MM_ELM_SHIFTS_I(u_int8_t)
MM_ELM_SHIFTS_I(u_int16_t)
MM_ELM_SHIFTS_I(u_int32_t)
MM_ELM_SHIFTS_I(u_int64_t)

#undef MM_ELM_SHIFTS_I

// 浮動小数点数ベクトルの要素シフト
#if !defined(AVX2) && defined(AVX)
  template <size_t N> static inline F32vec
  shift_l(F32vec x)
  {
      return _mm256_castsi256_ps(
	_mm256_emu_slli_si256<N*F32vec::element_size>(_mm256_castps_si256(x)));
  }

  template <size_t N> static inline F32vec
  shift_r(F32vec x)
  {
      return _mm256_castsi256_ps(
	_mm256_emu_srli_si256<N*F32vec::element_size>(_mm256_castps_si256(x)));
  }

  template <size_t N> static inline F64vec
  shift_l(F64vec x)
  {
      return _mm256_castsi256_pd(
	_mm256_emu_slli_si256<N*F64vec::element_size>(_mm256_castpd_si256(x)));
  }

  template <size_t N> static inline F64vec
  shift_r(F64vec x)
  {
      return _mm256_castsi256_pd(
	_mm256_emu_srli_si256<N*F64vec::element_size>(_mm256_castpd_si256(x)));
  }
#elif defined(SSE2)
  template <size_t N> static inline F32vec
  shift_l(F32vec x)
  {
      return cast<float>(shift_l<N>(cast<u_int32_t>(x)));
  }

  template <size_t N> static inline F32vec
  shift_r(F32vec x)
  {
      return cast<float>(shift_r<N>(cast<u_int32_t>(x)));
  }

  template <size_t N> static inline F64vec
  shift_l(F64vec x)
  {
      return cast<double>(shift_l<N>(cast<u_int64_t>(x)));
  }

  template <size_t N> static inline F64vec
  shift_r(F64vec x)
  {
      return cast<double>(shift_r<N>(cast<u_int64_t>(x)));
  }
#endif

/************************************************************************
*  Elementwise concatinated shift operators				*
************************************************************************/
//! 2つのベクトルを連結した2倍長のベクトルの要素を右シフトした後，下位ベクトルを取り出す．
/*!
  シフト後の上位にはyの要素が入る．
  \param N	シフト数(成分単位), 0 <= N <= vec<T>::size
  \param y	上位のベクトル
  \param x	下位のベクトル
  \return	シフトされたベクトル
*/
template <size_t N, class T> static vec<T> shift_r(vec<T> y, vec<T> x)	;

#if defined(AVX2)
#  define MM_ELM_SHIFT_R_I2(type)					\
    template <size_t N>							\
    MM_TMPL_FUNC(vec<type> shift_r(vec<type> y, vec<type> x),		\
		 emu_alignr<N*vec<type>::element_size>, (y, x),		\
		 void, int8_t, MM_SIGNED)
#else
#  define MM_ELM_SHIFT_R_I2(type)					\
    template <size_t N>							\
    MM_TMPL_FUNC(vec<type> shift_r(vec<type> y, vec<type> x),		\
		 alignr, (y, x, N*vec<type>::element_size),		\
		 void, int8_t, MM_SIGNED)
#endif
MM_ELM_SHIFT_R_I2(int8_t)
MM_ELM_SHIFT_R_I2(int16_t)
MM_ELM_SHIFT_R_I2(int32_t)
MM_ELM_SHIFT_R_I2(int64_t)
MM_ELM_SHIFT_R_I2(u_int8_t)
MM_ELM_SHIFT_R_I2(u_int16_t)
MM_ELM_SHIFT_R_I2(u_int32_t)
MM_ELM_SHIFT_R_I2(u_int64_t)

#undef MM_ELM_SHIFT_R_I2

// 浮動小数点数ベクトルの要素シフト
#if !defined(AVX2) && defined(AVX)
  template <size_t N> static inline F32vec
  shift_r(F32vec y, F32vec x)
  {
      return _mm256_castsi256_ps(
	_mm256_emu_alignr_epi8<N*F32vec::element_size>(_mm256_castps_si256(y),
						       _mm256_castps_si256(x)));
  }

  template <size_t N> static inline F64vec
  shift_r(F64vec y, F64vec x)
  {
      return _mm256_castsi256_pd(
	_mm256_emu_alignr_epi8<N*F64vec::element_size>(_mm256_castpd_si256(y),
						       _mm256_castpd_si256(x)));
  }
#elif defined(SSE2)
  template <size_t N> static inline F32vec
  shift_r(F32vec y, F32vec x)
  {
      return cast<float>(shift_r<N>(cast<u_int32_t>(y), cast<u_int32_t>(x)));
  }

  template <size_t N> static inline F64vec
  shift_r(F64vec y, F64vec x)
  {
      return cast<double>(shift_r<N>(cast<u_int64_t>(y), cast<u_int64_t>(x)));
  }
#endif

//! 2つのベクトルを連結した2倍長のベクトルの要素を左シフトした後，上位ベクトルを取り出す．
/*!
  シフト後の下位にはxの要素が入る．
  \param N	シフト数(成分単位), 0 <= N <= vec<T>::size
  \param y	上位のベクトル
  \param x	下位のベクトル
  \return	シフトされたベクトル
*/
template <size_t N, class T> static inline vec<T>
shift_l(vec<T> y, vec<T> x)
{
    const u_int	SIZE = vec<T>::size;
    return shift_r<SIZE - N>(y, x);
}

/************************************************************************
*  Element wise shift to left/right-most				*
************************************************************************/
//! 左端の要素が右端に来るまで右シフトする．
/*!
  シフト後の上位には0が入る．
  \param x	シフトされるベクトル
  \return	シフトされたベクトル
*/
template <class T> static inline vec<T>
shift_lmost_to_rmost(vec<T> x)
{
    return shift_r<vec<T>::size-1>(x);
}

//! 右端の要素が左端に来るまで左シフトする．
/*!
  シフト後の下位には0が入る．
  \param x	シフトされるベクトル
  \return	シフトされたベクトル
*/
template <class T> static inline vec<T>
shift_rmost_to_lmost(vec<T> x)
{
    return shift_l<vec<T>::size-1>(x);
}

//! 与えられた値を右端の成分にセットし残りを0としたベクトルを生成する．
/*!
  \param x	セットされる値
  \return	xを右端成分とするベクトル
*/
template <class T> static inline vec<T>
set_rmost(typename vec<T>::element_type x)
{
    return shift_lmost_to_rmost(vec<T>(x));
}

/************************************************************************
*  Replacing rightmost/leftmost element of x with that of y		*
************************************************************************/
template <class T> static vec<T>	replace_rmost(vec<T> x, vec<T> y);
template <class T> static vec<T>	replace_lmost(vec<T> x, vec<T> y);

#if defined(SSE4)
#  define MM_REPLACE(type)						\
    MM_FUNC(vec<type> replace_rmost(vec<type> x, vec<type> y),		\
	    blend, (x, y, 0x01), void, type, MM_SUFFIX)			\
    MM_FUNC(vec<type> replace_lmost(vec<type> x, vec<type> y),		\
	    blend, (x, y, 0x01 << vec<type>::size - 1),			\
	    void, type, MM_SUFFIX)

  MM_REPLACE(float)
  MM_REPLACE(double)
#elif defined(SSE)
  static inline F32vec
  replace_rmost(F32vec x, F32vec y)	{return _mm_move_ss(x, y);}
#  if defined(SSE2)
    static inline F64vec
    replace_rmost(F64vec x, F64vec y)	{return _mm_move_sd(x, y);}
#  endif
#endif
    
/************************************************************************
*  Rotation and reverse operators					*
************************************************************************/
// SSSE3以上では _mm[256]_alignr_epi8 を使って実装するのが簡単だが，
// AVX2ではconcatinationに先立って128bit単位のunpackが行なわれてしまう．

//! ベクトルの左回転を行なう．
template <class T> static inline vec<T>
rotate_l(vec<T> x)			{return shuffle<2, 1, 0, 3>(x);}
//! ベクトルの右回転を行なう．
template <class T> static inline vec<T>
rotate_r(vec<T> x)			{return shuffle<0, 3, 2, 1>(x);}
//! ベクトルの逆転を行なう．
template <class T> static inline vec<T>
reverse(vec<T> x)			{return shuffle<0, 1, 2, 3>(x);}

#define MM_ROTATE_REVERSE_4(type)					\
    template <> inline vec<type>					\
    rotate_l(vec<type> x)	{return shuffle<2, 1, 0, 3>(x, x);}	\
    template <> inline vec<type>					\
    rotate_r(vec<type> x)	{return shuffle<0, 3, 2, 1>(x, x);}	\
    template <> inline vec<type>					\
    reverse(vec<type> x)	{return shuffle<0, 1, 2, 3>(x, x);}

#if defined(AVX)
  MM_ROTATE_REVERSE_4(double)
#elif defined(SSE)
  MM_ROTATE_REVERSE_4(float)
#  if defined(SSE2)
    template <> inline F64vec
    rotate_l(F64vec x)			{return shuffle<0, 1>(x, x);}
    template <> inline F64vec
    rotate_r(F64vec x)			{return rotate_l(x);}
    template <> inline F64vec
    reverse(F64vec x)			{return rotate_l(x);}
#  endif
#endif

#undef MM_ROTATE_REVERSE_4

/************************************************************************
*  Bitwise shift operators						*
************************************************************************/
//! 整数ベクトルの左シフトを行う．
/*!
  \param x	整数ベクトル
  \param n	シフトするビット数
  \return	シフト後の整数ベクトル
*/
template <class T> static vec<T>	operator <<(vec<T> x, int n)	;

//! 整数ベクトルの算術右シフトを行なう．
/*!
  \param x	整数ベクトル
  \param n	シフトするビット数
  \return	シフト後の整数ベクトル
*/
template <class T> static vec<T>	operator >>(vec<T> x, int n)	;

#define MM_LOGICAL_SHIFT_LEFT(type)					\
    MM_FUNC(vec<type> operator <<(vec<type> x, int n),			\
	    slli, (x, n), void, type, MM_SIGNED)
#define MM_LOGICAL_SHIFT_RIGHT(type)					\
    MM_FUNC(vec<type> operator >>(vec<type> x, int n),			\
	    srli, (x, n), void, type, MM_SIGNED)
#define MM_NUMERIC_SHIFT_RIGHT(type)					\
    MM_FUNC(vec<type> operator >>(vec<type> x, int n),			\
	    srai, (x, n), void, type, MM_SIGNED)

MM_LOGICAL_SHIFT_LEFT(int16_t)
MM_LOGICAL_SHIFT_LEFT(int32_t)
MM_LOGICAL_SHIFT_LEFT(int64_t)
MM_LOGICAL_SHIFT_LEFT(u_int16_t)
MM_LOGICAL_SHIFT_LEFT(u_int32_t)
MM_LOGICAL_SHIFT_LEFT(u_int64_t)

MM_NUMERIC_SHIFT_RIGHT(int16_t)
MM_NUMERIC_SHIFT_RIGHT(int32_t)
MM_LOGICAL_SHIFT_RIGHT(u_int16_t)
MM_LOGICAL_SHIFT_RIGHT(u_int32_t)
MM_LOGICAL_SHIFT_RIGHT(u_int64_t)

#undef MM_LOGICAL_SHIFT_LEFT
#undef MM_LOGICAL_SHIFT_RIGHT
#undef MM_NUMERIC_SHIFT_RIGHT

/************************************************************************
*  Type conversion operators						*
************************************************************************/
//! T型ベクトルのI番目の部分をS型ベクトルに型変換する．
/*!
  整数ベクトル間の変換の場合，SのサイズはTの2/4/8倍である．また，S, Tは
  符号付き／符号なしのいずれでも良いが，符号付き -> 符号なしの変換はできない．
  \param x	変換されるベクトル
  \return	変換されたベクトル
*/
template <class S, size_t I=0, class T> static inline vec<S>
cvt(vec<T> x)
{
    typedef typename type_traits<S>::lower_type	L;
    
    return cvt<S, I&0x1>(cvt<L, I>>1>(x));
}
template <class S> static inline vec<S>
cvt(vec<S> x)
{
    return x;
}

//! 2つのT型整数ベクトルをより小さなS型整数ベクトルに型変換する．
/*!
  Tは符号付き整数型，SはTの半分のサイズを持つ符号付き／符号なし整数型
  である．Sが符号付き／符号なしのいずれの場合も飽和処理が行われる．
  \param x	変換されるベクトル
  \param y	変換されるベクトル
  \return	xが変換されたものを下位，yが変換されたものを上位に
		配したベクトル
*/
template <class S, class T> static vec<S>	cvt(vec<T> x, vec<T> y)	;

// [1] 整数ベクトル間の変換
#if defined(SSE4)
#  if defined(AVX2)
#    define MM_CVTUP0(from, to)						\
      template <> inline vec<to>					\
      cvt<to, 0>(vec<from> x)						\
      {									\
	  return MM_MNEMONIC(cvt, _mm256_, MM_SUFFIX(from),		\
			     MM_SIGNED(to))(_mm256_castsi256_si128(x));	\
      }
#    define MM_CVTUP1(from, to)						\
      template <> inline vec<to>					\
      cvt<to, 1>(vec<from> x)						\
      {									\
	  return MM_MNEMONIC(cvt, _mm256_, MM_SUFFIX(from),		\
			     MM_SIGNED(to))(				\
				 _mm256_extractf128_si256(x, 0x1));	\
      }
#  else	// SSE4 && !AVX2
#    define MM_CVTUP0(from, to)						\
      template <> inline vec<to>					\
      cvt<to, 0>(vec<from> x)						\
      {									\
	  return MM_MNEMONIC(cvt, _mm_,					\
			     MM_SUFFIX(from), MM_SIGNED(to))(x);	\
      }
#    define MM_CVTUP1(from, to)						\
      template <> inline vec<to>					\
      cvt<to, 1>(vec<from> x)						\
      {									\
	  return cvt<to>(shift_r<vec<from>::size/2>(x));		\
      }
#  endif
  MM_CVTUP0(int8_t,    int16_t)		// s_char -> short
  MM_CVTUP1(int8_t,    int16_t)		// s_char -> short
  MM_CVTUP0(int8_t,    int32_t)		// s_char -> int
  MM_CVTUP0(int8_t,    int64_t)		// s_char -> long
  
  MM_CVTUP0(int16_t,   int32_t)		// short  -> int
  MM_CVTUP1(int16_t,   int32_t)		// short  -> int
  MM_CVTUP0(int16_t,   int64_t)		// short  -> long
  
  MM_CVTUP0(int32_t,   int64_t)		// int    -> long
  MM_CVTUP1(int32_t,   int64_t)		// int    -> long

  MM_CVTUP0(u_int8_t,  int16_t)		// u_char -> short
  MM_CVTUP1(u_int8_t,  int16_t)		// u_char -> short
  MM_CVTUP0(u_int8_t,  u_int16_t)	// u_char -> u_short
  MM_CVTUP1(u_int8_t,  u_int16_t)	// u_char -> u_short
  MM_CVTUP0(u_int8_t,  int32_t)		// u_char -> int
  MM_CVTUP0(u_int8_t,  u_int32_t)	// u_char -> u_int
  MM_CVTUP0(u_int8_t,  int64_t)		// u_char -> long
  MM_CVTUP0(u_int8_t,  u_int64_t)	// u_char -> u_long
  
  MM_CVTUP0(u_int16_t, int32_t)		// u_short -> int
  MM_CVTUP1(u_int16_t, int32_t)		// u_short -> int
  MM_CVTUP0(u_int16_t, u_int32_t)	// u_short -> u_int
  MM_CVTUP1(u_int16_t, u_int32_t)	// u_short -> u_int
  MM_CVTUP0(u_int16_t, int64_t)		// u_short -> long
  MM_CVTUP0(u_int16_t, u_int64_t)	// u_short -> u_long
  
  MM_CVTUP0(u_int32_t, int64_t)		// u_int -> long
  MM_CVTUP1(u_int32_t, int64_t)		// u_int -> long
  MM_CVTUP0(u_int32_t, u_int64_t)	// u_int -> u_long
  MM_CVTUP1(u_int32_t, u_int64_t)	// u_int -> u_long

#  undef MM_CVTUP0
#  undef MM_CVTUP1
#else	// !SSE4
#  define MM_CVTUP_I(from, to)						\
    template <> inline vec<to>						\
    cvt<to, 0>(vec<from> x)						\
    {									\
	return cast<to>(dup<0>(x)) >> 8*vec<from>::element_size;	\
    }									\
    template <> inline vec<to>						\
    cvt<to, 1>(vec<from> x)						\
    {									\
	return cast<to>(dup<1>(x)) >> 8*vec<from>::element_size;	\
    }
#  define MM_CVTUP_UI(from, to)						\
    template <> inline vec<to>						\
    cvt<to, 0>(vec<from> x)						\
    {									\
	return cast<to>(unpack_low(x, zero<from>()));			\
    }									\
    template <> inline vec<to>						\
    cvt<to, 1>(vec<from> x)						\
    {									\
	return cast<to>(unpack_high(x, zero<from>()));			\
    }

  MM_CVTUP_I(int8_t,     int16_t)	// s_char  -> short
  MM_CVTUP_I(int16_t,    int32_t)	// short   -> int
  // epi64の算術右シフトが未サポートなので int -> long は実装できない

  MM_CVTUP_UI(u_int8_t,  int16_t)	// u_char  -> short
  MM_CVTUP_UI(u_int8_t,  u_int16_t)	// u_char  -> u_short
  MM_CVTUP_UI(u_int16_t, int32_t)	// u_short -> int
  MM_CVTUP_UI(u_int16_t, u_int32_t)	// u_short -> u_int
  MM_CVTUP_UI(u_int32_t, int64_t)	// u_int   -> long
  MM_CVTUP_UI(u_int32_t, u_int64_t)	// u_int   -> u_long

#  undef MM_CVTUP_I
#  undef MM_CVTUP_UI
#endif

#if defined(AVX2)
#  define MM_CVTDOWN_I(from, to)					\
    template <> inline vec<to>						\
    cvt(vec<from> x, vec<from> y)					\
    {									\
	return MM_MNEMONIC(packs, _mm256_, , MM_SUFFIX(from))(		\
			       _mm256_permute2f128_si256(x, y, 0x20),	\
			       _mm256_permute2f128_si256(x, y, 0x31));	\
    }
#  define MM_CVTDOWN_UI(from, to)					\
    template <> inline vec<to>						\
    cvt(vec<from> x, vec<from> y)					\
    {									\
	return MM_MNEMONIC(packus, _mm256_, , MM_SUFFIX(from))(		\
			       _mm256_permute2f128_si256(x, y, 0x20),	\
			       _mm256_permute2f128_si256(x, y, 0x31));	\
    }
#else
#  define MM_CVTDOWN_I(from, to)					\
    MM_FUNC(vec<to> cvt<to>(vec<from> x, vec<from> y),			\
	    packs, (x, y), void, from, MM_SIGNED)
#  define MM_CVTDOWN_UI(from, to)					\
    MM_FUNC(vec<to> cvt<to>(vec<from> x, vec<from> y),			\
	    packus, (x, y), void, from, MM_SIGNED)
#endif

#define _mm_packus_pi16	_mm_packs_pu16	// 不適切な命名をSSE2に合わせて修正

MM_CVTDOWN_I(int16_t,  int8_t)		// short -> s_char
MM_CVTDOWN_I(int32_t,  int16_t)		// int   -> short
MM_CVTDOWN_UI(int16_t, u_int8_t)	// short -> u_char
#if defined(SSE4)
  MM_CVTDOWN_UI(int32_t, u_int16_t)	// int -> u_short
#endif

#undef MM_CVTDOWN_I
#undef MM_CVTDOWN_UI

// [2] 整数ベクトルと浮動小数点数ベクトル間の変換
#define MM_CVT(from, to)						\
  MM_FUNC(vec<to> cvt<to>(vec<from> x), cvt, (x), from, to, MM_SUFFIX)
#define MM_CVT_2(type0, type1)						\
  MM_CVT(type0, type1)							\
  MM_CVT(type1, type0)

#if defined(AVX)
#  if defined(AVX2)
    MM_CVT_2(int32_t, float)		// int   <-> float

    template <> inline F64vec		// int    -> double
    cvt<double, 0>(Is32vec x)
    {
	return _mm256_cvtepi32_pd(_mm256_castsi256_si128(x));
    }

    template <> inline F64vec		// int    -> double
    cvt<double, 1>(Is32vec x)
    {
	return _mm256_cvtepi32_pd(_mm256_extractf128_si256(x, 0x1));
    }

    template <> inline Is32vec		// double -> int
    cvt<int32_t>(F64vec x, F64vec y)
    {
	return _mm256_insertf128_si256(_mm256_castsi128_si256(
					   _mm256_cvtpd_epi32(x)),
				       _mm256_cvtpd_epi32(y), 0x1);
    }

#    define MM_CVTI_F(itype, ftype)					\
      template <> inline vec<ftype>					\
      cvt<ftype>(vec<itype> x)						\
      {									\
	  return cvt<ftype>(cvt<int32_t>(x));				\
      }
#  else		// AVX && !AVX2
    template <> inline F32vec		// 2*int  -> float
    cvt<float>(Is32vec x, Is32vec y)
    {
	return _mm256_cvtepi32_ps(
		   _mm256_insertf128_si256(_mm256_castsi128_si256(x), y, 0x1));
    }

    template <> inline Is32vec		// float  -> int
    cvt<int32_t, 0>(F32vec x)
    {
	return _mm256_castsi256_si128(_mm256_cvtps_epi32(x));
    }

    template <> inline Is32vec		// float  -> int
    cvt<int32_t, 1>(F32vec x)
    {
	return _mm256_extractf128_si256(_mm256_cvtps_epi32(x), 0x1);
    }

    MM_CVT(int32_t, double)		// int    -> double

    template <> inline Is32vec		// double -> int
    cvt<int32_t>(F64vec x)
    {
	return _mm256_cvtpd_epi32(x);
    }

    template <> inline Is16vec		// float  -> short
    cvt<int16_t>(F32vec x)
    {
	__m256i	y = _mm256_cvtps_epi32(x);
	return cvt<int16_t>(vec<int32_t>(_mm256_castsi256_si128(y)),
			    vec<int32_t>(_mm256_extractf128_si256(y, 0x1)));
    }

#    define MM_CVTI_F(itype, ftype)					\
      template <> inline vec<ftype>					\
      cvt<ftype>(vec<itype> x)						\
      {									\
	  return MM_MNEMONIC(cvt, _mm256_, epi32, MM_SUFFIX(ftype))(	\
		     _mm256_insertf128_si256(				\
			 _mm256_castsi128_si256(cvt<int32_t>(x)),	\
			 cvt<int32_t>(shift_r<4>(x)), 0x1));		\
      }
#  endif

  MM_CVTI_F(int8_t,    float)		// s_char  -> float
  MM_CVTI_F(int16_t,   float)		// short   -> float
  MM_CVTI_F(u_int8_t,  float)		// u_char  -> float
  MM_CVTI_F(u_int16_t, float)		// u_short -> float
#  undef MM_CVTI_F

#elif defined(SSE2)	// !AVX && SSE2
  MM_CVT_2(int32_t, float)		// int    <-> float

  MM_CVT(int32_t, double)		// int	   -> double

  template <> inline F64vec		// int	   -> double
  cvt<double, 1>(Is32vec x)
  {
      return _mm_cvtepi32_pd(_mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));
  }

  template <> inline Is32vec		// double  -> int
  cvt<int32_t>(F64vec x, F64vec y)
  {
      return _mm_unpacklo_epi64(_mm_cvtpd_epi32(x), _mm_cvtpd_epi32(y));
  }

#  define MM_CVTI_F(itype, suffix)					\
    template <> inline F32vec						\
    cvt<float>(vec<itype> x)						\
    {									\
	return MM_MNEMONIC(cvt, _mm_, suffix, ps)(_mm_movepi64_pi64(x));\
    }
#  define MM_CVTF_I(itype, suffix)					\
    template <> inline vec<itype>					\
    cvt<itype>(F32vec x)						\
    {									\
	return _mm_movpi64_epi64(MM_MNEMONIC(cvt, _mm_, ps, suffix)(x));\
    }
#  define MM_CVT_2FI(itype, suffix)					\
    MM_CVTI_F(itype, suffix)						\
    MM_CVTF_I(itype, suffix)

  MM_CVT_2FI(int8_t,   pi8)		// s_char <-> float
  MM_CVT_2FI(int16_t,  pi16)		// short  <-> float
  MM_CVTI_F(u_int8_t,  pu8)		// u_char  -> float
  MM_CVTI_F(u_int16_t, pu16)		// u_short -> float

#  undef MM_CVTI_F
#  undef MM_CVTF_I
#  undef MM_CVT_2FI

#elif defined(SSE)	// !SSE2 && SSE
  template <> inline F32vec
  cvt<float>(Is32vec x, Is32vec y)	// 2*int   -> float
  {
      return _mm_cvtpi32x2_ps(x, y);
  }

  MM_CVT(float, int32_t)		// float   -> int

  template <> inline Is32vec		// float   -> int
  cvt<int32_t, 1>(F32vec x)
  {
      return _mm_cvtps_pi32(_mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2)));
  }

  MM_CVT_2(int8_t,  float)		// s_char <-> float
  MM_CVT_2(int16_t, float)		// short  <-> float
  MM_CVT(u_int8_t,  float)		// u_char  -> float
  MM_CVT(u_int16_t, float)		// u_short -> float
#endif
  
// [3] 浮動小数点数ベクトル間の変換
#if defined(AVX)
  template <> F64vec
  cvt<double, 0>(F32vec x)		// float -> double
  {
      return _mm256_cvtps_pd(_mm256_castps256_ps128(x));
  }
  template <> F64vec
  cvt<double, 1>(F32vec x)		// float -> double
  {
      return _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
  }

  template <> F32vec			// double -> float
  cvt<float>(F64vec x, F64vec y)
  {
      return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(x)),
				  _mm256_cvtpd_ps(y), 1);
  }
#elif defined(SSE2)
  template <> F64vec
  cvt<double, 0>(F32vec x)		// float -> double
  {
      return _mm_cvtps_pd(x);
  }
  template <> F64vec
  cvt<double, 1>(F32vec x)		// float -> double
  {
      return _mm_cvtps_pd(_mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2)));
  }
	  
  template <> F32vec			// double -> float
  cvt<float>(F64vec x, F64vec y)
  {
      return _mm_shuffle_ps(_mm_cvtpd_ps(x), _mm_cvtpd_ps(y),
			    _MM_SHUFFLE(1, 0, 1, 0));
  }
#endif
  
#undef MM_CVT
#undef MM_CVT_2

/************************************************************************
*  Mask conversion operators						*
************************************************************************/
//! T型マスクベクトルのI番目の部分をS型マスクベクトルに型変換する．
/*!
  整数ベクトル間の変換の場合，SのサイズはTの2/4/8倍である．また，S, Tは
  符号付き／符号なしのいずれでも良い．
  \param x	変換されるマスクベクトル
  \return	変換されたマスクベクトル
*/
template <class S, size_t I=0, class T> static inline vec<S>
cvt_mask(vec<T> x)
{
    typedef typename type_traits<S>::lower_type	L;
    
    return cvt_mask<S, I&0x1>(cvt_mask<L, I>>1>(x));
}
template <class S> static inline vec<S>
cvt_mask(vec<S> x)
{
    return x;
}

//! 2つのT型整数マスクベクトルをより小さなS型整数マスクベクトルに型変換する．
/*!
  SのサイズはTの倍である．また，S, Tは符号付き／符号なしのいずれでも良い．
  \param x	変換されるマスクベクトル
  \param y	変換されるマスクベクトル
  \return	xが変換されたものを下位，yが変換されたものを上位に
		配したマスクベクトル
*/
template <class S, class T> static vec<S>
cvt_mask(vec<T> x, vec<T> y)						;

// [1] 整数ベクトル間のマスク変換
#if defined(AVX2)
#  define MM_CVTUP_MASK(from, to)					\
    template <> inline vec<to>						\
    cvt_mask<to, 0>(vec<from> x)					\
    {									\
	return MM_MNEMONIC(cvt, _mm256_,				\
			   MM_SIGNED(from), MM_SIGNED(to))(		\
		   _mm256_castsi256_si128(x));				\
    }									\
    template <> inline vec<to>						\
    cvt_mask<to, 1>(vec<from> x)					\
    {									\
	return MM_MNEMONIC(cvt, _mm256_,				\
			   MM_SIGNED(from), MM_SIGNED(to))(		\
		   _mm256_extractf128_si256(x, 0x1));			\
    }
#  define MM_CVTDOWN_MASK(from, to)					\
    template <> inline vec<to>						\
    cvt_mask(vec<from> x, vec<from> y)					\
    {									\
	return MM_MNEMONIC(packs, _mm256_, , MM_SIGNED(from))(		\
			       _mm256_permute2f128_si256(x, y, 0x20),	\
			       _mm256_permute2f128_si256(x, y, 0x31));	\
    }
#else
#  define MM_CVTUP_MASK(from, to)					\
    template <> inline vec<to>						\
    cvt_mask<to, 0>(vec<from> x)					\
    {									\
	return MM_MNEMONIC(unpacklo, _mm_, , MM_SIGNED(from))(x, x);	\
    }									\
    template <> inline vec<to>						\
    cvt_mask<to, 1>(vec<from> x)					\
    {									\
	return MM_MNEMONIC(unpackhi, _mm_, , MM_SIGNED(from))(x, x);	\
    }
#  define MM_CVTDOWN_MASK(from, to)					\
    MM_FUNC(vec<to> cvt_mask(vec<from> x, vec<from> y),			\
	    packs, (x, y), void, from, MM_SIGNED)
#endif
#define MM_CVT_MASK(type0, type1)					\
    MM_CVTUP_MASK(type0, type1)						\
    MM_CVTDOWN_MASK(type1, type0)

MM_CVT_MASK(u_int8_t,	 u_int16_t)	// u_char  <-> u_short
MM_CVT_MASK(u_int16_t,	 u_int32_t)	// u_short <-> u_int
MM_CVTUP_MASK(u_int32_t, u_int64_t)	// u_int    -> u_long

#undef MM_CVTUP_MASK
#undef MM_CVTDOWN_MASK
#undef MM_CVT_MASK
    
// [2] 整数ベクトルと浮動小数点数ベクトル間のマスク変換
#if defined(SSE2)
#  if !defined(AVX) || defined(AVX2)	// Is32vec::size == F32vec::size
    template <> inline vec<float>
    cvt_mask<float>(vec<u_int32_t> x)
    {
	return cast_base<fvec_t>(ivec_t(x));
    }
    template <> inline vec<u_int32_t>
    cvt_mask<u_int32_t>(vec<float> x)
    {
	return cast_base<ivec_t>(fvec_t(x));
    }
    template <> inline vec<double>
    cvt_mask<double>(vec<u_int64_t> x)
    {
	return cast_base<dvec_t>(ivec_t(x));
    }
    template <> inline vec<u_int64_t>
    cvt_mask<u_int64_t>(vec<double> x)
    {
	return cast_base<ivec_t>(dvec_t(x));
    }
#    undef MM_CVT_MASK_2FI
#  else	// AVX && !AVX2
#    define MM_CVT_MASK_IF(itype, ftype)				\
    template <> inline vec<ftype>					\
      cvt_mask<ftype>(vec<itype> x)					\
      {									\
	  typedef type_traits<itype>::upper_type	upper_type;	\
      									\
	  return MM_MNEMONIC(cast, _mm256_, si256, MM_SUFFIX(ftype))(	\
		     _mm256_insertf128_si256(				\
			 _mm256_castsi128_si256(			\
			     cvt_mask<upper_type, 0>(x)),		\
			 cvt_mask<upper_type, 1>(x), 0x1));		\
      }
#    define MM_CVT_MASK_FI(itype)					\
      template <> inline vec<itype>					\
      cvt_mask<itype>(F32vec x)						\
      {									\
	  typedef type_traits<itype>::upper_type	upper_type;	\
	  								\
	  return cvt_mask<itype>(					\
		     vec<upper_type>(					\
			 _mm256_castsi256_si128(			\
			     _mm256_castps_si256(x))),			\
		     vec<upper_type>(					\
		         _mm256_extractf128_si256(			\
			     _mm256_castps_si256(x), 0x1)));		\
      }

    MM_CVT_MASK_IF(u_int16_t, float)	// u_short -> float
    MM_CVT_MASK_FI(u_int16_t)		// float   -> u_short
    MM_CVT_MASK_IF(u_int32_t, double)	// u_int   -> double

#    undef MM_CVT_MASK_IF
#    undef MM_CVT_MASK_FI
#  endif
#endif

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class T> static vec<T>	operator &(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator |(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator ^(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	andnot(vec<T> x, vec<T> y)	;

template <class T> static inline vec<T>
operator &(typename boost::disable_if<
	       boost::is_same<typename vec<T>::mask_type, vec<T> >,
	       typename vec<T>::mask_type>::type x,
	   vec<T> y)
{
    return cast<T>(x) & y;
}
    
template <class T> static inline vec<T>
operator |(typename boost::disable_if<
	       boost::is_same<typename vec<T>::mask_type, vec<T> >,
	       typename vec<T>::mask_type>::type x,
	   vec<T> y)
{
    return cast<T>(x) | y;
}
    
template <class T> static inline vec<T>
operator ^(typename boost::disable_if<
	       boost::is_same<typename vec<T>::mask_type, vec<T> >,
	       typename vec<T>::mask_type>::type x,
	   vec<T> y)
{
    return cast<T>(x) ^ y;
}
    
template <class T> static inline vec<T>
andnot(typename boost::disable_if<
	  boost::is_same<typename vec<T>::mask_type, vec<T> >,
	  typename vec<T>::mask_type>::type x,
       vec<T> y)
{
    return andnot(cast<T>(x), y);
}

template <class T> static inline vec<T>
operator &(vec<T> x,
	   typename boost::disable_if<
	       boost::is_same<typename vec<T>::mask_type, vec<T> >,
	       typename vec<T>::mask_type>::type y)
{
    return x & cast<T>(y);
}
    
template <class T> static inline vec<T>
operator |(vec<T> x,
	   typename boost::disable_if<
	       boost::is_same<typename vec<T>::mask_type, vec<T> >,
	       typename vec<T>::mask_type>::type y)
{
    return x | cast<T>(y);
}
    
template <class T> static inline vec<T>
operator ^(vec<T> x,
	   typename boost::disable_if<
	       boost::is_same<typename vec<T>::mask_type, vec<T> >,
	       typename vec<T>::mask_type>::type y)
{
    return x ^ cast<T>(y);
}

#define MM_LOGICALS(type)						\
    MM_BASE_FUNC_2(operator &, and,    type)				\
    MM_BASE_FUNC_2(operator |, or,     type)				\
    MM_BASE_FUNC_2(operator ^, xor,    type)				\
    MM_BASE_FUNC_2(andnot,     andnot, type)

MM_LOGICALS(int8_t)
MM_LOGICALS(int16_t)
MM_LOGICALS(int32_t)
MM_LOGICALS(int64_t)
MM_LOGICALS(u_int8_t)
MM_LOGICALS(u_int16_t)
MM_LOGICALS(u_int32_t)
MM_LOGICALS(u_int64_t)

#if defined(SSE)
  MM_LOGICALS(float)
#endif
#if defined(SSE2)
  MM_LOGICALS(double)
#endif

#undef MM_LOGICALS
    
/************************************************************************
*  Lookup								*
************************************************************************/
#if defined(AVX2)
#  define MM_LOOKUP(type)						\
    template <class S> static inline vec<type>				\
    lookup(const S* p, vec<type> idx)					\
    {									\
	typedef type_traits<type_traits<type>::upper_type>		\
		    ::signed_type		signed_upper_type;	\
	return cvt<type>(lookup(p, cvt<signed_upper_type, 0>(idx)),	\
			 lookup(p, cvt<signed_upper_type, 1>(idx)));	\
    }

  template <class S> static inline Is32vec
  lookup(const S* p, Is32vec idx)
  {
      const size_t	n = sizeof(int32_t) - sizeof(S);
      const void*	q = (const int8_t*)p - n;
      return (boost::is_signed<S>::value ?
	      _mm256_srai_epi32(_mm256_i32gather_epi32((const int32_t*)q,
						       idx, sizeof(S)), 8*n) :
	      _mm256_srli_epi32(_mm256_i32gather_epi32((const int32_t*)q,
						       idx, sizeof(S)), 8*n));
  }

  MM_LOOKUP(int16_t)
  MM_LOOKUP(u_int16_t)
  MM_LOOKUP(int8_t)
  MM_LOOKUP(u_int8_t)
#  undef MM_LOOKUP

  static inline F32vec
  lookup(const float* p, Is32vec idx)
  {
      return _mm256_i32gather_ps(p, idx, sizeof(float));
  }

  static inline F64vec
  lookup(const double* p, Is32vec idx)
  {
      return _mm256_i32gather_pd(p, _mm256_extractf128_si256(idx, 0x0),
				 sizeof(double));
  }
#else	// !AVX2
#  define MM_LOOKUP4(type)						\
    template <class S> static inline vec<type>				\
    lookup(const S* p, vec<type> idx)					\
    {									\
	return vec<type>(p[extract<3>(idx)], p[extract<2>(idx)],	\
			 p[extract<1>(idx)], p[extract<0>(idx)]);	\
    }
#  if defined(SSE2)
#    define MM_LOOKUP8(type)						\
    template <class S> static inline vec<type>				\
    lookup(const S* p, vec<type> idx)					\
    {									\
	return vec<type>(p[extract<7>(idx)], p[extract<6>(idx)],	\
			 p[extract<5>(idx)], p[extract<4>(idx)],	\
			 p[extract<3>(idx)], p[extract<2>(idx)],	\
			 p[extract<1>(idx)], p[extract<0>(idx)]);	\
    }
#  else		// !SSE2
#    define MM_LOOKUP8(type)						\
    template <class S> static inline vec<type>				\
    lookup(const S* p, vec<type> idx)					\
    {									\
	const Is16vec	idx_lo = cvt<int16_t, 0>(idx),			\
			idx_hi = cvt<int16_t, 1>(idx);			\
	return vec<type>(p[extract<3>(idx_hi)], p[extract<2>(idx_hi)],	\
			 p[extract<1>(idx_hi)], p[extract<0>(idx_hi)],	\
			 p[extract<3>(idx_lo)], p[extract<2>(idx_lo)],	\
			 p[extract<1>(idx_lo)], p[extract<0>(idx_lo)]);	\
    }
#  endif
#  if defined(SSE4)
#    define MM_LOOKUP16(type)						\
    template <class S> static inline vec<type>				\
    lookup(const S* p, vec<type> idx)					\
    {									\
	return vec<type>(p[extract<15>(idx)], p[extract<14>(idx)],	\
			 p[extract<13>(idx)], p[extract<12>(idx)],	\
			 p[extract<11>(idx)], p[extract<10>(idx)],	\
			 p[extract< 9>(idx)], p[extract< 8>(idx)],	\
			 p[extract< 7>(idx)], p[extract< 6>(idx)],	\
			 p[extract< 5>(idx)], p[extract< 4>(idx)],	\
			 p[extract< 3>(idx)], p[extract< 2>(idx)],	\
			 p[extract< 1>(idx)], p[extract< 0>(idx)]);	\
    }
#  else		// !SSE4
#    define MM_LOOKUP16(type)						\
    template <class S> static inline vec<type>				\
    lookup(const S* p, vec<type> idx)					\
    {									\
	const Is16vec	idx_lo = cvt<int16_t, 0>(idx),			\
			idx_hi = cvt<int16_t, 1>(idx);			\
	return vec<type>(p[extract<7>(idx_hi)], p[extract<6>(idx_hi)],	\
			 p[extract<5>(idx_hi)], p[extract<4>(idx_hi)],	\
			 p[extract<3>(idx_hi)], p[extract<2>(idx_hi)],	\
			 p[extract<1>(idx_hi)], p[extract<0>(idx_hi)],	\
			 p[extract<7>(idx_lo)], p[extract<6>(idx_lo)],	\
			 p[extract<5>(idx_lo)], p[extract<4>(idx_lo)],	\
			 p[extract<3>(idx_lo)], p[extract<2>(idx_lo)],	\
			 p[extract<1>(idx_lo)], p[extract<0>(idx_lo)]);	\
    }
#  endif

#  if defined(SSE2)
  MM_LOOKUP16(int8_t)
  MM_LOOKUP16(u_int8_t)
  MM_LOOKUP8(int16_t)
  MM_LOOKUP8(u_int16_t)
#    if defined(SSE4)
    MM_LOOKUP4(u_int32_t)
    MM_LOOKUP4(int32_t)
#    endif
#  else		// !SSE2
  MM_LOOKUP8(int8_t)
  MM_LOOKUP8(u_int8_t)
  MM_LOOKUP4(int16_t)
  MM_LOOKUP4(u_int16_t)
#  endif
#  undef MM_LOOKUP4
#  undef MM_LOOKUP8
#  undef MM_LOOKUP16
#endif
    
/************************************************************************
*  Selection								*
************************************************************************/
//! 2つのベクトル中の成分のいずれかをマスク値に応じて選択する．
/*!
 \param mask	マスク
 \param x	ベクトル
 \param y	ベクトル
 \return	maskにおいて1が立っている成分はxから，そうでない成分は
		yからそれぞれ選択して生成されたベクトル
*/
template <class T> static inline vec<T>
select(typename vec<T>::mask_type mask, vec<T> x, vec<T> y)
{
    return (mask & x) | andnot(mask, y);
}

#if defined(SSE4)
#  define MM_SELECT(type)						\
    template <> inline vec<type>					\
    select(vec<type>::mask_type mask, vec<type> x, vec<type> y)		\
    {									\
	   return MM_MNEMONIC(blendv, MM_PREFIX(type), ,		\
			      MM_SIGNED(int8_t))(y, x, mask);		\
    }
#  define MM_SELECT_F(type)						\
    MM_FUNC(vec<type> select(vec<type>::mask_type mask,			\
			     vec<type> x, vec<type> y),			\
	    blendv, (y, x, mask), void, type, MM_BASE)

  MM_SELECT(int8_t)
  MM_SELECT(int16_t)
  MM_SELECT(int32_t)
  MM_SELECT(int64_t)
  MM_SELECT(u_int8_t)
  MM_SELECT(u_int16_t)
  MM_SELECT(u_int32_t)
  MM_SELECT(u_int64_t)

  MM_SELECT_F(float)
  MM_SELECT_F(double)
  
#  undef MM_SELECT
#  undef MM_SELECT_F
#endif

/************************************************************************
*  Compare operators							*
************************************************************************/
template <class T> static typename vec<T>::mask_type
operator ==(vec<T> x, vec<T> y)						;
template <class T> static typename vec<T>::mask_type
operator > (vec<T> x, vec<T> y)						;
template <class T> static typename vec<T>::mask_type
operator < (vec<T> x, vec<T> y)						;
template <class T> static typename vec<T>::mask_type
operator !=(vec<T> x, vec<T> y)						;
template <class T> static typename vec<T>::mask_type
operator >=(vec<T> x, vec<T> y)						;
template <class T> static typename vec<T>::mask_type
operator <=(vec<T> x, vec<T> y)						;

// MMX, SSE, AVX2 には整数に対する cmplt ("less than") がない！
#define MM_COMPARE(func, op, type)					\
    MM_FUNC(vec<type>::mask_type func(vec<type> x, vec<type> y),	\
	    op, (x, y), void, type, MM_SIGNED)
#define MM_COMPARE_R(func, op, type)					\
    MM_FUNC(vec<type>::mask_type func(vec<type> x, vec<type> y),	\
	    op, (y, x), void, type, MM_SIGNED)
#define MM_COMPARES(type)						\
    MM_COMPARE(  operator ==, cmpeq, type)				\
    MM_COMPARE(  operator >,  cmpgt, type)				\
    MM_COMPARE_R(operator <,  cmpgt, type)

// 符号なし数に対しては等値性チェックしかできない！
MM_COMPARE(operator ==, cmpeq, u_int8_t)
MM_COMPARE(operator ==, cmpeq, u_int16_t)
MM_COMPARE(operator ==, cmpeq, u_int32_t)
#if defined(SSE4)
  MM_COMPARE(operator ==, cmpeq, u_int64_t)
#endif

MM_COMPARES(int8_t)
MM_COMPARES(int16_t)
MM_COMPARES(int32_t)

#if defined(AVX)	// AVX の浮動小数点数比較演算子はパラメータ形式
#  define MM_COMPARE_F(func, type, opcode)				\
    MM_FUNC(vec<type>::mask_type func(vec<type> x, vec<type> y),	\
	    cmp, (x, y, opcode), void, type, MM_SUFFIX)
#  define MM_COMPARES_F(type)						\
    MM_COMPARE_F(operator ==, type, _CMP_EQ_OQ)				\
    MM_COMPARE_F(operator >,  type, _CMP_GT_OS)				\
    MM_COMPARE_F(operator <,  type, _CMP_LT_OS)				\
    MM_COMPARE_F(operator !=, type, _CMP_NEQ_OQ)			\
    MM_COMPARE_F(operator >=, type, _CMP_GE_OS)				\
    MM_COMPARE_F(operator <=, type, _CMP_LE_OS)

  MM_COMPARES_F(float)
  MM_COMPARES_F(double)

#  undef MM_COMPARE_F
#  undef MM_COMPARES_F
#elif defined(SSE)
#  define MM_COMPARES_SUP(type)						\
    MM_COMPARE(  operator !=, cmpneq, type)				\
    MM_COMPARE(  operator >=, cmpge,  type)				\
    MM_COMPARE_R(operator <=, cmpge,  type)

  MM_COMPARES(float)
  MM_COMPARES_SUP(float)
#  if defined(SSE2)
    MM_COMPARES(double)
    MM_COMPARES_SUP(double)
#  endif

#  undef MM_COMPARES_SUP
#endif

#undef MM_COMPARE
#undef MM_COMPARES

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class T> static vec<T>	operator +(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator -(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator *(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator /(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator %(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	operator -(vec<T> x)		;
template <class T> static vec<T>	sat_add(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	sat_sub(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	mulhi(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	min(vec<T> x, vec<T> y)		;
template <class T> static vec<T>	max(vec<T> x, vec<T> y)		;
template <class T> static vec<T>	rcp(vec<T> x)			;
template <class T> static vec<T>	sqrt(vec<T> x)			;
template <class T> static vec<T>	rsqrt(vec<T> x)			;

template <class T> inline vec<T>
min(vec<T> x, vec<T> y)
{
    return select(x < y, x, y);
}

template <class T> inline vec<T>
max(vec<T> x, vec<T> y)
{
    return select(x > y, x, y);
}

template <class T> inline vec<T>
operator -(vec<T> x)
{
    return zero<T>() - x;
}

#define MM_ADD_SUB(type)						\
    MM_NUMERIC_FUNC_2(operator +, add, type)				\
    MM_NUMERIC_FUNC_2(operator -, sub, type)

// 符号なし数は，飽和演算によって operator [+|-] を定義する．
#define MM_ADD_SUB_U(type)						\
    MM_NUMERIC_FUNC_2(operator +, adds, type)				\
    MM_NUMERIC_FUNC_2(operator -, subs, type)

// 符号あり数は，飽和演算に sat_[add|sub] という名前を与える．
#define MM_SAT_ADD_SUB(type)						\
    MM_NUMERIC_FUNC_2(sat_add, adds, type)				\
    MM_NUMERIC_FUNC_2(sat_sub, subs, type)

#define MM_MIN_MAX(type)						\
    MM_NUMERIC_FUNC_2(min, min, type)					\
    MM_NUMERIC_FUNC_2(max, max, type)

// 加減算
MM_ADD_SUB(int8_t)
MM_ADD_SUB(int16_t)
MM_ADD_SUB(int32_t)
MM_ADD_SUB(int64_t)
MM_ADD_SUB_U(u_int8_t)
MM_ADD_SUB_U(u_int16_t)
MM_SAT_ADD_SUB(int8_t)
MM_SAT_ADD_SUB(int16_t)
MM_SAT_ADD_SUB(u_int8_t)
MM_SAT_ADD_SUB(u_int16_t)

// 乗算
MM_NUMERIC_FUNC_2(operator *, mullo, int16_t)
MM_NUMERIC_FUNC_2(mulhi,      mulhi, int16_t)

#if defined(SSE)
  // 加減算
  MM_ADD_SUB(float)

  // 乗除算
  MM_NUMERIC_FUNC_2(operator *, mul, float)
  MM_NUMERIC_FUNC_2(operator /, div, float)

  // Min/Max
  MM_MIN_MAX(u_int8_t)
  MM_MIN_MAX(int16_t)
  MM_MIN_MAX(float)

  // その他
  MM_NUMERIC_FUNC_1(sqrt,  sqrt,  float)
  MM_NUMERIC_FUNC_1(rsqrt, rsqrt, float)
  MM_NUMERIC_FUNC_1(rcp,   rcp,   float)
#endif

#if defined(SSE2)
  // 加減算
  MM_ADD_SUB(double)

  // 乗除算
  MM_NUMERIC_FUNC_2(operator *, mul, u_int32_t)
  MM_NUMERIC_FUNC_2(operator *, mul, double)
  MM_NUMERIC_FUNC_2(operator /, div, double)

  // Min/Max
  MM_MIN_MAX(double)

  // その他
  MM_NUMERIC_FUNC_1(sqrt, sqrt, double)
#endif

#if defined(SSE4)
  // 乗算
  MM_NUMERIC_FUNC_2(operator *, mullo, int32_t)

  // Min/Max
  MM_MIN_MAX(int8_t)
  MM_MIN_MAX(int32_t)
  MM_MIN_MAX(u_int16_t)
  MM_MIN_MAX(u_int32_t)
#endif

#undef MM_ADD_SUB
#undef MM_ADD_SUB_U
#undef MM_SAT_ADD_SUB
#undef MM_MIN_MAX

template <class T> static inline vec<T>
operator *(T c, vec<T> x)
{
    return vec<T>(c) * x;
}

template <class T> static inline vec<T>
operator *(vec<T> x, T c)
{
    return x * vec<T>(c);
}

template <class T> static inline vec<T>
operator /(vec<T> x, T c)
{
    return x / vec<T>(c);
}

/************************************************************************
*  "[Greater|Less] than or equal to" operators				*
************************************************************************/
template <class T> static inline typename vec<T>::mask_type
operator >=(vec<T> x, vec<T> y)
{
    return max(x, y) == x;
}

template <class T> static inline typename vec<T>::mask_type
operator <=(vec<T> x, vec<T> y)
{
    return y >= x;
}

/************************************************************************
*  Average values							*
************************************************************************/
template <class T> static inline vec<T>
avg(vec<T> x, vec<T> y)			{return (x + y) >> 1;}
template <class T> static inline vec<T>
sub_avg(vec<T> x, vec<T> y)		{return (x - y) >> 1;}

#if defined(SSE)
  MM_NUMERIC_FUNC_2(avg, avg, u_int8_t)
  MM_NUMERIC_FUNC_2(avg, avg, u_int16_t)
  template <> inline F32vec
  avg(F32vec x, F32vec y)		{return (x + y) * F32vec(0.5f);}
  template <> inline F32vec
  sub_avg(F32vec x, F32vec y)		{return (x - y) * F32vec(0.5f);}
#endif

#if defined(SSE2)
  template <> inline F64vec
  avg(F64vec x, F64vec y)		{return (x + y) * F64vec(0.5d);}
  template <> inline F64vec
  sub_avg(F64vec x, F64vec y)		{return (x - y) * F64vec(0.5d);}
#endif
  
/************************************************************************
*  Absolute values							*
************************************************************************/
template <class T> static inline vec<T>	abs(vec<T> x)	{return max(x, -x);}
template <> inline Iu8vec		abs(Iu8vec x)	{return x;}
template <> inline Iu16vec		abs(Iu16vec x)	{return x;}
template <> inline Iu32vec		abs(Iu32vec x)	{return x;}

#if defined(SSSE3)
  MM_NUMERIC_FUNC_1(abs, abs, int8_t)
  MM_NUMERIC_FUNC_1(abs, abs, int16_t)
  MM_NUMERIC_FUNC_1(abs, abs, int32_t)
#endif
  
/************************************************************************
*  Absolute differences							*
************************************************************************/
template <class T> inline vec<T>
diff(vec<T> x, vec<T> y)	{return select(x > y, x - y, y - x);}
template <> inline Is8vec
diff(Is8vec x, Is8vec y)	{return sat_sub(x, y) | sat_sub(y, x);}
template <> inline Iu8vec
diff(Iu8vec x, Iu8vec y)	{return sat_sub(x, y) | sat_sub(y, x);}
template <> inline Is16vec
diff(Is16vec x, Is16vec y)	{return sat_sub(x, y) | sat_sub(y, x);}
template <> inline Iu16vec
diff(Iu16vec x, Iu16vec y)	{return sat_sub(x, y) | sat_sub(y, x);}
  
/************************************************************************
*  Horizontal sum of vector elements					*
************************************************************************/
template <size_t I, size_t N, class T> static inline vec<T>
hsum(vec<T> x)
{
    if (I < N)
	return hsum<I << 1, N>(x + shift_r<I>(x));
    else
	return x;
}
    
template <class T> static inline T
hsum(vec<T> x)
{
    return extract<0>(hsum<1, vec<T>::size>(x));
}

/************************************************************************
*  Inner product							*
************************************************************************/
template <class T> static inline T
inner_product(vec<T> x, vec<T> y)
{
    return hsum(x * y);
}

/************************************************************************
*  SVML(Short Vector Math Library) functions				*
************************************************************************/
template <class T> static vec<T>	erf(vec<T> x)			;
template <class T> static vec<T>	erfc(vec<T> x)			;

template <class T> static vec<T>	floor(vec<T> x)			;
template <class T> static vec<T>	ceil(vec<T> x)			;

template <class T> static vec<T>	exp(vec<T> x)			;
template <class T> static vec<T>	cexp(vec<T> x)			;
template <class T> static vec<T>	exp2(vec<T> x)			;
template <class T> static vec<T>	pow(vec<T> x, vec<T> y)		;

template <class T> static vec<T>	log(vec<T> x)			;
template <class T> static vec<T>	log2(vec<T> x)			;
template <class T> static vec<T>	log10(vec<T> x)			;
template <class T> static vec<T>	clog(vec<T> x)			;

template <class T> static vec<T>	invsqrt(vec<T> x)		;
template <class T> static vec<T>	cbrt(vec<T> x)			;
template <class T> static vec<T>	invcbrt(vec<T> x)		;
template <class T> static vec<T>	csqrt(vec<T> x)			;

template <class T> static vec<T>	cos(vec<T> x)			;
template <class T> static vec<T>	sin(vec<T> x)			;
template <class T> static vec<T>	tan(vec<T> x)			;
template <class T> static vec<T>	sincos(typename
					       vec<T>::base_type* pcos,
					       vec<T> x)		;
template <class T> static vec<T>	acos(vec<T> x)			;
template <class T> static vec<T>	asin(vec<T> x)			;
template <class T> static vec<T>	atan(vec<T> x)			;
template <class T> static vec<T>	atan2(vec<T> x, vec<T> y)	;
template <class T> static vec<T>	cosh(vec<T> x)			;
template <class T> static vec<T>	sinh(vec<T> x)			;
template <class T> static vec<T>	tanh(vec<T> x)			;
template <class T> static vec<T>	acosh(vec<T> x)			;
template <class T> static vec<T>	asinh(vec<T> x)			;
template <class T> static vec<T>	atanh(vec<T> x)			;

#if defined(SSE)
  MM_NUMERIC_FUNC_1(erf,     erf,	 float)
  MM_NUMERIC_FUNC_1(erfc,    erfc,	 float)

#  if defined(SSE4)
  MM_NUMERIC_FUNC_1(floor,   floor,	 float)
  MM_NUMERIC_FUNC_1(ceil,    ceil,	 float)
#  else
  MM_NUMERIC_FUNC_1(floor,   svml_floor, float)
  MM_NUMERIC_FUNC_1(ceil,    svml_ceil,	 float)
#  endif
  
  MM_NUMERIC_FUNC_1(exp,     exp,	 float)
  MM_NUMERIC_FUNC_1(cexp,    cexp,	 float)
  MM_NUMERIC_FUNC_1(exp2,    exp2,	 float)
  MM_NUMERIC_FUNC_2(pow,     pow,	 float)

  MM_NUMERIC_FUNC_1(log,     log,	 float)
  MM_NUMERIC_FUNC_1(log2,    log2,	 float)
  MM_NUMERIC_FUNC_1(log10,   log10,	 float)
  MM_NUMERIC_FUNC_1(clog,    clog,	 float)

  MM_NUMERIC_FUNC_1(invsqrt, invsqrt,	 float)
  MM_NUMERIC_FUNC_1(cbrt,    cbrt,	 float)
  MM_NUMERIC_FUNC_1(invcbrt, invcbrt,	 float)
  MM_NUMERIC_FUNC_1(csqrt,   csqrt,	 float)

  MM_NUMERIC_FUNC_1(cos,     cos,	 float)
  MM_NUMERIC_FUNC_1(sin,     sin,	 float)
  MM_NUMERIC_FUNC_1(tan,     tan,	 float)
  MM_FUNC(F32vec sincos(fvec_t* pcos, F32vec x),
	  sincos, (pcos, x), void, float, MM_SUFFIX)
  MM_NUMERIC_FUNC_1(acos,    acos,	 float)
  MM_NUMERIC_FUNC_1(asin,    asin,	 float)
  MM_NUMERIC_FUNC_1(atan,    atan,	 float)
  MM_NUMERIC_FUNC_2(atan2,   atan2,	 float)
  MM_NUMERIC_FUNC_1(cosh,    cosh,	 float)
  MM_NUMERIC_FUNC_1(sinh,    sinh,	 float)
  MM_NUMERIC_FUNC_1(tanh,    tanh,	 float)
  MM_NUMERIC_FUNC_1(acosh,   acosh,	 float)
  MM_NUMERIC_FUNC_1(asinh,   asinh,	 float)
  MM_NUMERIC_FUNC_1(atanh,   atanh,	 float)
#endif

#if defined(SSE2)
#  if defined(_mm_idiv_epi32)
  // 整数除算
  MM_NUMERIC_FUNC_2(operator /, div, int8_t)
  MM_NUMERIC_FUNC_2(operator /, div, int16_t)
  MM_NUMERIC_FUNC_2(operator /, div, u_int8_t)
  MM_NUMERIC_FUNC_2(operator /, div, u_int16_t)

  // 剰余
  MM_NUMERIC_FUNC_2(operator %, rem, int8_t)
  MM_NUMERIC_FUNC_2(operator %, rem, int16_t)
  MM_NUMERIC_FUNC_2(operator %, rem, u_int8_t)
  MM_NUMERIC_FUNC_2(operator %, rem, u_int16_t)
#  endif
  
  // 整数除算
  MM_FUNC_2(operator /, idiv, int32_t)
  MM_FUNC_2(operator /, udiv, u_int32_t)

  // 剰余
  MM_FUNC_2(operator %, irem, int32_t)
  MM_FUNC_2(operator %, urem, u_int32_t)

  // 除算と剰余
  template <class T> static vec<T>	divrem(vec<T>& r,
					       vec<T> x, vec<T> y)	;

  MM_FUNC(Is32vec divrem(Is32vec& r, Is32vec x, Is32vec y),
	  idivrem, ((ivec_t*)&r, x, y), void, int32_t, MM_SIGNED)
  MM_FUNC(Iu32vec divrem(Iu32vec& r, Iu32vec x, Iu32vec y),
	  udivrem, ((ivec_t*)&r, x, y), void, u_int32_t, MM_SIGNED)

  MM_NUMERIC_FUNC_1(erf,     erf,        double)
  MM_NUMERIC_FUNC_1(erfc,    erfc,       double)

#  if defined(SSE4)
  MM_NUMERIC_FUNC_1(floor,   floor,	 double)
  MM_NUMERIC_FUNC_1(ceil,    ceil,	 double)
#  else
  MM_NUMERIC_FUNC_1(floor,   svml_floor, double)
  MM_NUMERIC_FUNC_1(ceil,    svml_ceil,	 double)
#  endif
  MM_NUMERIC_FUNC_1(exp,     exp,        double)
  MM_NUMERIC_FUNC_1(exp2,    exp2,       double)
  MM_NUMERIC_FUNC_2(pow,     pow,        double)

  MM_NUMERIC_FUNC_1(log,     log,        double)
  MM_NUMERIC_FUNC_1(log2,    log2,       double)
  MM_NUMERIC_FUNC_1(log10,   log10,      double)

  MM_NUMERIC_FUNC_1(invsqrt, invsqrt,    double)
  MM_NUMERIC_FUNC_1(cbrt,    cbrt,       double)
  MM_NUMERIC_FUNC_1(invcbrt, invcbrt,    double)

  MM_NUMERIC_FUNC_1(cos,     cos,        double)
  MM_NUMERIC_FUNC_1(sin,     sin,        double)
  MM_NUMERIC_FUNC_1(tan,     tan,        double)
  MM_FUNC(F64vec sincos(dvec_t* pcos, F64vec x),
	  sincos, (pcos, x), void, double, MM_SUFFIX)
  MM_NUMERIC_FUNC_1(acos,    acos,       double)
  MM_NUMERIC_FUNC_1(asin,    asin,       double)
  MM_NUMERIC_FUNC_1(atan,    atan,       double)
  MM_NUMERIC_FUNC_2(atan2,   atan2,      double)
  MM_NUMERIC_FUNC_1(cosh,    cosh,       double)
  MM_NUMERIC_FUNC_1(sinh,    sinh,       double)
  MM_NUMERIC_FUNC_1(tanh,    tanh,       double)
  MM_NUMERIC_FUNC_1(acosh,   acosh,      double)
  MM_NUMERIC_FUNC_1(asinh,   asinh,      double)
  MM_NUMERIC_FUNC_1(atanh,   atanh,      double)
#endif

/************************************************************************
*  Control functions							*
************************************************************************/
inline void	empty()			{_mm_empty();}
    
/************************************************************************
*  Undefine macros							*
************************************************************************/
#undef MM_PREFIX
#undef MM_SUFFIX
#undef MM_BASE

#undef MM_PREFIX_int8_t
#undef MM_PREFIX_int16_t
#undef MM_PREFIX_int32_t
#undef MM_PREFIX_int64_t
#undef MM_PREFIX_u_int8_t
#undef MM_PREFIX_u_int16_t
#undef MM_PREFIX_u_int32_t
#undef MM_PREFIX_u_int64_t
#undef MM_PREFIX_ivec_t

#undef MM_SUFFIX_int8_t
#undef MM_SUFFIX_int16_t
#undef MM_SUFFIX_int32_t
#undef MM_SUFFIX_int64_t
#undef MM_SUFFIX_u_int8_t
#undef MM_SUFFIX_u_int16_t
#undef MM_SUFFIX_u_int32_t
#undef MM_SUFFIX_u_int64_t
#undef MM_SUFFIX_ivec_t
#undef MM_SUFFIX_void

#undef MM_SIGNED_int8_t
#undef MM_SIGNED_int16_t
#undef MM_SIGNED_int32_t
#undef MM_SIGNED_int64_t
#undef MM_SIGNED_u_int8_t
#undef MM_SIGNED_u_int16_t
#undef MM_SIGNED_u_int32_t
#undef MM_SIGNED_u_int64_t

#undef MM_BASE_int8_t
#undef MM_BASE_int16_t
#undef MM_BASE_int32_t
#undef MM_BASE_int64_t
#undef MM_BASE_u_int8_t
#undef MM_BASE_u_int16_t
#undef MM_BASE_u_int32_t
#undef MM_BASE_u_int64_t
#undef MM_BASE_u_ivec_t

#if defined(SSE)
#  undef MM_PREFIX_float
#  undef MM_PREFIX_fvec_t
#  undef MM_SUFFIX_float
#  undef MM_SUFFIX_fvec_t
#  undef MM_SIGNED_float
#  undef MM_BASE_float
#  undef MM_BASE_fvec_t
#endif

#if defined(SSE2)
#  undef MM_PREFIX_double
#  undef MM_PREFIX_dvec_t
#  undef MM_SUFFIX_double
#  undef MM_SUFFIX_dvec_t
#  undef MM_SIGNED_double
#  undef MM_BASE_double
#  undef MM_BASE_dvec_t
#endif

#undef MM_CAT
#undef MM_MNEMONIC
#undef MM_TMPL_FUNC
#undef MM_FUNC
#undef MM_FUNC_2
#undef MM_NUMERIC_FUNC_1
#undef MM_NUMERIC_FUNC_2

/************************************************************************
*  functions for supporting memory alignment				*
************************************************************************/
//! 指定されたアドレスがアライメントされているか調べる
/*!
  \param p	調べたいアドレス
  \return	アライメントされていれば true, そうでなければ flase
*/
template <class T> inline bool
is_aligned(T* p)
{
    return (reinterpret_cast<size_t>(p) % sizeof(vec<T>) == 0);
}
    
//! 指定されたアドレスの直後のアライメントされているアドレスを返す
/*!
  \param p	アドレス
  \return	pの直後(pを含む)のアライメントされているアドレス
*/
template <class T> inline T*
begin(T* p)
{
    const size_t	nelms = reinterpret_cast<size_t>(p) / sizeof(T);
    const size_t	vsize = vec<T>::size;
    
    return p + (vsize - 1 - (nelms - 1) % vsize);
}
    
//! 指定されたアドレスの直後のアライメントされているアドレスを返す
/*!
  \param p	アドレス
  \return	pの直後(pを含む)のアライメントされているアドレス
*/
template <class T> inline const T*
cbegin(const T* p)
{
    const size_t	nelms = reinterpret_cast<size_t>(p) / sizeof(T);
    const size_t	vsize = vec<T>::size;
    
    return p + (vsize - 1 - (nelms - 1) % vsize);
}
    
//! 指定されたアドレスの直前のアライメントされているアドレスを返す
/*!
  \param p	アドレス
  \return	pの直前(pを含む)のアライメントされているアドレス
*/
template <class T> inline T*
end(T* p)
{
    const size_t	nelms = reinterpret_cast<size_t>(p) / sizeof(T);
    const size_t	vsize = vec<T>::size;
    
    return p - (nelms % vsize);
}

//! 指定されたアドレスの直前のアライメントされているアドレスを返す
/*!
  \param p	アドレス
  \return	pの直前(pを含む)のアライメントされているアドレス
*/
template <class T> inline const T*
cend(const T* p)
{
    const size_t	nelms = reinterpret_cast<size_t>(p) / sizeof(T);
    const size_t	vsize = vec<T>::size;
    
    return p - (nelms % vsize);
}

/************************************************************************
*  struct tuple2vec<T, N>						*
************************************************************************/
//! 同じ成分を持つboost::tupleをSIMDベクトルに変換する関数オブジェクト
/*!
  \param T	成分の型
  \param N	成分の個数(vec<T>::sizeに等しい)
*/
template <class T, size_t N=vec<T>::size>	struct tuple2vec;
template <class T>
struct tuple2vec<T, 1>
{
    typedef boost::tuple<T>				argument_type;
    typedef vec<T>					result_type;

    result_type	operator ()(const argument_type& t) const
		{
		    return result_type(boost::get<0>(t));
		}
};
template <class T>
struct tuple2vec<T, 2>
{
    typedef boost::tuple<T, T>				argument_type;
    typedef vec<T>					result_type;

    result_type	operator ()(const argument_type& t) const
		{
		    return result_type(boost::get<1>(t), boost::get<0>(t));
		}
};
template <class T>
struct tuple2vec<T, 4>
{
    typedef boost::tuple<T, T, T, T>			argument_type;
    typedef vec<T>					result_type;

    result_type	operator ()(const argument_type& t) const
		{
		    return result_type(boost::get<3>(t), boost::get<2>(t),
				       boost::get<1>(t), boost::get<0>(t));
		}
};
template <class T>
struct tuple2vec<T, 8>
{
    typedef boost::tuple<T, T, T, T, T, T, T, T>	argument_type;
    typedef vec<T>					result_type;

    result_type	operator ()(const argument_type& t) const
		{
		    return result_type(boost::get<7>(t), boost::get<6>(t),
				       boost::get<5>(t), boost::get<4>(t),
				       boost::get<3>(t), boost::get<2>(t),
				       boost::get<1>(t), boost::get<0>(t));
		}
};

namespace detail
{
    /********************************************************************
    *  class vec_tuple<TUPLE, S>					*
    ********************************************************************/
    //! 与えられたtupleと同数のSIMDベクトルから成るtuple
    /*!
      \param S		tupleを構成するSIMDベクトルの要素の型
      \param TUPLE	元のtuple
    */
    template <class TUPLE, class S=void>
    struct vec_tuple
    {
	template <class, class _S>
	struct impl
	{
	    typedef vec<_S>	type;
	};

	typedef typename boost::mpl::
	    if_<boost::is_same<S, void>,
		typename TUPLE::head_type::element_type,
		S>::type					element_type;
	typedef typename boost::detail::tuple_impl_specific::
	    tuple_meta_transform<
		TUPLE,
		impl<boost::mpl::_1, element_type> >::type	type;

	enum	{element_size = sizeof(element_type),
		 size	      = vec<element_type>::size};
    };
    template <class T, class S>
    struct vec_tuple<vec<T>, S>
    {
	typedef typename boost::mpl::
	    if_<boost::is_same<S, void>, T, S>::type		element_type;
	typedef vec<element_type>				type;

	enum	{element_size = sizeof(element_type),
		 size	      = vec<element_type>::size};
    };
}

/************************************************************************
*  class load_iterator<ITER, ALIGNED>					*
************************************************************************/
//! 反復子が指すアドレスからSIMDベクトルを読み込む反復子
/*!
  \param ITER		SIMDベクトルの読み込み元を指す反復子の型
  \param ALIGNED	読み込み元のアドレスがalignmentされていればtrue,
			そうでなければfalse
*/
template <class ITER, bool ALIGNED=false>
class load_iterator : public boost::iterator_adaptor<
			load_iterator<ITER, ALIGNED>,
			ITER,
			vec<typename std::iterator_traits<ITER>::value_type>,
			boost::use_default,
			vec<typename std::iterator_traits<ITER>::value_type> >
{
  private:
    typedef typename std::iterator_traits<ITER>::value_type	element_type;
    typedef boost::iterator_adaptor<load_iterator,
				    ITER,
				    vec<element_type>,
				    boost::use_default,
				    vec<element_type> >		super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

    
  public:
    load_iterator(ITER iter)	:super(iter)	{}
    load_iterator(const value_type* p)
	:super(reinterpret_cast<ITER>(p))	{}
	       
  private:
    reference		dereference() const
			{
			    return load<ALIGNED>(super::base());
			}
    void		advance(difference_type n)
			{
			    super::base_reference() += n * value_type::size;
			}
    void		increment()
			{
			    super::base_reference() += value_type::size;
			}
    void		decrement()
			{
			    super::base_reference() -= value_type::size;
			}
    difference_type	distance_to(load_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

namespace detail
{
    template <class ITER, class ALIGNED>
    struct loader
    {
	typedef load_iterator<ITER, ALIGNED::value>		type;
    };
    template <class ITER_TUPLE, class ALIGNED>
    struct loader<fast_zip_iterator<ITER_TUPLE>, ALIGNED>
    {
	typedef fast_zip_iterator<
	    typename boost::detail::tuple_impl_specific::
	    tuple_meta_transform<
		ITER_TUPLE,
		loader<boost::mpl::_1, ALIGNED> >::type>	type;
    };
}	// namespace detail
    
template <class ITER_TUPLE, bool ALIGNED>
class load_iterator<fast_zip_iterator<ITER_TUPLE>, ALIGNED>
    : public detail::loader<fast_zip_iterator<ITER_TUPLE>,
			    boost::mpl::bool_<ALIGNED> >::type
{
  private:
    typedef typename
	detail::loader<fast_zip_iterator<ITER_TUPLE>,
		       boost::mpl::bool_<ALIGNED> >::type	super;
    
    struct invoke
    {
	template <class ITER>
	struct apply
	{
	    typedef load_iterator<ITER, ALIGNED>		type;
	};

	template <class ITER> typename apply<ITER>::type
	operator ()(ITER const& iter) const
	{
	    return typename apply<ITER>::type(iter);
	}
    };

    struct base_iterator
    {
	template <class ITER>
	struct apply
	{
	    typedef typename ITER::base_type	type;
	};

	template <class ITER> typename apply<ITER>::type
	operator ()(ITER const& iter) const
	{
	    return iter.base();
	}
    };

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;
    typedef ITER_TUPLE				base_type;
    
  public:
    load_iterator(fast_zip_iterator<ITER_TUPLE> const& iter)
	:super(boost::detail::tuple_impl_specific::
	       tuple_transform(iter.get_iterator_tuple(), invoke()))	{}
    load_iterator(super const& iter)	:super(iter)			{}

    base_type	base() const
		{
		    return boost::detail::tuple_impl_specific::
			tuple_transform(super::get_iterator_tuple(),
					base_iterator());
		}
};

template <bool ALIGNED=false, class ITER> load_iterator<ITER, ALIGNED>
make_load_iterator(ITER iter)
{
    return load_iterator<ITER, ALIGNED>(iter);
}

template <bool ALIGNED=false, class T> load_iterator<const T*, ALIGNED>
make_load_iterator(const vec<T>* p)
{
    return load_iterator<const T*, ALIGNED>(p);
}

/************************************************************************
*  class store_iterator<ITER, ALIGNED>					*
************************************************************************/
namespace detail
{
    template <class ITER, bool ALIGNED=false>
    class store_proxy
    {
      public:
	typedef typename std::iterator_traits<ITER>::value_type	element_type;
	typedef vec<element_type>				value_type;
	typedef store_proxy					self;
	
      public:
	store_proxy(ITER iter)		:_iter(iter)			{}

		operator value_type() const
		{
		    return load<ALIGNED>(_iter);
		}
	self&	operator =(value_type val)
		{
		    store<ALIGNED>(_iter, val);
		    return *this;
		}
	self&	operator +=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) + val);
		}
	self&	operator -=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) - val);
		}
	self&	operator *=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) * val);
		}
	self&	operator /=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) / val);
		}
	self&	operator %=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) % val);
		}
	self&	operator &=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) & val);
		}
	self&	operator |=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) | val);
		}
	self&	operator ^=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) ^ val);
		}

      private:
	ITER 	_iter;
    };
}	// namespace detail

//! 反復子が指す書き込み先にSIMDベクトルを書き込む反復子
/*!
  \param ITER		SIMDベクトルの書き込み先を指す反復子の型
  \param ALIGNED	書き込み先アドレスがalignmentされていればtrue,
			そうでなければfalse
*/
template <class ITER, bool ALIGNED=false>
class store_iterator
    : public boost::iterator_adaptor<
		store_iterator<ITER, ALIGNED>,
		ITER,
		typename detail::store_proxy<ITER, ALIGNED>::value_type,
		boost::use_default,
		detail::store_proxy<ITER, ALIGNED> >
{
  private:
    typedef boost::iterator_adaptor<
		store_iterator,
		ITER,
		typename detail::store_proxy<ITER, ALIGNED>::value_type,
		boost::use_default,
		detail::store_proxy<ITER, ALIGNED> >		super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;
    
    friend class				boost::iterator_core_access;

  public:
    store_iterator(ITER iter)	:super(iter)	{}
    store_iterator(value_type* p)
	:super(reinterpret_cast<ITER>(p))	{}

    value_type		operator ()() const
			{
			    return load<ALIGNED>(super::base());
			}
    
  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type n)
			{
			    super::base_reference() += n * value_type::size;
			}
    void		increment()
			{
			    super::base_reference() += value_type::size;
			}
    void		decrement()
			{
			    super::base_reference() -= value_type::size;
			}
    difference_type	distance_to(store_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

namespace detail
{
    template <class ITER, class ALIGNED>
    struct storer
    {
	typedef store_iterator<ITER, ALIGNED::value>		type;
    };
    template <class ITER_TUPLE, class ALIGNED>
    struct storer<fast_zip_iterator<ITER_TUPLE>, ALIGNED>
    {
	typedef fast_zip_iterator<
	    typename boost::detail::tuple_impl_specific::
	    tuple_meta_transform<
		ITER_TUPLE,
		storer<boost::mpl::_1, ALIGNED> >::type>	type;
    };
}	// namespace detail

template <class ITER_TUPLE, bool ALIGNED>
class store_iterator<fast_zip_iterator<ITER_TUPLE>, ALIGNED>
    : public detail::storer<fast_zip_iterator<ITER_TUPLE>,
			    boost::mpl::bool_<ALIGNED> >::type
{
  private:
    typedef typename
	detail::storer<fast_zip_iterator<ITER_TUPLE>,
		       boost::mpl::bool_<ALIGNED> >::type	super;

    struct invoke
    {
	template <class ITER>
	struct apply
	{
	    typedef store_iterator<ITER, ALIGNED>		type;
	};

	template <class ITER> typename apply<ITER>::type
	operator ()(ITER const& iter) const
	{
	    return typename apply<ITER>::type(iter);
	}
    };

    struct base_iterator
    {
	template <class ITER>
	struct apply
	{
	    typedef typename ITER::base_type	type;
	};

	template <class ITER> typename apply<ITER>::type
	operator ()(ITER const& iter) const
	{
	    return iter.base();
	}
    };

    struct load
    {
	template <class ITER>
	struct apply
	{
	    typedef typename std::iterator_traits<ITER>::value_type	type;
	};

	template <class ITER> typename apply<ITER>::type
	operator ()(ITER const& iter) const
	{
	    return iter();
	}
    };

    template <class _ITER>
    struct value
    {
	typedef typename std::iterator_traits<_ITER>::value_type	type;
    };
    template <class _TUPLE>
    struct value<fast_zip_iterator<_TUPLE> >
    {
	typedef typename boost::detail::tuple_impl_specific::
	tuple_meta_transform<_TUPLE,
			     value<boost::mpl::_1> >::type		type;
    };
    
  public:
    typedef typename super::difference_type	difference_type;
    typedef typename value<super>::type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;
    typedef ITER_TUPLE				base_type;
    
  public:
    store_iterator(fast_zip_iterator<ITER_TUPLE> const& iter)
	:super(boost::detail::tuple_impl_specific::
	       tuple_transform(iter.get_iterator_tuple(), invoke()))	{}
    store_iterator(super const& iter)	:super(iter)			{}

    base_type	base() const
		{
		    return boost::detail::tuple_impl_specific::
			tuple_transform(super::get_iterator_tuple(),
					base_iterator());
		}
    
    value_type	operator ()() const
		{
		    return boost::detail::tuple_impl_specific::
			tuple_transform(super::get_iterator_tuple(), load());
		}
};
    
template <bool ALIGNED=false, class ITER> store_iterator<ITER, ALIGNED>
make_store_iterator(ITER iter)
{
    return store_iterator<ITER, ALIGNED>(iter);
}

template <bool ALIGNED=false, class T> store_iterator<T*, ALIGNED>
make_store_iterator(vec<T>* p)
{
    return store_iterator<T*, ALIGNED>(p);
}

/************************************************************************
*  class cvtdown_iterator<T, ITER>					*
************************************************************************/
//! SIMDベクトルを出力する反復子を介して複数のSIMDベクトルを読み込み，それをより小さな成分を持つSIMDベクトルに変換する反復子
/*!
  \param T	変換先のSIMDベクトルの成分の型
  \param ITER	SIMDベクトルを出力する反復子
*/
template <class T, class ITER>
class cvtdown_iterator
    : public boost::iterator_adaptor<
		cvtdown_iterator<T, ITER>,			// self
		ITER,						// base
		typename detail::vec_tuple<
		    typename std::iterator_traits<ITER>::value_type, T>::type,
		boost::single_pass_traversal_tag,
		typename detail::vec_tuple<
		    typename std::iterator_traits<ITER>::value_type, T>::type>
{
  private:
    typedef typename std::iterator_traits<ITER>::value_type	src_vec;
    typedef boost::iterator_adaptor<cvtdown_iterator,
				    ITER,
				    typename detail::vec_tuple<
					src_vec, T>::type,
				    boost::single_pass_traversal_tag,
				    typename detail::vec_tuple<
					src_vec, T>::type>	super;

    typedef typename detail::vec_tuple<src_vec>::element_type
							element_type;
    typedef typename type_traits<element_type>::complementary_type
							complementary_type;
    typedef typename detail::vec_tuple<
		src_vec, complementary_type>::type	complementary_vec;

    template <class _S>
    struct invoke
    {
	template <class _T> struct apply	{ typedef vec<_S> type; };
	template <class _T> typename apply<_T>::type
	operator ()(vec<_T> x)		  const	{ return cvt<_S>(x); }
	template <class _T> typename apply<_T>::type
	operator ()(vec<_T> x, vec<_T> y) const	{ return cvt<_S>(x, y); }
    };

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
		cvtdown_iterator(ITER const& iter)	:super(iter)	{}

  private:
    template <class _S, class _T> static
    vec<_S>	cvt(vec<_T> x)
		{
		    return mm::cvt<_S>(x);
		}
    template <class _S, class _TUPLE>
    static typename detail::vec_tuple<src_vec, _S>::type
		cvt(const _TUPLE& x)
		{
		    return boost::detail::tuple_impl_specific::
			tuple_transform(x, invoke<_S>());
		}
    template <class _S, class _T> static
    vec<_S>	cvt(vec<_T> x, vec<_T> y)
		{
		    return mm::cvt<_S>(x, y);
		}
    template <class _S, class _TUPLE>
    static typename detail::vec_tuple<src_vec, _S>::type
		cvt(const _TUPLE& x, const _TUPLE& y)
		{
		    return TU::detail::tuple_transform(x, y, invoke<_S>());
		}

    void	cvtdown(src_vec& x)
		{
		    x = *super::base();
		    ++super::base_reference();
		}
    void	cvtdown(complementary_vec& x)
		{
		    const bool	SameSize = (vec<complementary_type>::size ==
					    vec<element_type>::size);
		    cvtdown(x, boost::mpl::bool_<SameSize>());
		}
    void	cvtdown(complementary_vec& x, boost::mpl::true_)
		{
		    src_vec	y;
		    cvtdown(y);
		    x = cvt<complementary_type>(y);
		}
    void	cvtdown(complementary_vec& x, boost::mpl::false_)
		{
		    src_vec	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt<complementary_type>(y, z);
		}
    template <class _VEC>
    void	cvtdown(_VEC& x)
		{
		    typedef typename detail::
				vec_tuple<_VEC>::element_type	S;
		    typedef typename type_traits<S>::upper_type	upper_type;
		    typedef typename boost::mpl::if_<
				boost::is_floating_point<S>,
				upper_type,
				typename type_traits<upper_type>::signed_type>
				::type			signed_upper_type;
		    
		    typename detail::vec_tuple<
			src_vec, signed_upper_type>::type	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt<S>(y, z);
		}

    reference	dereference() const
		{
		    reference	x;
		    const_cast<cvtdown_iterator*>(this)->cvtdown(x);
		    return x;
		}
    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}
};
    
template <class T, class ITER> cvtdown_iterator<T, ITER>
make_cvtdown_iterator(ITER iter)
{
    return cvtdown_iterator<T, ITER>(iter);
}

/************************************************************************
*  class cvtup_iterator<ITER>						*
************************************************************************/
namespace detail
{
    template <class ITER>
    class cvtup_proxy
    {
      public:
	typedef typename detail::vec_tuple<
		    typename std::iterator_traits<ITER>
				::value_type>::type	value_type;
	typedef typename detail::vec_tuple<value_type>
			       ::element_type		element_type;
	typedef cvtup_proxy				self;

      private:
	template <class _S, size_t _I>
	struct invoke
	{
	    template <class _T> struct apply	{ typedef vec<_S> type; };
	    template <class _T> typename apply<_T>::type
	    operator ()(vec<_T> x)	const	{ return cvt<_S, _I>(x); }
	};

	typedef typename std::iterator_traits<ITER>::reference
							reference;
	typedef typename type_traits<element_type>::complementary_type
							complementary_type;
	typedef typename detail::vec_tuple<
		    value_type,
		    complementary_type>::type		complementary_vec;
	typedef typename boost::mpl::if_<
		    boost::is_floating_point<element_type>,
		    complementary_type,
		    element_type>::type			integral_type;
	typedef typename type_traits<
		    typename type_traits<integral_type>::lower_type>
		    ::unsigned_type			unsigned_lower_type;
	typedef typename detail::vec_tuple<
		    value_type,
		    unsigned_lower_type>::type		unsigned_lower_vec;
	
      private:
	template <class _S, size_t _I=0, class _T> static
	vec<_S>	cvt(vec<_T> x)
		{
		    return mm::cvt<_S, _I>(x);
		}
	template <class _S, size_t _I=0, class _TUPLE>
	static typename detail::vec_tuple<value_type, _S>::type
		cvt(const _TUPLE& x)
		{
		    return boost::detail::tuple_impl_specific::
			tuple_transform(x, invoke<_S, _I>());
		}

	template <class _OP>
	void	cvtup(value_type x)
		{
		    _OP()(x, *_iter);
		    ++_iter;
		}
	template <class _OP>
	void	cvtup(unsigned_lower_vec x)
		{
		    cvtup<_OP>(cvt<integral_type, 0>(x));
		    cvtup<_OP>(cvt<integral_type, 1>(x));
		}
	template <class _OP>
	void	cvtup(complementary_vec x)
		{
		    const bool	SameSize = (vec<complementary_type>::size ==
					    vec<element_type>::size);
		    cvtup<_OP>(x, boost::mpl::bool_<SameSize>());
		}
	template <class _OP>
	void	cvtup(complementary_vec x, boost::mpl::true_)
		{
		    cvtup<_OP>(cvt<element_type>(x));
		}
	template <class _OP>
	void	cvtup(complementary_vec x, boost::mpl::false_)
		{
		    cvtup<_OP>(cvt<element_type, 0>(x));
		    cvtup<_OP>(cvt<element_type, 1>(x));
		}
	template <class _OP, class _VEC>
	void	cvtup(_VEC x)
		{
		    typedef typename detail::
				vec_tuple<_VEC>::element_type	S;
		    typedef typename type_traits<S>::upper_type	upper_type;

		    cvtup<_OP>(cvt<upper_type, 0>(x));
		    cvtup<_OP>(cvt<upper_type, 1>(x));
		}

      public:
	cvtup_proxy(ITER const& iter) :_iter(const_cast<ITER&>(iter)) {}
	
	template <class _VEC>
	self&	operator =(_VEC x)
		{
		    cvtup<assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator +=(_VEC x)
		{
		    cvtup<plus_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator -=(_VEC x)
		{
		    cvtup<minus_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator *=(_VEC x)
		{
		    cvtup<multiplies_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator /=(_VEC x)
		{
		    cvtup<divides_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator %=(_VEC x)
		{
		    cvtup<modulus_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator &=(_VEC x)
		{
		    cvtup<bit_and_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator |=(_VEC x)
		{
		    cvtup<bit_or_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator ^=(_VEC x)
		{
		    cvtup<bit_xor_assign<value_type, reference> >(x);
		    return *this;
		}
	
      private:
	ITER&	_iter;
    };
}

//! SIMDベクトルを受け取ってより大きな成分を持つ複数のSIMDベクトルに変換し，それらを指定された反復子を介して書き込む反復子
/*!
  \param ITER	変換されたSIMDベクトルの書き込み先を指す反復子
*/
template <class ITER>
class cvtup_iterator
    : public boost::iterator_adaptor<
		 cvtup_iterator<ITER>,
		 ITER,
		 typename detail::cvtup_proxy<ITER>::value_type,
		 boost::single_pass_traversal_tag,
		 detail::cvtup_proxy<ITER> >
{
  private:
    typedef boost::iterator_adaptor<
		cvtup_iterator,
		ITER,
		typename detail::cvtup_proxy<ITER>::value_type,
		boost::single_pass_traversal_tag,
		detail::cvtup_proxy<ITER> >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    cvtup_iterator(ITER const& iter)	:super(iter)			{}

  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type)			{}
    void		increment()					{}
    void		decrement()					{}
    difference_type	distance_to(cvtup_iterator const& iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

template <class ITER> cvtup_iterator<ITER>
make_cvtup_iterator(ITER iter)
{
    return cvtup_iterator<ITER>(iter);
}

/************************************************************************
*  class cvtdown_mask_iterator<T, ITER>					*
************************************************************************/
//! SIMDマスクベクトルを出力する反復子を介して複数のSIMDマスクベクトルを読み込み，それをより小さな成分を持つSIMDマスクベクトルに変換する反復子
/*!
  \param T	変換先のSIMDマスクベクトルの成分の型
  \param ITER	SIMDマスクベクトルを出力する反復子
*/
template <class T, class ITER>
class cvtdown_mask_iterator
    : public boost::iterator_adaptor<
		cvtdown_mask_iterator<T, ITER>,			// self
		ITER,						// base
		typename detail::vec_tuple<
		    typename std::iterator_traits<ITER>::value_type, T>::type,
		boost::single_pass_traversal_tag,
		typename detail::vec_tuple<
		    typename std::iterator_traits<ITER>::value_type, T>::type>
{
  private:
    typedef typename std::iterator_traits<ITER>::value_type	src_vec;
    typedef boost::iterator_adaptor<cvtdown_mask_iterator,
				    ITER,
				    typename detail::vec_tuple<
					src_vec, T>::type,
				    boost::single_pass_traversal_tag,
				    typename detail::vec_tuple<
					src_vec, T>::type>	super;

    typedef typename detail::vec_tuple<src_vec>::element_type
							element_type;
    typedef typename type_traits<element_type>::complementary_mask_type
							complementary_type;
    typedef typename detail::vec_tuple<
		src_vec, complementary_type>::type	complementary_vec;

    template <class _S>
    struct invoke
    {
	template <class _T> struct apply	{ typedef vec<_S> type; };
	template <class _T> typename apply<_T>::type
	operator ()(vec<_T> x)		  const	{ return cvt_mask<_S>(x); }
	template <class _T> typename apply<_T>::type
	operator ()(vec<_T> x, vec<_T> y) const	{ return cvt_mask<_S>(x, y); }
    };

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
		cvtdown_mask_iterator(ITER const& iter)	:super(iter)	{}

  private:
    template <class _S, class _T> static
    vec<_S>	cvt_mask(vec<_T> x)
		{
		    return mm::cvt_mask<_S>(x);
		}
    template <class _S, class _TUPLE>
    static typename detail::vec_tuple<src_vec, _S>::type
		cvt_mask(const _TUPLE& x)
		{
		    return boost::detail::tuple_impl_specific::
			tuple_transform(x, invoke<_S>());
		}
    template <class _S, class _T> static
    vec<_S>	cvt_mask(vec<_T> x, vec<_T> y)
		{
		    return mm::cvt_mask<_S>(x, y);
		}
    template <class _S, class _TUPLE>
    static typename detail::vec_tuple<src_vec, _S>::type
		cvt_mask(const _TUPLE& x, const _TUPLE& y)
		{
		    return TU::detail::tuple_transform(x, y, invoke<_S>());
		}

    void	cvtdown(src_vec& x)
		{
		    x = *super::base();
		    ++super::base_reference();
		}
    void	cvtdown(complementary_vec& x)
		{
		    src_vec	y;
		    cvtdown(y);
		    x = cvt_mask<complementary_type>(y);
		}
    template <class _VEC>
    void	cvtdown(_VEC& x)
		{
		    typedef typename detail::
				vec_tuple<_VEC>::element_type	S;
		    typedef typename type_traits<S>::upper_type	upper_type;
		    typename detail::vec_tuple<
				 src_vec, upper_type>::type	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt_mask<S>(y, z);
		}

    reference	dereference() const
		{
		    reference	x;
		    const_cast<cvtdown_mask_iterator*>(this)->cvtdown(x);
		    return x;
		}
    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}
};
    
template <class T, class ITER> cvtdown_mask_iterator<T, ITER>
make_cvtdown_mask_iterator(ITER iter)
{
    return cvtdown_mask_iterator<T, ITER>(iter);
}

/************************************************************************
*  class cvtup_mask_iterator<ITER>					*
************************************************************************/
namespace detail
{
    template <class ITER>
    class cvtup_mask_proxy
    {
      public:
	typedef typename detail::vec_tuple<
		    typename std::iterator_traits<ITER>
				::value_type>::type	value_type;
	typedef typename detail::vec_tuple<value_type>
			       ::element_type		element_type;
	typedef cvtup_mask_proxy				self;

      private:
	template <class _S, size_t _I>
	struct invoke
	{
	    template <class _T> struct apply	{ typedef vec<_S> type; };
	    template <class _T> typename apply<_T>::type
	    operator ()(vec<_T> x)	const	{ return cvt_mask<_S, _I>(x); }
	};

	typedef typename std::iterator_traits<ITER>::reference
							reference;
	typedef typename type_traits<element_type>::complementary_mask_type
							complementary_type;
	typedef typename detail::vec_tuple<
		    value_type,
		    complementary_type>::type		complementary_vec;
	
      private:
	template <class _S, size_t _I=0, class _T> static
	vec<_S>	cvt_mask(vec<_T> x)
		{
		    return mm::cvt_mask<_S, _I>(x);
		}
	template <class _S, size_t _I=0, class _TUPLE>
	static typename detail::vec_tuple<value_type, _S>::type
		cvt_mask(const _TUPLE& x)
		{
		    return boost::detail::tuple_impl_specific::
			tuple_transform(x, invoke<_S, _I>());
		}

	template <class _OP>
	void	cvtup(value_type x)
		{
		    _OP()(x, *_iter);
		    ++_iter;
		}
	template <class _OP>
	void	cvtup(complementary_vec x)
		{
		    cvtup<_OP>(cvt_mask<element_type>(x));
		}
	template <class _OP, class _VEC>
	void	cvtup(_VEC x)
		{
		    typedef typename detail::
				vec_tuple<_VEC>::element_type	S;
		    typedef typename type_traits<S>::upper_type	upper_type;

		    cvtup<_OP>(cvt_mask<upper_type, 0>(x));
		    cvtup<_OP>(cvt_mask<upper_type, 1>(x));
		}

      public:
	cvtup_mask_proxy(ITER const& iter) :_iter(const_cast<ITER&>(iter)) {}
	
	template <class _VEC>
	self&	operator =(_VEC x)
		{
		    cvtup<assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator &=(_VEC x)
		{
		    cvtup<bit_and_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator |=(_VEC x)
		{
		    cvtup<bit_or_assign<value_type, reference> >(x);
		    return *this;
		}
	template <class _VEC>
	self&	operator ^=(_VEC x)
		{
		    cvtup<bit_xor_assign<value_type, reference> >(x);
		    return *this;
		}
	
      private:
	ITER&	_iter;
    };
}

//! SIMDベクトルを受け取ってより大きな成分を持つ複数のSIMDベクトルに変換し，それらを指定された反復子を介して書き込む反復子
/*!
  \param ITER	変換されたSIMDベクトルの書き込み先を指す反復子
*/
template <class ITER>
class cvtup_mask_iterator
    : public boost::iterator_adaptor<
		 cvtup_mask_iterator<ITER>,
		 ITER,
		 typename detail::cvtup_mask_proxy<ITER>::value_type,
		 boost::single_pass_traversal_tag,
		 detail::cvtup_mask_proxy<ITER> >
{
  private:
    typedef boost::iterator_adaptor<
		cvtup_mask_iterator,
		ITER,
		typename detail::cvtup_mask_proxy<ITER>::value_type,
		boost::single_pass_traversal_tag,
		detail::cvtup_mask_proxy<ITER> >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    cvtup_mask_iterator(ITER const& iter)	:super(iter)		{}

  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type)			{}
    void		increment()					{}
    void		decrement()					{}
    difference_type	distance_to(cvtup_mask_iterator const& iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

template <class ITER> cvtup_mask_iterator<ITER>
make_cvtup_mask_iterator(ITER iter)
{
    return cvtup_mask_iterator<ITER>(iter);
}

/************************************************************************
*  class shift_iterator<ITER>						*
************************************************************************/
template <class ITER>
class shift_iterator
    : public boost::iterator_adaptor<
			shift_iterator<ITER>,
			ITER,
			boost::use_default,
			boost::forward_traversal_tag,
			typename std::iterator_traits<ITER>::value_type>
{
  private:
    typedef boost::iterator_adaptor<
		shift_iterator,
		ITER,
		boost::use_default,
		boost::forward_traversal_tag,
		typename std::iterator_traits<ITER>::value_type>	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
		shift_iterator(ITER iter, size_t pos=0)
		    :super(iter), _pos(0), _val(*iter), _next(), _valid(true)
		{
		    while (pos--)
			increment();
		}

  private:
    void	shift() const
		{
		    _val  = shift_r<1>(_next, _val);
		    _next = shift_r<1>(_next);
		}
    void	load_and_shift() const
		{
		    _next  = *super::base();
		    _valid = true;
		    shift();
		}
    reference	dereference() const
		{
		    if (!_valid)		// !_valid なら必ず _pos == 1
			load_and_shift();
		    return _val;
		}
    void	increment()
		{
		    switch (++_pos)
		    {
		      case 1:
			++super::base_reference();
			_valid = false;
			break;
		      case value_type::size:
			_pos = 0;		// default:に落ちる
		      default:
			if (!_valid)		// !_valid なら必ず _pos == 2
			    load_and_shift();
			shift();
			break;
		    }
		}
    bool	equal(const shift_iterator& iter) const
		{
		    return (super::base() == iter.base()) &&
			   (_pos == iter._pos);
		}

  private:
    size_t		_pos;
    mutable value_type	_val, _next;
    mutable bool	_valid;	//!< _nextに入力値が読み込まれていればtrue
};

template <class ITER> shift_iterator<ITER>
make_shift_iterator(ITER iter)
{
    return shift_iterator<ITER>(iter);
}
    
/************************************************************************
*  class row_vec_iterator<T, ROW>					*
************************************************************************/
template <class T, class ROW>
class row_vec_iterator
    : public boost::iterator_adaptor<row_vec_iterator<T, ROW>,
				     row_iterator<
					 fast_zip_iterator<
					     typename iterator_tuple<
						 ROW, vec<T>::size>::type>,
					 boost::transform_iterator<
					     tuple2vec<T>,
					     typename subiterator<
						 fast_zip_iterator<
						     typename iterator_tuple<
							 ROW,
							 vec<T>::size>::type>
						 >::type>,
					 tuple2vec<T> > >
{
  private:
    typedef fast_zip_iterator<
      typename iterator_tuple<ROW, vec<T>::size>::type>	row_zip_iterator;
    typedef boost::iterator_adaptor<
		row_vec_iterator,
		row_iterator<
		    row_zip_iterator,
		    boost::transform_iterator<
			tuple2vec<T>,
			typename subiterator<
			    row_zip_iterator>::type>,
		    tuple2vec<T> > >			super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;
    
  public:
    row_vec_iterator(ROW const& row)
	:super(make_row_transform_iterator(
		   make_fast_zip_iterator(
		       make_iterator_tuple<vec<T>::size>(row)),
		   tuple2vec<T>()))					{}

    void		advance(difference_type n)
			{
			    super::base_reference() += n * vec<T>::size;
			}
    void		increment()
			{
			    super::base_reference() += vec<T>::size;
			}
    void		decrement()
			{
			    super::base_reference() -= vec<T>::size;
			}
    difference_type	distance_to(row_vec_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / vec<T>::size;
			}
};

template <class T, class ROW> inline row_vec_iterator<T, ROW>
make_row_vec_iterator(ROW const& row)
{
    return row_vec_iterator<T, ROW>(row);
}

}	// namespace mm
}	// namespace TU
#endif	// MMX

#endif	// !__mmInstructions_h
