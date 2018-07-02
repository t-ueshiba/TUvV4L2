/*!
  \file		vec.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトルクラスの定義
*/
#if !defined(TU_SIMD_VEC_H)
#define TU_SIMD_VEC_H

#include <iostream>
#include <utility>
#include <cassert>
#include "TU/type_traits.h"
#include "TU/simd/config.h"
#include "TU/simd/type_traits.h"

namespace TU
{
//! SIMD命令を利用するためのクラスや関数が定義されている名前空間
namespace simd
{
/************************************************************************
*  class vec<T>								*
************************************************************************/
//! SIMDベクトル型
/*!
 \param T	SIMDベクトルの成分の型
*/
template <class T>
class vec
{
  public:
    using element_type	= T;			//!< 成分の型    
    using base_type	= simd::base_type<T>;	//!< ベースとなるSIMDデータ型


    constexpr static size_t	element_size = sizeof(element_type);
    constexpr static size_t	size	     = sizeof(base_type)/element_size;
    constexpr static size_t	lane_size    = (sizeof(base_type) > 16 ?
						16/element_size : size);

  public:
    vec()						{}
    vec(element_type a)					;
    vec(element_type a0,  element_type a1)		;
    vec(element_type a0,  element_type a1,
	element_type a2,  element_type a3)		;
    vec(element_type a0,  element_type a1,
	element_type a2,  element_type a3,
	element_type a4,  element_type a5,
	element_type a6,  element_type a7)		;
    vec(element_type a0,  element_type a1,
	element_type a2,  element_type a3,
	element_type a4,  element_type a5,
	element_type a6,  element_type a7,
	element_type a8,  element_type a9,
	element_type a10, element_type a11,
	element_type a12, element_type a13,
	element_type a14, element_type a15)		;
    vec(element_type a0,  element_type a1,
	element_type a2,  element_type a3,
	element_type a4,  element_type a5,
	element_type a6,  element_type a7,
	element_type a8,  element_type a9,
	element_type a10, element_type a11,
	element_type a12, element_type a13,
	element_type a14, element_type a15,
	element_type a16, element_type a17,
	element_type a18, element_type a19,
	element_type a20, element_type a21,
	element_type a22, element_type a23,
	element_type a24, element_type a25,
	element_type a26, element_type a27,
	element_type a28, element_type a29,
	element_type a30, element_type a31)		;
    vec(element_type a0,  element_type a1,
	element_type a2,  element_type a3,
	element_type a4,  element_type a5,
	element_type a6,  element_type a7,
	element_type a8,  element_type a9,
	element_type a10, element_type a11,
	element_type a12, element_type a13,
	element_type a14, element_type a15,
	element_type a16, element_type a17,
	element_type a18, element_type a19,
	element_type a20, element_type a21,
	element_type a22, element_type a23,
	element_type a24, element_type a25,
	element_type a26, element_type a27,
	element_type a28, element_type a29,
	element_type a30, element_type a31,
	element_type a32, element_type a33,
	element_type a34, element_type a35,
	element_type a36, element_type a37,
	element_type a38, element_type a39,
	element_type a40, element_type a41,
	element_type a42, element_type a43,
	element_type a44, element_type a45,
	element_type a46, element_type a47,
	element_type a48, element_type a49,
	element_type a50, element_type a51,
	element_type a52, element_type a53,
	element_type a54, element_type a55,
	element_type a56, element_type a57,
	element_type a58, element_type a59,
	element_type a60, element_type a61,
	element_type a62, element_type a63)		;

    vec&		operator =(element_type a)	;
    
    template <size_t ...IDX>
    vec(std::index_sequence<IDX...>)	:vec(IDX...)	{}
  // ベース型との間の型変換
    vec(base_type m)	:_base(m)		{}
			operator base_type()	{ return _base; }

    vec&		flip_sign()		{ return *this = -*this; }
    vec&		operator +=(vec x)	{ return *this = *this + x; }
    vec&		operator -=(vec x)	{ return *this = *this - x; }
    vec&		operator *=(vec x)	{ return *this = *this * x; }
    vec&		operator &=(vec x)	{ return *this = *this & x; }
    vec&		operator |=(vec x)	{ return *this = *this | x; }
    vec&		operator ^=(vec x)	{ return *this = *this ^ x; }
    vec&		andnot(vec x)		;

    element_type	operator [](size_t i) const
			{
			    assert(i < size);
			    return *(reinterpret_cast<const element_type*>(
					 &_base) + i);
			}
    element_type&	operator [](size_t i)
			{
			    assert(i < size);
			    return *(reinterpret_cast<element_type*>(&_base)
				     + i);
			}
    
    static size_t	floor(size_t n)	{ return size*(n/size); }
    static size_t	ceil(size_t n)	{ return (n == 0 ? 0 :
						  size*((n - 1)/size + 1)); }

  private:
    base_type		_base;
};

//! 連続した整数値で初期化されたSIMDベクトルを生成する．
template <class T> vec<T>
make_contiguous_vec()
{
    return vec<T>(std::make_index_sequence<vec<T>::size>());
}
    
//! SIMDベクトルの内容をストリームに出力する．
/*!
  \param out	出力ストリーム
  \param vec	SIMDベクトル
  \return	outで指定した出力ストリーム
*/
template <class T> std::ostream&
operator <<(std::ostream& out, const vec<T>& x)
{
    using element_type	= std::conditional_t<
				(std::is_same<T, int8_t >::value ||
				 std::is_same<T, uint8_t>::value),
				int32_t, T>;

    for (size_t i = 0; i < vec<T>::size; ++i)
	out << ' ' << element_type(x[i]);

    return out;
}
    
using Is8vec	= vec<int8_t>;		//!< 符号付き8bit整数ベクトル
using Is16vec	= vec<int16_t>;		//!< 符号付き16bit整数ベクトル
using Is32vec	= vec<int32_t>;		//!< 符号付き32bit整数ベクトル
using Is64vec	= vec<int64_t>;		//!< 符号付き64bit整数ベクトル
using Iu8vec	= vec<uint8_t>;		//!< 符号なし8bit整数ベクトル
using Iu16vec	= vec<uint16_t>;	//!< 符号なし16bit整数ベクトル
using Iu32vec	= vec<uint32_t>;	//!< 符号なし32bit整数ベクトル
using Iu64vec	= vec<uint64_t>;	//!< 符号なし64bit整数ベクトル

/************************************************************************
*  predicate: is_vec<T>							*
************************************************************************/
//! 与えられた型が simd::vec 又はそれに変換可能であるか判定する
/*!
  \param T	判定対象となる型
*/
template <class T>
using is_vec	= is_convertible<T, vec>;

/************************************************************************
*  Control functions							*
************************************************************************/
void	empty()								;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/vec.h"
#elif defined(NEON)
#  include "TU/simd/arm/vec.h"
#endif

#endif	// !TU_SIMD_VEC_H
