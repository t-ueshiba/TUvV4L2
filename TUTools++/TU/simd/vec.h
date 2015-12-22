/*
 *  $Id$
 */
#if !defined(__TU_SIMD_VEC_H)
#define __TU_SIMD_VEC_H

#include <iostream>
#include "TU/simd/config.h"
#include "TU/simd/type_traits.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class vec<T>								*
************************************************************************/
template <class T>
class vec
{
  public:
    typedef T			element_type;	//!< 成分の型    
    typedef simd::base_type<T>	base_type;	//!< ベースとなるSIMDデータ型


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

    vec&		operator =(element_type a)	;
    
    template <size_t ...IDX>
    vec(index_sequence<IDX...>)	:vec(IDX...)	{}
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
    return vec<T>(make_index_sequence<vec<T>::size>());
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
    typedef typename std::conditional<
	(std::is_same<T, int8_t  >::value ||
	 std::is_same<T, u_int8_t>::value), int32_t, T>::type	element_type;

    for (size_t i = 0; i < vec<T>::size; ++i)
	out << ' ' << element_type(x[i]);

    return out;
}
    
typedef vec<int8_t>	Is8vec;		//!< 符号付き8bit整数ベクトル
typedef vec<int16_t>	Is16vec;	//!< 符号付き16bit整数ベクトル
typedef vec<int32_t>	Is32vec;	//!< 符号付き32bit整数ベクトル
typedef vec<int64_t>	Is64vec;	//!< 符号付き64bit整数ベクトル
typedef vec<u_int8_t>	Iu8vec;		//!< 符号なし8bit整数ベクトル
typedef vec<u_int16_t>	Iu16vec;	//!< 符号なし16bit整数ベクトル
typedef vec<u_int32_t>	Iu32vec;	//!< 符号なし32bit整数ベクトル
typedef vec<u_int64_t>	Iu64vec;	//!< 符号なし64bit整数ベクトル

/************************************************************************
*  Predicates for template meta-programming				*
************************************************************************/
//! 与えられた型が何らかの mm::vec であるかを判定する meta-programming 用の predicate
template <class T> struct is_vec		: std::false_type	{};
template <class T> struct is_vec<vec<T> >	: std::true_type	{};

/************************************************************************
*  Control functions							*
************************************************************************/
void	empty()								;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/vec.h"
#elif defined(NEON)
#  include "TU/simd/arm/vec.h"
#endif

#endif	// !__TU_SIMD_VEC_H
