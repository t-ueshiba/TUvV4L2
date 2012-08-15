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
 *  $Id: IIRFilter.h,v 1.17 2012-08-15 07:58:19 ueshiba Exp $
 */
/*!
  \file		IIRFilter.h
  \brief	各種infinite impulse response filterに関するクラスの定義と実装
*/
#ifndef	__TUIIRFilterPP_h
#define	__TUIIRFilterPP_h

#include <iterator>
#include <algorithm>
#include "TU/Array++.h"
#include "TU/mmInstructions.h"
#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif

#if defined(SSE2)
namespace mm
{
/************************************************************************
*  static functions							*
************************************************************************/
static inline F32vec
forward2(F32vec in3210, F32vec c0123, F32vec& tmp)
{
    tmp = shift_r<1>(tmp) + c0123 * shuffle<0, 0, 0, 0>(tmp, in3210);
    F32vec	out0123 = tmp;
    tmp = shift_r<1>(tmp) + c0123 * shuffle<1, 1, 0, 0>(tmp, in3210);
    out0123 = replace_rmost(shift_l<1>(out0123), tmp);
    tmp = shift_r<1>(tmp) + c0123 * shuffle<2, 2, 0, 0>(tmp, in3210);
    out0123 = replace_rmost(shift_l<1>(out0123), tmp);
    tmp = shift_r<1>(tmp) + c0123 * shuffle<3, 3, 0, 0>(tmp, in3210);
    return reverse(replace_rmost(shift_l<1>(out0123), tmp));
}

static inline F32vec
backward2(F32vec in3210, F32vec c1032, F32vec& tmp)
{
    F32vec	out3210 = tmp;
    tmp = shift_r<1>(tmp) + c1032 * shuffle<3, 3, 0, 0>(tmp, in3210);
    out3210 = replace_rmost(shift_l<1>(out3210), tmp);
    tmp = shift_r<1>(tmp) + c1032 * shuffle<2, 2, 0, 0>(tmp, in3210);
    out3210 = replace_rmost(shift_l<1>(out3210), tmp);
    tmp = shift_r<1>(tmp) + c1032 * shuffle<1, 1, 0, 0>(tmp, in3210);
    out3210 = replace_rmost(shift_l<1>(out3210), tmp);
    tmp = shift_r<1>(tmp) + c1032 * shuffle<0, 0, 0, 0>(tmp, in3210);
    return out3210;
}

template <class S> static void
forward2(const S*& in, float*& out, F32vec c0123, F32vec& tmp);

template <> inline void
forward2(const u_char*& in, float*& out, F32vec c0123, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size,
		nelms8 = Iu8vec::size;
    Iu8vec	in8 = loadu(in);
    storeu(out, forward2(cvt<float>(in8), c0123, tmp));
    out += nelmsF;
    storeu(out, forward2(cvt<float>(shift_r<nelmsF>(in8)), c0123, tmp));
    out += nelmsF;
    storeu(out, forward2(cvt<float>(shift_r<2*nelmsF>(in8)), c0123, tmp));
    out += nelmsF;
    storeu(out, forward2(cvt<float>(shift_r<3*nelmsF>(in8)), c0123, tmp));
    out += nelmsF;
    in  += nelms8;
}

template <> inline void
forward2(const float*& in, float*& out, F32vec c0123, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size;
    storeu(out, forward2(loadu(in), c0123, tmp));
    out += nelmsF;
    in  += nelmsF;
}

template <class S> static void
backward2(const S*& in, float*& out, F32vec c1032, F32vec& tmp);

template <> inline void
backward2(const u_char*& in, float*& out, F32vec c1032, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size,
		nelms8 = Iu8vec::size;
    in -= nelms8;
    Iu8vec	in8 = loadu(in);
    out -= nelmsF;
    storeu(out, backward2(cvt<float>(shift_r<3*nelmsF>(in8)), c1032, tmp));
    out -= nelmsF;
    storeu(out, backward2(cvt<float>(shift_r<2*nelmsF>(in8)), c1032, tmp));
    out -= nelmsF;
    storeu(out, backward2(cvt<float>(shift_r<nelmsF>(in8)), c1032, tmp));
    out -= nelmsF;
    storeu(out, backward2(cvt<float>(in8), c1032, tmp));
}

template <> inline void
backward2(const float*& in, float*& out, F32vec c1032, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size;
    in  -= nelmsF;
    out -= nelmsF;
    storeu(out, backward2(loadu(in), c1032, tmp));
}

static inline F32vec
forward4(F32vec in3210, F32vec c0123, F32vec c4567, F32vec& tmp)
{
    tmp = shift_r<1>(tmp) + c0123 * set1<0>(in3210) + c4567 * set1<0>(tmp);
    F32vec	out0123 = tmp;
    tmp = shift_r<1>(tmp) + c0123 * set1<1>(in3210) + c4567 * set1<0>(tmp);
    out0123 = replace_rmost(shift_l<1>(out0123), tmp);
    tmp = shift_r<1>(tmp) + c0123 * set1<2>(in3210) + c4567 * set1<0>(tmp);
    out0123 = replace_rmost(shift_l<1>(out0123), tmp);
    tmp = shift_r<1>(tmp) + c0123 * set1<3>(in3210) + c4567 * set1<0>(tmp);
    return reverse(replace_rmost(shift_l<1>(out0123), tmp));
}

static inline F32vec
backward4(F32vec in3210, F32vec c3210, F32vec c7654, F32vec& tmp)
{
    F32vec	out3210 = tmp;
    tmp = shift_r<1>(tmp) + c3210 * set1<3>(in3210) + c7654 * set1<0>(tmp);
    out3210 = replace_rmost(shift_l<1>(out3210), tmp);
    tmp = shift_r<1>(tmp) + c3210 * set1<2>(in3210) + c7654 * set1<0>(tmp);
    out3210 = replace_rmost(shift_l<1>(out3210), tmp);
    tmp = shift_r<1>(tmp) + c3210 * set1<1>(in3210) + c7654 * set1<0>(tmp);
    out3210 = replace_rmost(shift_l<1>(out3210), tmp);
    tmp = shift_r<1>(tmp) + c3210 * set1<0>(in3210) + c7654 * set1<0>(tmp);
    return out3210;
}

template <class S> static void
forward4(const S*& in, float*& out, F32vec c0123, F32vec c4567, F32vec& tmp);

template <> inline void
forward4(const u_char*& in, float*& out,
	 F32vec c0123, F32vec c4567, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size,
		nelms8 = Iu8vec::size;
    Iu8vec	in8 = loadu(in);
    storeu(out, forward4(cvt<float>(in8), c0123, c4567, tmp));
    out += nelmsF;
    storeu(out,
	   forward4(cvt<float>(shift_r<nelmsF>(in8)), c0123, c4567, tmp));
    out += nelmsF;
    storeu(out,
	   forward4(cvt<float>(shift_r<2*nelmsF>(in8)), c0123, c4567, tmp));
    out += nelmsF;
    storeu(out,
	   forward4(cvt<float>(shift_r<3*nelmsF>(in8)), c0123, c4567, tmp));
    out += nelmsF;
    in  += nelms8;
}

template <> inline void
forward4(const float*& in, float*& out,
	 F32vec c0123, F32vec c4567, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size;
    storeu(out, forward4(loadu(in), c0123, c4567, tmp));
    out += nelmsF;
    in  += nelmsF;
}

template <class S> static void
backward4(const S*& in, float*& out, F32vec c3210, F32vec c7654, F32vec& tmp);

template <> inline void
backward4(const u_char*& in, float*& out,
	  F32vec c3210, F32vec c7654, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size,
		nelms8 = Iu8vec::size;
    in -= nelms8;
    Iu8vec	in8 = loadu(in);
    out -= nelmsF;
    storeu(out,
	   backward4(cvt<float>(shift_r<3*nelmsF>(in8)), c3210, c7654, tmp));
    out -= nelmsF;
    storeu(out,
	   backward4(cvt<float>(shift_r<2*nelmsF>(in8)), c3210, c7654, tmp));
    out -= nelmsF;
    storeu(out,
	   backward4(cvt<float>(shift_r<nelmsF>(in8)), c3210, c7654, tmp));
    out -= nelmsF;
    storeu(out, backward4(cvt<float>(in8), c3210, c7654, tmp));
}

template <> inline void
backward4(const float*& in, float*& out,
	  F32vec c3210, F32vec c7654, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size;
    in  -= nelmsF;
    out -= nelmsF;
    storeu(out, backward4(loadu(in), c3210, c7654, tmp));
}
}
#endif

namespace TU
{
/************************************************************************
*  class IIRFilter<D, T>						*
************************************************************************/
//! 片側Infinite Inpulse Response Filterを表すクラス
template <u_int D, class T=float> class IIRFilter
{
  public:
    IIRFilter&	initialize(const T c[D+D])				;
    void	limitsF(T& limit0F, T& limit1F, T& limit2F)	const	;
    void	limitsB(T& limit0B, T& limit1B, T& limit2B)	const	;
    template <class IN, class OUT>
    OUT		forward(IN ib, IN ie, OUT out)			const	;
    template <class IN, class OUT> OUT
		backward(IN ib, IN ie, OUT out)			const	;
    
  private:
    T		_c[D+D];	// coefficients
};

//! フィルタのz変換係数をセットする
/*!
  \param c	z変換係数. z変換関数は，前進フィルタの場合は
		\f[
		  H(z^{-1}) = \frac{c_{D-1} + c_{D-2}z^{-1} + c_{D-3}z^{-2} +
		  \cdots
		  + c_{0}z^{-(D-1)}}{1 - c_{2D-1}z^{-1} - c_{2D-2}z^{-2} -
		  \cdots - c_{D}z^{-D}}
		\f]
		後退フィルタの場合は
		\f[
		  H(z) = \frac{c_{0}z + c_{1}z^2 + \cdots + c_{D-1}z^D}
		       {1 - c_{D}z - c_{D+1}z^2 - \cdots - c_{2D-1}z^D}
		\f]
  \return	このフィルタ自身
*/
template <u_int D, class T> IIRFilter<D, T>&
IIRFilter<D, T>::initialize(const T c[D+D])
{
    for (u_int i = 0; i < D+D; ++i)
	_c[i] = c[i];

    return *this;
}

//! 特定の入力データ列に対して前進方向にフィルタを適用した場合の極限値を求める
/*!
  \param limit0F	一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1F	傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2F	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <u_int D, class T> void
IIRFilter<D, T>::limitsF(T& limit0F, T& limit1F, T& limit2F) const
{
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (u_int i = 0; i < D; ++i)
    {
	n0 +=	      _c[i];
	d0 -=	      _c[D+i];
	n1 +=	    i*_c[D-1-i];
	d1 -=	(i+1)*_c[D+D-1-i];
	n2 += (i-1)*i*_c[D-1-i];
	d2 -= i*(i+1)*_c[D+D-1-i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    limit0F =  x0;
    limit1F = -x1;
    limit2F =  x1 + x2;
}

//! 特定の入力データ列に対して後退方向にフィルタを適用した場合の極限値を求める
/*!
  \param limit0B	一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1B	傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2B	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <u_int D, class T> void
IIRFilter<D, T>::limitsB(T& limit0B, T& limit1B, T& limit2B) const
{
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (u_int i = 0; i < D; ++i)
    {
	n0 +=	      _c[i];
	d0 -=	      _c[D+i];
	n1 +=	(i+1)*_c[i];
	d1 -=	(i+1)*_c[D+i];
	n2 += i*(i+1)*_c[i];
	d2 -= i*(i+1)*_c[D+i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    limit0B = x0;
    limit1B = x1;
    limit2B = x1 + x2;
}

//! 前進方向にフィルタを適用する
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <u_int D, class T> template <class IN, class OUT> OUT
IIRFilter<D, T>::forward(IN ib, IN ie, OUT out) const
{
    typedef typename std::iterator_traits<IN>::difference_type	diff_t;
    
    for (IN in = ib; in != ie; ++in)
    {
	diff_t	d = in - ib;

	if (d < diff_t(D))
	    *out = _c[D-1-d]*in[-d];
	else
	{
	    d = D;
	    *out = 0;
	}
	for (diff_t i = -d; i++ < 0; )
	    *out += (_c[D-1+i]*in[i] + _c[D+D-1+i]*out[i-1]);
	++out;
    }

    return out;
}
    
//! 後退方向にフィルタを適用する
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <u_int D, class T> template <class IN, class OUT> OUT
IIRFilter<D, T>::backward(IN ib, IN ie, OUT oe) const
{
    typedef typename std::iterator_traits<IN>::difference_type	diff_t;
    
    for (IN in = ie; in != ib; )
    {
	const diff_t	d = std::min(diff_t(D), ie - in);

	--in;
      	*--oe = 0;
	for (diff_t i = 0; i < d; ++i)
	    *oe += (_c[i]*in[i+1] + _c[D+i]*oe[i+1]);
    }

    return oe;
}

template <> template <class IN, class OUT> OUT
IIRFilter<2u, float>::forward(IN ib, IN ie, OUT out) const
{
    typedef typename std::iterator_traits<IN>::value_type	ivalue_type;
    
    const ivalue_type*	in = ib;
#if defined(SSE2)
    using namespace	mm;
    
    const F32vec	c0123(_c[0], _c[1], _c[2], _c[3]);
    F32vec		tmp(_c[0]*in[1], _c[0]*in[0] + _c[1]*in[1],
			    _c[1]*in[0], 0.0);
    in += 2;		// forward2()のために2つ前進．
    for (IN tail2 = ie - vec<ivalue_type>::size - 2;
	 in <= tail2; )	// inがoutよりも2つ前進しているのでoverrunに注意．
	forward2(in, out, c0123, tmp);
    empty();
    in -= 2;		// 2つ前進していた分を元に戻す．
#else
    *out = _c[1]*in[0];
    ++in;
    ++out;
    *out = _c[0]*in[-1] + _c[1]*in[0] + _c[3]*out[-1];
    ++in;
    ++out;
#endif
    for (; in < ie; ++in)
    {
	*out = _c[0]*in[-1] + _c[1]*in[0] + _c[2]*out[-2] + _c[3]*out[-1];
	++out;
    }

    return out;
}
    
template <> template <class IN, class OUT> OUT
IIRFilter<2u, float>::backward(IN ib, IN ie, OUT oe) const
{
    typedef typename std::iterator_traits<IN>::value_type	ivalue_type;
    
    const ivalue_type*	in = ie;
#if defined(SSE2)
    using namespace	mm;

    const F32vec	c1032(_c[1], _c[0], _c[3], _c[2]);
    F32vec		tmp(_c[1]*in[-2], _c[1]*in[-1] + _c[0]*in[-2],
			    _c[0]*in[-1], 0.0);
    in -= 2;		// backward2()のために2つ後退．
    for (IN head2 = ib + vec<ivalue_type>::size + 2;
	 in >= head2; )	// inがoeよりも2つ後退しているのでoverrunに注意．
	backward2(in, oe, c1032, tmp);
    empty();
    in += 2;			// 2つ後退していた分を元に戻す．
#else
    --in;
    --oe;
    *oe = 0.0;
    --in;
    --oe;
    *oe = _c[0]*in[1] + _c[2]*oe[1];
#endif
    while (--in >= ib)
    {
	--oe;
	*oe = _c[0]*in[1] + _c[1]*in[2] + _c[2]*oe[1] + _c[3]*oe[2];
    }

    return oe;
}

template <> template <class IN, class OUT> OUT
IIRFilter<4u, float>::forward(IN ib, IN ie, OUT out) const
{
    typedef typename std::iterator_traits<IN>::value_type	ivalue_type;
    
    const ivalue_type*	in = ib;
#if defined(SSE2)
    using namespace	mm;

    const F32vec	c0123(_c[0], _c[1], _c[2], _c[3]),
			c4567(_c[4], _c[5], _c[6], _c[7]);
    F32vec		tmp = zero<float>();
    for (IN tail2 = ie - vec<ivalue_type>::size; in <= tail2; )
	forward4(in, out, c0123, c4567, tmp);
    empty();
#else
    *out = _c[3]*in[0];
    ++in;
    ++out;
    *out = _c[2]*in[-1] + _c[3]*in[0]
			+ _c[7]*out[-1];
    ++in;
    ++out;
    *out = _c[1]*in[-2] + _c[2]*in[-1]  + _c[3]*in[0]
			+ _c[6]*out[-2] + _c[7]*out[-1];
    ++in;
    ++out;
    *out = _c[0]*in[-3] + _c[1]*in[-2]  + _c[2]*in[-1]  + _c[3]*in[0]
			+ _c[5]*out[-3] + _c[6]*out[-2] + _c[7]*out[-1];
    ++in;
    ++out;
#endif
    for (; in < ie; ++in)
    {
	*out = _c[0]*in[-3]  + _c[1]*in[-2]  + _c[2]*in[-1]  + _c[3]*in[0]
	     + _c[4]*out[-4] + _c[5]*out[-3] + _c[6]*out[-2] + _c[7]*out[-1];
	++out;
    }

    return out;
}
    
template <> template <class IN, class OUT> OUT
IIRFilter<4u, float>::backward(IN ib, IN ie, OUT oe) const
{
    typedef typename std::iterator_traits<IN>::value_type	ivalue_type;

    const ivalue_type*	in = ie;
#if defined(SSE2)
    using namespace	mm;

    const F32vec	c3210(_c[3], _c[2], _c[1], _c[0]),
			c7654(_c[7], _c[6], _c[5], _c[4]);
    F32vec		tmp = zero<float>();
    for (IN head2 = ib + vec<ivalue_type>::size; in >= head2; )
	backward4(in, oe, c3210, c7654, tmp);
    empty();
#else
    --in;
    --oe;
    *oe = 0.0;
    --in;
    --oe;
    *oe = _c[0]*in[1]
	+ _c[4]*oe[1];
    --in;
    --oe;
    *oe = _c[0]*in[1] + _c[1]*in[2]
	+ _c[4]*oe[1] + _c[5]*oe[2];
    --in;
    --oe;
    *oe = _c[0]*in[1] + _c[1]*in[2] + _c[2]*in[3]
	+ _c[4]*oe[1] + _c[5]*oe[2] + _c[6]*oe[3];
#endif
    while (--in >= ib)
    {
	--oe;
	*oe = _c[0]*in[1] + _c[1]*in[2] + _c[2]*in[3] + _c[3]*in[4]
	    + _c[4]*oe[1] + _c[5]*oe[2] + _c[6]*oe[3] + _c[7]*oe[4];
    }

    return oe;
}

/************************************************************************
*  class BidirectionalIIRFilter<D, T>					*
************************************************************************/
//! 両側Infinite Inpulse Response Filterを表すクラス
template <u_int D, class T=float> class BidirectionalIIRFilter
{
  private:
    typedef Array<T>	buf_type;
    
  public:
  //! 微分の階数
    enum Order
    {
	Zeroth,		//!< 0階微分
	First,		//!< 1階微分
	Second		//!< 2階微分
    };

    BidirectionalIIRFilter&
		initialize(const T cF[D+D], const T cB[D+D])		;
    BidirectionalIIRFilter&
		initialize(const T c[D+D], Order order)			;
    void	limits(T& limit0, T& limit1, T& limit2)		const	;
    template <class IN, class OUT>
    OUT		operator ()(IN ib, IN ie, OUT out)		const	;

  private:
    IIRFilter<D, T>	_iirF;
    IIRFilter<D, T>	_iirB;
    mutable buf_type	_bufF;
    mutable buf_type	_bufB;
};

//! フィルタのz変換係数をセットする
/*!
  \param cF	前進z変換係数. z変換は 
		\f[
		  H^F(z^{-1}) = \frac{c^F_{D-1} + c^F_{D-2}z^{-1}
		  + c^F_{D-3}z^{-2} + \cdots
		  + c^F_{0}z^{-(D-1)}}{1 - c^F_{2D-1}z^{-1}
		  - c^F_{2D-2}z^{-2} - \cdots - c^F_{D}z^{-D}}
		\f]
		となる. 
  \param cB	後退z変換係数. z変換は
		\f[
		  H^B(z) = \frac{c^B_{0}z + c^B_{1}z^2 + \cdots + c^B_{D-1}z^D}
		       {1 - c^B_{D}z - c^B_{D+1}z^2 - \cdots - c^B_{2D-1}z^D}
		\f]
		となる.
*/
template <u_int D, class T> inline BidirectionalIIRFilter<D, T>&
BidirectionalIIRFilter<D, T>::initialize(const T cF[D+D], const T cB[D+D])
{
    _iirF.initialize(cF);
    _iirB.initialize(cB);
#ifdef _DEBUG
  /*T	limit0, limit1, limit2;
    limits(limit0, limit1, limit2);
    std::cerr << "limit0 = " << limit0 << ", limit1 = " << limit1
    << ", limit2 = " << limit2 << std::endl;*/
#endif
    return *this;
}

//! 両側フィルタのz変換係数をセットする
/*!
  \param c	前進方向z変換係数. z変換関数は
		\f[
		  H(z^{-1}) = \frac{c_{D-1} + c_{D-2}z^{-1} + c_{D-3}z^{-2} +
		  \cdots
		  + c_{0}z^{-(D-1)}}{1 - c_{2D-1}z^{-1} - c_{2D-2}z^{-2} -
		  \cdots - c_{D}z^{-D}}
		\f]
  \param order	フィルタの微分階数． #Zeroth または #Second ならば対称フィルタ
		として， #First ならば反対称フィルタとして自動的に後退方向の
		z変換係数を計算する． #Zeroth, #First, #Second のときに，それ
		ぞれ in(n) = 1, in(n) = n, in(n) = n^2 に対する出力が
		1, 1, 2になるよう，全体のスケールも調整される．
  \return	このフィルタ自身
*/
template <u_int D, class T> BidirectionalIIRFilter<D, T>&
BidirectionalIIRFilter<D, T>::initialize(const T c[D+D], Order order)
{
  // Compute 0th, 1st and 2nd derivatives of the forward z-transform
  // functions at z = 1.
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (u_int i = 0; i < D; ++i)
    {
	n0 +=	      c[i];
	d0 -=	      c[D+i];
	n1 +=	    i*c[D-1-i];
	d1 -=	(i+1)*c[D+D-1-i];
	n2 += (i-1)*i*c[D-1-i];
	d2 -= i*(i+1)*c[D+D-1-i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    
  // Compute denominators.
    T	cF[D+D], cB[D+D];
    for (u_int i = 0; i < D; ++i)
	cB[D+D-1-i] = cF[D+i] = c[D+i];

  // Compute nominators.
    if (order == First)	// Antisymmetric filter
    {
	const T	k = -0.5/x1;
	cF[D-1] = cB[D-1] = 0;
	for (u_int i = 0; i < D-1; ++i)
	{
	    cF[i]     = k*c[i];				// i(n-D+1+i)
	    cB[D-2-i] = -cF[i];				// i(n+D-1-i)
	}
    }
    else		// Symmetric filter
    {
	const T	k = (order == Second ? 1.0 / (x1 + x2)
				     : 1.0 / (2.0*x0 - c[D-1]));
	cF[D-1] = k*c[D-1];				// i(n)
	cB[D-1] = cF[D-1] * c[D];			// i(n+D)
	for (u_int i = 0; i < D-1; ++i)
	{
	    cF[i]     = k*c[i];				// i(n-D+1+i)
	    cB[D-2-i] = cF[i] + cF[D-1] * cF[D+1+i];	// i(n+D-1-i)
	}
    }

    return initialize(cF, cB);
}
    
//! 特定の入力データ列に対してフィルタを適用した場合の極限値を求める
/*!
  \param limit0		一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1		傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2		2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <u_int D, class T> void
BidirectionalIIRFilter<D, T>::limits(T& limit0, T& limit1, T& limit2) const
{
    T	limit0F, limit1F, limit2F;
    _iirF.limitsF(limit0F, limit1F, limit2F);

    T	limit0B, limit1B, limit2B;
    _iirB.limitsB(limit0B, limit1B, limit2B);

    limit0 = limit0F + limit0B;
    limit1 = limit1F + limit1B;
    limit2 = limit2F + limit2B;
}

//! フィルタによる畳み込みを行う. 
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <u_int D, class T> template <class IN, class OUT> inline OUT
BidirectionalIIRFilter<D, T>::operator ()(IN ib, IN ie, OUT out) const
{
    size_t	size = std::distance(ib, ie);
    _bufF.resize(size);
    _bufB.resize(size);

    _iirF.forward (ib, ie, _bufF.begin());
    _iirB.backward(ib, ie, _bufB.end());

    return std::transform(_bufF.begin(), _bufF.end(), _bufB.begin(),
			  out, std::plus<T>());
}

/************************************************************************
*  class BidirectionalIIRFilter2<D, T>					*
************************************************************************/
//! 2次元両側Infinite Inpulse Response Filterを表すクラス
template <u_int D, class T=float> class BidirectionalIIRFilter2
{
  public:
    typedef typename BidirectionalIIRFilter<D, T>::Order	Order;

  private:
    typedef Array2<Array<T> >				buf_type;
    typedef typename buf_type::iterator			buf_iterator;
    typedef typename buf_type::const_iterator		const_buf_iterator;
    
    template <class RowIter>
    struct vertical_iterator
	: public std::iterator<
	    std::bidirectional_iterator_tag,
	    typename std::iterator_traits<RowIter>::value_type::iterator>
    {
      private:
	typedef typename
	    std::iterator_traits<RowIter>::value_type	col_type;

      public:
	typedef typename col_type::difference_type	difference_type;
	typedef typename col_type::value_type		value_type;
	typedef typename col_type::reference		reference;
	typedef typename col_type::iterator		iterator;

      public:
	vertical_iterator(RowIter row, difference_type offset)
	    :_row(row), _offset(offset)				{}

	iterator
		operator ->()	 const	{ return _row->begin() + _offset; }
	reference
		operator *()	 const	{ return *operator ->(); }
	vertical_iterator&
		operator ++()		{ ++_row; return *this; }
	vertical_iterator
		operator ++(int) const	{
					    vertical_iterator	tmp = *this;
					    operator ++();
					    return tmp;
					}

      private:
	RowIter			_row;
	const difference_type	_offset;
    };
#if defined(USE_TBB)
    template <class IN, class OUT> class ConvolveRows
    {
      public:
	ConvolveRows(const BidirectionalIIRFilter<D, T>& biir,
		     IN in, OUT out)
	    :_biir(biir), _in(in), _out(out)			{}

	void	operator ()(const tbb::blocked_range<u_int>& r) const
		{
		    for (u_int i = r.begin(); i != r.end(); ++i)
			_biir(_in[i].begin(), _in[i].end(),
			      vertical_iterator<OUT>(_out, i));
		}
	
      private:
	const BidirectionalIIRFilter<D, T>	_biir;
	const IN				_in;
	const OUT				_out;
    };
#endif
    
  public:
    BidirectionalIIRFilter2&
		initialize(const T cHF[], const T cHB[],
			   const T cVF[], const T cVB[])	;
    BidirectionalIIRFilter2&
		initialize(const T cHF[], Order orderH,
			   const T cVF[], Order orderV)		;
    template <class IN, class OUT>
    OUT		operator ()(IN ib, IN ie, OUT out)	const	;
    
  private:
    BidirectionalIIRFilter<D, T>	_biirH;
    BidirectionalIIRFilter<D, T>	_biirV;
    mutable buf_type			_buf;
};
    
//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数
  \param cHB	横方向後退z変換係数
  \param cVF	縦方向前進z変換係数
  \param cVB	縦方向後退z変換係数
  \return	このフィルタ自身
*/
template <u_int D, class T> inline BidirectionalIIRFilter2<D, T>&
BidirectionalIIRFilter2<D, T>::initialize(const T cHF[], const T cHB[],
					  const T cVF[], const T cVB[])
{
    _biirH.initialize(cHF, cHB);
    _biirV.initialize(cVF, cVB);

    return *this;
}

//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数
  \param orderH 横方向微分階数
  \param cVF	縦方向前進z変換係数
  \param orderV	縦方向微分階数
  \return	このフィルタ自身
*/
template <u_int D, class T> inline BidirectionalIIRFilter2<D, T>&
BidirectionalIIRFilter2<D, T>::initialize(const T cHF[], Order orderH,
					  const T cVF[], Order orderV)
{
    _biirH.initialize(cHF, orderH);
    _biirV.initialize(cVF, orderV);

    return *this;
}

//! 与えられた2次元配列とこのフィルタの畳み込みを行う
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <u_int D, class T> template <class IN, class OUT> OUT
BidirectionalIIRFilter2<D, T>::operator ()(IN ib, IN ie, OUT out) const
{
    _buf.resize((ib != ie ? std::distance(ib->begin(), ib->end()) : 0),
		std::distance(ib, ie));
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<u_int>(0, _buf.ncol(), 1),
		      ConvolveRows<IN, buf_iterator>(
			  _biirH, ib, _buf.begin()));

    tbb::parallel_for(tbb::blocked_range<u_int>(0, _buf.nrow(), 1),
		      ConvolveRows<const_buf_iterator, OUT>(
			  _biirV, _buf.begin(), out));

    return out + _buf.ncol();
#else
    for (buf_iterator brow = _buf.begin(); ib != ie; ++ib)
	_biirH(ib->begin(), ib->end(),
	       vertical_iterator<buf_iterator>(_buf.begin(),
					       brow++ - _buf.begin()));

    const OUT	out0 = out;
    for (const_buf_iterator brow = _buf.begin(); brow != _buf.end(); ++brow)
	_biirV(brow->begin(), brow->end(),
	       vertical_iterator<OUT>(out0, out++ - out0));

    return out;
#endif
}

}
#endif	/* !__TUIIRFilterPP_h */
