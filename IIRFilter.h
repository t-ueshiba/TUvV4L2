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
 *  $Id: IIRFilter.h,v 1.14 2012-01-22 10:52:19 ueshiba Exp $
 */
/*!
  \file		IIRFilter.h
  \brief	各種infinite impulse response filterに関するクラスの定義と実装
*/
#ifndef	__TUIIRFilterPP_h
#define	__TUIIRFilterPP_h

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
forward2(const S*& src, float*& dst, F32vec c0123, F32vec& tmp);

template <> inline void
forward2(const u_char*& src, float*& dst, F32vec c0123, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size,
		nelms8 = Iu8vec::size;
    Iu8vec	in8 = loadu(src);
    storeu(dst, forward2(cvt<float>(in8), c0123, tmp));
    dst += nelmsF;
    storeu(dst, forward2(cvt<float>(shift_r<nelmsF>(in8)), c0123, tmp));
    dst += nelmsF;
    storeu(dst, forward2(cvt<float>(shift_r<2*nelmsF>(in8)), c0123, tmp));
    dst += nelmsF;
    storeu(dst, forward2(cvt<float>(shift_r<3*nelmsF>(in8)), c0123, tmp));
    dst += nelmsF;
    src += nelms8;
}

template <> inline void
forward2(const float*& src, float*& dst, F32vec c0123, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size;
    storeu(dst, forward2(loadu(src), c0123, tmp));
    dst += nelmsF;
    src += nelmsF;
}

template <class S> static void
backward2(const S*& src, float*& dst, F32vec c1032, F32vec& tmp);

template <> inline void
backward2(const u_char*& src, float*& dst, F32vec c1032, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size,
		nelms8 = Iu8vec::size;
    src -= nelms8;
    Iu8vec	in8 = loadu(src);
    dst -= nelmsF;
    storeu(dst, backward2(cvt<float>(shift_r<3*nelmsF>(in8)), c1032, tmp));
    dst -= nelmsF;
    storeu(dst, backward2(cvt<float>(shift_r<2*nelmsF>(in8)), c1032, tmp));
    dst -= nelmsF;
    storeu(dst, backward2(cvt<float>(shift_r<nelmsF>(in8)), c1032, tmp));
    dst -= nelmsF;
    storeu(dst, backward2(cvt<float>(in8), c1032, tmp));
}

template <> inline void
backward2(const float*& src, float*& dst, F32vec c1032, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size;
    src -= nelmsF;
    dst -= nelmsF;
    storeu(dst, backward2(loadu(src), c1032, tmp));
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
forward4(const S*& src, float*& dst, F32vec c0123, F32vec c4567, F32vec& tmp);

template <> inline void
forward4(const u_char*& src, float*& dst,
	 F32vec c0123, F32vec c4567, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size,
		nelms8 = Iu8vec::size;
    Iu8vec	in8 = loadu(src);
    storeu(dst, forward4(cvt<float>(in8), c0123, c4567, tmp));
    dst += nelmsF;
    storeu(dst,
	   forward4(cvt<float>(shift_r<nelmsF>(in8)), c0123, c4567, tmp));
    dst += nelmsF;
    storeu(dst,
	   forward4(cvt<float>(shift_r<2*nelmsF>(in8)), c0123, c4567, tmp));
    dst += nelmsF;
    storeu(dst,
	   forward4(cvt<float>(shift_r<3*nelmsF>(in8)), c0123, c4567, tmp));
    dst += nelmsF;
    src += nelms8;
}

template <> inline void
forward4(const float*& src, float*& dst,
	 F32vec c0123, F32vec c4567, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size;
    storeu(dst, forward4(loadu(src), c0123, c4567, tmp));
    dst += nelmsF;
    src += nelmsF;
}

template <class S> static void
backward4(const S*& src, float*& dst, F32vec c3210, F32vec c7654, F32vec& tmp);

template <> inline void
backward4(const u_char*& src, float*& dst,
	  F32vec c3210, F32vec c7654, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size,
		nelms8 = Iu8vec::size;
    src -= nelms8;
    Iu8vec	in8 = loadu(src);
    dst -= nelmsF;
    storeu(dst,
	   backward4(cvt<float>(shift_r<3*nelmsF>(in8)), c3210, c7654, tmp));
    dst -= nelmsF;
    storeu(dst,
	   backward4(cvt<float>(shift_r<2*nelmsF>(in8)), c3210, c7654, tmp));
    dst -= nelmsF;
    storeu(dst,
	   backward4(cvt<float>(shift_r<nelmsF>(in8)), c3210, c7654, tmp));
    dst -= nelmsF;
    storeu(dst, backward4(cvt<float>(in8), c3210, c7654, tmp));
}

template <> inline void
backward4(const float*& src, float*& dst,
	  F32vec c3210, F32vec c7654, F32vec& tmp)
{
    const u_int	nelmsF = F32vec::size;
    src -= nelmsF;
    dst -= nelmsF;
    storeu(dst, backward4(loadu(src), c3210, c7654, tmp));
}
}
#endif

namespace TU
{
/************************************************************************
*  class IIRFilter<D>							*
************************************************************************/
//! 片側Infinite Inpulse Response Filterを表すクラス
template <u_int D> class IIRFilter
{
  public:
    IIRFilter&	initialize(const float c[D+D])				;
    template <class T, class B, class B2> const IIRFilter&
		forward(const Array<T, B>& in,
			Array<float, B2>& out)			const	;
    template <class T, class B, class B2> const IIRFilter&
		backward(const Array<T, B>& in,
			 Array<float, B2>& out)			const	;
    void	limitsF(float& limit0F,
			float& limit1F, float& limit2F)		const	;
    void	limitsB(float& limit0B,
			float& limit1B, float& limit2B)		const	;
    
  private:
    float	_c[D+D];	// coefficients
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
template <u_int D> IIRFilter<D>&
IIRFilter<D>::initialize(const float c[D+D])
{
    for (int i = 0; i < D+D; ++i)
	_c[i] = c[i];

    return *this;
}

//! 前進方向にフィルタを適用する
/*!
  \param in	入力データ列
  \param out	出力データ列
  \return	このフィルタ自身
*/
template <u_int D> template <class T, class B, class B2>
const IIRFilter<D>&
IIRFilter<D>::forward(const Array<T, B>& in, Array<float, B2>& out) const
{
    out.resize(in.dim());

    const T*	src = in.begin();
    for (float* dst = out.begin(); dst < out.end(); ++dst)
    {
	int	d = dst - out.begin();

	if (d < D)
	    *dst = _c[D-1-d]*src[-d];
	else
	{
	    d = D;
	    *dst = 0;
	}
	for (int i = -d; i++ < 0; )
	    *dst += (_c[D-1+i]*src[i] + _c[D+D-1+i]*dst[i-1]);
	++src;
    }

    return *this;
}
    
//! 後退方向にフィルタを適用する
/*!
  \param in	入力データ列
  \param out	出力データ列
  \return	このフィルタ自身
*/
template <u_int D> template <class T, class B, class B2>
const IIRFilter<D>&
IIRFilter<D>::backward(const Array<T, B>& in, Array<float, B2>& out) const
{
    out.resize(in.dim());

    const T*	src = in.end();
    for (float* dst = out.end(); dst > out.begin(); )
    {
	const int	d = std::min(int(D), out.end() - dst);

	--src;
      	*--dst = 0;
	for (int i = 0; i < d; ++i)
	    *dst += (_c[i]*src[i+1] + _c[D+i]*dst[i+1]);
    }

    return *this;
}

template <> template <class T, class B, class B2> const IIRFilter<2u>&
IIRFilter<2u>::forward(const Array<T, B>& in, Array<float, B2>& out) const
{
    if (in.dim() < 2)
	return *this;

    out.resize(in.dim());
    const T*	src = in.begin();
    float*	dst = out.begin();
#if defined(SSE2)
    using namespace	mm;
    
    const F32vec	c0123(_c[0], _c[1], _c[2], _c[3]);
    F32vec		tmp(_c[0]*src[1], _c[0]*src[0] + _c[1]*src[1],
			    _c[1]*src[0], 0.0);
    src += 2;			// forward2()のために2つ前進．
    for (const float* const tail2 = out.end() - vec<T>::size - 2;
	 dst <= tail2; )	// srcがdstよりも2つ前進しているのでoverrunに注意．
	forward2(src, dst, c0123, tmp);
    empty();
    src -= 2;			// 2つ前進していた分を元に戻す．
#else
    *dst = _c[1]*src[0];
    ++src;
    ++dst;
    *dst = _c[0]*src[-1] + _c[1]*src[0] + _c[3]*dst[-1];
    ++src;
    ++dst;
#endif
    for (; dst < out.end(); ++dst)
    {
	*dst = _c[0]*src[-1] + _c[1]*src[0] + _c[2]*dst[-2] + _c[3]*dst[-1];
	++src;
    }

    return *this;
}
    
template <> template <class T, class B, class B2> const IIRFilter<2u>&
IIRFilter<2u>::backward(const Array<T, B>& in, Array<float, B2>& out) const
{
    if (in.dim() < 2)
	return *this;

    out.resize(in.dim());
    float*	dst = out.end();
    const T*	src = in.end();
#if defined(SSE2)
    using namespace	mm;

    const F32vec	c1032(_c[1], _c[0], _c[3], _c[2]);
    F32vec		tmp(_c[1]*src[-2], _c[1]*src[-1] + _c[0]*src[-2],
			    _c[0]*src[-1], 0.0);
    src -= 2;			// backward2()のために2つ後退．
    for (const float* const head2 = out.begin() + vec<T>::size + 2;
	 dst >= head2; )	// srcがdstよりも2つ後退しているのでoverrunに注意．
	backward2(src, dst, c1032, tmp);
    empty();
    src += 2;			// 2つ後退していた分を元に戻す．
#else
    --src;
    --dst;
    *dst = 0.0;
    --src;
    --dst;
    *dst = _c[0]*src[1] + _c[2]*dst[1];
#endif
    while (--dst >= out.begin())
    {
	--src;
	*dst = _c[0]*src[1] + _c[1]*src[2] + _c[2]*dst[1] + _c[3]*dst[2];
    }

    return *this;
}

template <> template <class T, class B, class B2> const IIRFilter<4u>&
IIRFilter<4u>::forward(const Array<T, B>& in, Array<float, B2>& out) const
{
    if (in.dim() < 4)
	return *this;

    out.resize(in.dim());
    const T*	src = in.begin();
    float*	dst = out.begin();
#if defined(SSE2)
    using namespace	mm;

    const F32vec	c0123(_c[0], _c[1], _c[2], _c[3]),
			c4567(_c[4], _c[5], _c[6], _c[7]);
    F32vec		tmp = zero<float>();
    for (const float* const tail2 = out.end() - vec<T>::size;
	 dst <= tail2; )
	forward4(src, dst, c0123, c4567, tmp);
    empty();
#else
    *dst = _c[3]*src[0];
    ++src;
    ++dst;
    *dst = _c[2]*src[-1] + _c[3]*src[0]
			 + _c[7]*dst[-1];
    ++src;
    ++dst;
    *dst = _c[1]*src[-2] + _c[2]*src[-1] + _c[3]*src[0]
			 + _c[6]*dst[-2] + _c[7]*dst[-1];
    ++src;
    ++dst;
    *dst = _c[0]*src[-3] + _c[1]*src[-2] + _c[2]*src[-1] + _c[3]*src[0]
			 + _c[5]*dst[-3] + _c[6]*dst[-2] + _c[7]*dst[-1];
    ++src;
    ++dst;
#endif
    for (; dst < out.end(); ++dst)
    {
	*dst = _c[0]*src[-3] + _c[1]*src[-2] + _c[2]*src[-1] + _c[3]*src[0]
	     + _c[4]*dst[-4] + _c[5]*dst[-3] + _c[6]*dst[-2] + _c[7]*dst[-1];
	++src;
    }

    return *this;
}
    
template <> template <class T, class B, class B2> const IIRFilter<4u>&
IIRFilter<4u>::backward(const Array<T, B>& in, Array<float, B2>& out) const
{
    if (in.dim() < 4)
	return *this;

    out.resize(in.dim());
    float*	dst = out.end();
    const T*	src = in.end();
#if defined(SSE2)
    using namespace	mm;

    const F32vec	c3210(_c[3], _c[2], _c[1], _c[0]),
			c7654(_c[7], _c[6], _c[5], _c[4]);
    F32vec		tmp = zero<float>();
    for (const float* const head2 = out.begin() + vec<T>::size;
	 dst >= head2; )
	backward4(src, dst, c3210, c7654, tmp);
    empty();
#else
    --src;
    --dst;
    *dst = 0.0;
    --src;
    --dst;
    *dst = _c[0]*src[1]
	 + _c[4]*dst[1];
    --src;
    --dst;
    *dst = _c[0]*src[1] + _c[1]*src[2]
	 + _c[4]*dst[1] + _c[5]*dst[2];
    --src;
    --dst;
    *dst = _c[0]*src[1] + _c[1]*src[2] + _c[2]*src[3]
	 + _c[4]*dst[1] + _c[5]*dst[2] + _c[6]*dst[3];
#endif
    while (--dst >= out.begin())
    {
	--src;
	*dst = _c[0]*src[1] + _c[1]*src[2] + _c[2]*src[3] + _c[3]*src[4]
	     + _c[4]*dst[1] + _c[5]*dst[2] + _c[6]*dst[3] + _c[7]*dst[4];
    }

    return *this;
}

//! 特定の入力データ列に対して前進方向にフィルタを適用した場合の極限値を求める
/*!
  \param limit0F	一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1F	傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2F	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <u_int D> void
IIRFilter<D>::limitsF(float& limit0F, float& limit1F, float& limit2F) const
{
    float	n0 = 0.0, d0 = 1.0, n1 = 0.0, d1 = 0.0, n2 = 0.0, d2 = 0.0;
    for (int i = 0; i < D; ++i)
    {
	n0 +=	      _c[i];
	d0 -=	      _c[D+i];
	n1 +=	    i*_c[D-1-i];
	d1 -=	(i+1)*_c[D+D-1-i];
	n2 += (i-1)*i*_c[D-1-i];
	d2 -= i*(i+1)*_c[D+D-1-i];
    }
    const float	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2.0*x1*d1 - x0*d2)/d0;
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
template <u_int D> void
IIRFilter<D>::limitsB(float& limit0B, float& limit1B, float& limit2B) const
{
    float	n0 = 0.0, d0 = 1.0, n1 = 0.0, d1 = 0.0, n2 = 0.0, d2 = 0.0;
    for (int i = 0; i < D; ++i)
    {
	n0 +=	      _c[i];
	d0 -=	      _c[D+i];
	n1 +=	(i+1)*_c[i];
	d1 -=	(i+1)*_c[D+i];
	n2 += i*(i+1)*_c[i];
	d2 -= i*(i+1)*_c[D+i];
    }
    const float	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2.0*x1*d1 - x0*d2)/d0;
    limit0B = x0;
    limit1B = x1;
    limit2B = x1 + x2;
}

/************************************************************************
*  class BilateralIIRFilter<D>						*
************************************************************************/
//! 両側Infinite Inpulse Response Filterを表すクラス
template <u_int D> class BilateralIIRFilter
{
  public:
  //! 微分の階数
    enum Order
    {
	Zeroth,						//!< 0階微分
	First,						//!< 1階微分
	Second						//!< 2階微分
    };
    
    BilateralIIRFilter&
		initialize(const float cF[D+D], const float cB[D+D])	;
    BilateralIIRFilter&
		initialize(const float c[D+D], Order order)		;
    template <class T, class B> const BilateralIIRFilter&
		convolve(const Array<T, B>& in)			const	;
    template <class T1, class B1, class T2, class B2> const BilateralIIRFilter&
		convolve(const Array<T1, B1>& in,
			 Array<T2, B2>& out)			const	;
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    const BilateralIIRFilter&
		operator ()(const Array2<T1, B1, R1>& in,
			    Array2<T2, B2, R2>& out,
			    u_int is=0, u_int ie=0)		const	;
    u_int	dim()						const	;
    float	operator [](int i)				const	;
    void	limits(float& limit0,
		       float& limit1, float& limit2)		const	;

  private:
#if defined(USE_TBB)
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    class ConvolveRow
    {
      public:
	ConvolveRow(const BilateralIIRFilter& biir,
		    const Array2<T1, B1, R1>& in,
		    Array2<T2, B2, R2>& out)
	    :_biir(biir), _in(in), _out(out)				{}

	void	operator ()(const tbb::blocked_range<u_int>& r) const
		{
		    for (u_int i = r.begin(); i != r.end(); ++i)
			_biir.convolveRow(_in[i], _out, i);
		}

      private:
	BilateralIIRFilter		_biir;	// 参照ではなく実体
	const Array2<T1, B1, R1>&	_in;
	Array2<T2, B2, R2>&		_out;
    };
#endif
    template <class T, class B, class T2, class B2, class R2>
    void	convolveRow(const Array<T, B>& in,
			    Array2<T2, B2, R2>& out, u_int i)	const	;
    
  private:
    IIRFilter<D>		_iirF;
    mutable Array<float>	_bufF;
    IIRFilter<D>		_iirB;
    mutable Array<float>	_bufB;
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
template <u_int D> inline BilateralIIRFilter<D>&
BilateralIIRFilter<D>::initialize(const float cF[D+D], const float cB[D+D])
{
    _iirF.initialize(cF);
    _iirB.initialize(cB);
#ifdef _DEBUG
  /*float	limit0, limit1, limit2;
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
template <u_int D> BilateralIIRFilter<D>&
BilateralIIRFilter<D>::initialize(const float c[D+D], Order order)
{
  // Compute 0th, 1st and 2nd derivatives of the forward z-transform
  // functions at z = 1.
    float	n0 = 0.0, d0 = 1.0, n1 = 0.0, d1 = 0.0, n2 = 0.0, d2 = 0.0;
    for (int i = 0; i < D; ++i)
    {
	n0 +=	      c[i];
	d0 -=	      c[D+i];
	n1 +=	    i*c[D-1-i];
	d1 -=	(i+1)*c[D+D-1-i];
	n2 += (i-1)*i*c[D-1-i];
	d2 -= i*(i+1)*c[D+D-1-i];
    }
    const float	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2.0*x1*d1 - x0*d2)/d0;
    
  // Compute denominators.
    float	cF[D+D], cB[D+D];
    for (int i = 0; i < D; ++i)
	cB[D+D-1-i] = cF[D+i] = c[D+i];

  // Compute nominators.
    if (order == First)	// Antisymmetric filter
    {
	const float	k = -0.5/x1;
	cF[D-1] = cB[D-1] = 0.0;			// i(n), i(n+D)
	for (int i = 0; i < D-1; ++i)
	{
	    cF[i]     = k*c[i];				// i(n-D+1+i)
	    cB[D-2-i] = -cF[i];				// i(n+D-1-i)
	}
    }
    else		// Symmetric filter
    {
	const float	k = (order == Second ? 1.0 / (x1 + x2)
					     : 1.0 / (2.0*x0 - c[D-1]));
	cF[D-1] = k*c[D-1];				// i(n)
	cB[D-1] = cF[D-1] * c[D];			// i(n+D)
	for (int i = 0; i < D-1; ++i)
	{
	    cF[i]     = k*c[i];				// i(n-D+1+i)
	    cB[D-2-i] = cF[i] + cF[D-1] * cF[D+1+i];	// i(n+D-1-i)
	}
    }

    return initialize(cF, cB);
}
    
//! フィルタによる畳み込みを行う. 出力は #operator []() で取り出す
/*!
  \param in	入力データ列
  \return	このフィルタ自身
*/
template <u_int D> template <class T, class B>
inline const BilateralIIRFilter<D>&
BilateralIIRFilter<D>::convolve(const Array<T, B>& in) const
{
    _iirF.forward(in, _bufF);
    _iirB.backward(in, _bufB);

    return *this;
}

//! フィルタによる畳み込みを行う. 
/*!
  \param in	入力データ列
  \param out	出力データ列
  \return	このフィルタ自身
*/
template <u_int D> template <class T1, class B1, class T2, class B2>
inline const BilateralIIRFilter<D>&
BilateralIIRFilter<D>::convolve(const Array<T1, B1>& in,
				Array<T2, B2>& out) const
{
    convolve(in);
    out.resize(dim());
    for (int i = 0; i < dim(); ++i)
	out[i] = (*this)[i];

    return *this;
}

//! 2次元配列の各行に対してフィルタによる畳み込みを行う.
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \param is	処理を開始する行
  \param ie	処理を終了する次の行．0を与えると最後の行まで処理する．
  \return	このフィルタ自身
*/
template <u_int D>
template <class T1, class B1, class R1, class T2, class B2, class R2>
const BilateralIIRFilter<D>&
BilateralIIRFilter<D>::operator ()(const Array2<T1, B1, R1>& in,
				   Array2<T2, B2, R2>& out,
				   u_int is, u_int ie) const
{
    out.resize(in.ncol(), in.nrow());
    if (ie == 0)
	ie = in.nrow();

#if defined(USE_TBB)
    using namespace	tbb;

    parallel_for(blocked_range<u_int>(is, ie, 1),
		 ConvolveRow<T1, B1, R1, T2, B2, R2>(*this, in, out));
#else
    for (u_int i = is; i < ie; ++i)
	convolveRow(in[i], out, i);
#endif
    return *this;
}

//! 畳み込みの出力データ列の次元を返す
/*!
  \return	出力データ列の次元
*/
template <u_int D> inline u_int
BilateralIIRFilter<D>::dim() const
{
    return _bufF.dim();
}

//! 畳み込みの出力データの特定の要素を返す
/*!
  \param i	要素のindex
  \return	要素の値
*/
template <u_int D> inline float
BilateralIIRFilter<D>::operator [](int i) const
{
    return _bufF[i] + _bufB[i];
}

//! 特定の入力データ列に対してフィルタを適用した場合の極限値を求める
/*!
  \param limit0		一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1		傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2		2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <u_int D> void
BilateralIIRFilter<D>::limits(float& limit0,
			      float& limit1, float& limit2) const
{
    float	limit0F, limit1F, limit2F;
    _iirF.limitsF(limit0F, limit1F, limit2F);

    float	limit0B, limit1B, limit2B;
    _iirB.limitsB(limit0B, limit1B, limit2B);

    limit0 = limit0F + limit0B;
    limit1 = limit1F + limit1B;
    limit2 = limit2F + limit2B;
}

template <u_int D>
template <class T, class B, class T2, class B2, class R2> void
BilateralIIRFilter<D>::convolveRow(const Array<T, B>& in,
				   Array2<T2, B2, R2>& out, u_int i) const
{
    convolve(in);
    T2*	col = &out[0];
    for (u_int j = 0; j < dim(); ++j)
	(*col++)[i] = (*this)[j];
}
    
/************************************************************************
*  class BilateralIIRFilter2<D>						*
************************************************************************/
//! 2次元両側Infinite Inpulse Response Filterを表すクラス
template <u_int D> class BilateralIIRFilter2
{
  public:
    typedef typename BilateralIIRFilter<D>::Order	Order;
    
    BilateralIIRFilter2()	:_biirH(), _biirV()			{}

    BilateralIIRFilter2&
		initialize(float cHF[], float cHB[],
			   float cVF[], float cVB[])			;
    BilateralIIRFilter2&
		initialize(float cHF[], Order orderH,
			   float cVF[], Order orderV)			;
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    const BilateralIIRFilter2&
		convolve(const Array2<T1, B1, R1>& in,
			 Array2<T2, B2, R2>& out)		const	;
    
  private:
    BilateralIIRFilter<D>		_biirH;
    BilateralIIRFilter<D>		_biirV;
    mutable Array2<Array<float> >	_buf;
};
    
//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数
  \param cHB	横方向後退z変換係数
  \param cVF	縦方向前進z変換係数
  \param cVB	縦方向後退z変換係数
  \return	このフィルタ自身
*/
template <u_int D> inline BilateralIIRFilter2<D>&
BilateralIIRFilter2<D>::initialize(float cHF[], float cHB[],
				   float cVF[], float cVB[])
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
template <u_int D> inline BilateralIIRFilter2<D>&
BilateralIIRFilter2<D>::initialize(float cHF[], Order orderH,
				   float cVF[], Order orderV)
{
    _biirH.initialize(cHF, orderH);
    _biirV.initialize(cVF, orderV);

    return *this;
}

//! 与えられた2次元配列とこのフィルタの畳み込みを行う
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このフィルタ自身
*/
template <u_int D>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline const BilateralIIRFilter2<D>&
BilateralIIRFilter2<D>::convolve(const Array2<T1, B1, R1>& in,
				 Array2<T2, B2, R2>& out) const
{
    _biirH(in, _buf);
    _biirV(_buf, out);

    return *this;
}

}
#endif	/* !__TUIIRFilterPP_h */
