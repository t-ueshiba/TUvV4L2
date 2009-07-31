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
 *  $Id: IIRFilter.h,v 1.7 2009-07-31 07:04:44 ueshiba Exp $
 */
#ifndef	__TUIIRFilterPP_h
#define	__TUIIRFilterPP_h

#include "TU/Array++.h"
#include "TU/mmInstructions.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
#if defined(SSE2)
static inline mmFlt
mmForward2(mmFlt in3210, mmFlt c0123, mmFlt& tmp)
{
    tmp = mmShiftElmR<1>(tmp) + c0123 * _mm_shuffle_ps(tmp, in3210,
						       _MM_SHUFFLE(0,0,0,0));
    mmFlt	out0123 = tmp;
    tmp = mmShiftElmR<1>(tmp) + c0123 * _mm_shuffle_ps(tmp, in3210,
						       _MM_SHUFFLE(1,1,0,0));
    out0123 = mmReplaceRMost(mmShiftElmL<1>(out0123), tmp);
    tmp = mmShiftElmR<1>(tmp) + c0123 * _mm_shuffle_ps(tmp, in3210,
						       _MM_SHUFFLE(2,2,0,0));
    out0123 = mmReplaceRMost(mmShiftElmL<1>(out0123), tmp);
    tmp = mmShiftElmR<1>(tmp) + c0123 * _mm_shuffle_ps(tmp, in3210,
						       _MM_SHUFFLE(3,3,0,0));
    return mmReverseElm(mmReplaceRMost(mmShiftElmL<1>(out0123), tmp));
}

static inline mmFlt
mmBackward2(mmFlt in3210, mmFlt c1032, mmFlt& tmp)
{
    mmFlt	out3210 = tmp;
    tmp = mmShiftElmR<1>(tmp) + c1032 * _mm_shuffle_ps(tmp, in3210,
						       _MM_SHUFFLE(3,3,0,0));
    out3210 = mmReplaceRMost(mmShiftElmL<1>(out3210), tmp);
    tmp = mmShiftElmR<1>(tmp) + c1032 * _mm_shuffle_ps(tmp, in3210,
						       _MM_SHUFFLE(2,2,0,0));
    out3210 = mmReplaceRMost(mmShiftElmL<1>(out3210), tmp);
    tmp = mmShiftElmR<1>(tmp) + c1032 * _mm_shuffle_ps(tmp, in3210,
						       _MM_SHUFFLE(1,1,0,0));
    out3210 = mmReplaceRMost(mmShiftElmL<1>(out3210), tmp);
    tmp = mmShiftElmR<1>(tmp) + c1032 * _mm_shuffle_ps(tmp, in3210,
						       _MM_SHUFFLE(0,0,0,0));
    return out3210;
}

template <class S> static void
mmForward2(const S*& src, float*& dst, mmFlt c0123, mmFlt& tmp);

template <> inline void
mmForward2(const u_char*& src, float*& dst, mmFlt c0123, mmFlt& tmp)
{
    mmUInt8	in8 = mmLoadU(src);
    mmStoreU(dst, mmForward2(mmCvt<mmFlt>(in8), c0123, tmp));
    dst += mmFlt::NElms;
    mmStoreU(dst, mmForward2(mmCvt<mmFlt>(mmShiftElmR<mmFlt::NElms>(in8)),
			     c0123, tmp));
    dst += mmFlt::NElms;
    mmStoreU(dst, mmForward2(mmCvt<mmFlt>(mmShiftElmR<2*mmFlt::NElms>(in8)),
			     c0123, tmp));
    dst += mmFlt::NElms;
    mmStoreU(dst, mmForward2(mmCvt<mmFlt>(mmShiftElmR<3*mmFlt::NElms>(in8)),
			     c0123, tmp));
    dst += mmFlt::NElms;
    src += mmUInt8::NElms;
}

template <> inline void
mmForward2(const float*& src, float*& dst, mmFlt c0123, mmFlt& tmp)
{
    mmStoreU(dst, mmForward2(mmLoadU(src), c0123, tmp));
    dst += mmFlt::NElms;
    src += mmFlt::NElms;
}

template <class S> static void
mmBackward2(const S*& src, float*& dst, mmFlt c1032, mmFlt& tmp);

template <> inline void
mmBackward2(const u_char*& src, float*& dst, mmFlt c1032, mmFlt& tmp)
{
    src -= mmUInt8::NElms;
    mmUInt8	in8 = mmLoadU(src);
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward2(mmCvt<mmFlt>(mmShiftElmR<3*mmFlt::NElms>(in8)),
			      c1032, tmp));
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward2(mmCvt<mmFlt>(mmShiftElmR<2*mmFlt::NElms>(in8)),
			      c1032, tmp));
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward2(mmCvt<mmFlt>(mmShiftElmR<mmFlt::NElms>(in8)),
			      c1032, tmp));
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward2(mmCvt<mmFlt>(in8), c1032, tmp));
}

template <> inline void
mmBackward2(const float*& src, float*& dst, mmFlt c1032, mmFlt& tmp)
{
    src -= mmFlt::NElms;
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward2(mmLoadU(src), c1032, tmp));
}

static inline mmFlt
mmForward4(mmFlt in3210, mmFlt c0123, mmFlt c4567, mmFlt& tmp)
{
    tmp = mmShiftElmR<1>(tmp) + c0123 * mmSetAll<0>(in3210)
			      + c4567 * mmSetAll<0>(tmp);
    mmFlt	out0123 = tmp;
    tmp = mmShiftElmR<1>(tmp) + c0123 * mmSetAll<1>(in3210)
			      + c4567 * mmSetAll<0>(tmp);
    out0123 = mmReplaceRMost(mmShiftElmL<1>(out0123), tmp);
    tmp = mmShiftElmR<1>(tmp) + c0123 * mmSetAll<2>(in3210)
			      + c4567 * mmSetAll<0>(tmp);
    out0123 = mmReplaceRMost(mmShiftElmL<1>(out0123), tmp);
    tmp = mmShiftElmR<1>(tmp) + c0123 * mmSetAll<3>(in3210)
			      + c4567 * mmSetAll<0>(tmp);
    return mmReverseElm(mmReplaceRMost(mmShiftElmL<1>(out0123), tmp));
}

static inline mmFlt
mmBackward4(mmFlt in3210, mmFlt c3210, mmFlt c7654, mmFlt& tmp)
{
    mmFlt	out3210 = tmp;
    tmp = mmShiftElmR<1>(tmp) + c3210 * mmSetAll<3>(in3210)
			      + c7654 * mmSetAll<0>(tmp);
    out3210 = mmReplaceRMost(mmShiftElmL<1>(out3210), tmp);
    tmp = mmShiftElmR<1>(tmp) + c3210 * mmSetAll<2>(in3210)
			      + c7654 * mmSetAll<0>(tmp);
    out3210 = mmReplaceRMost(mmShiftElmL<1>(out3210), tmp);
    tmp = mmShiftElmR<1>(tmp) + c3210 * mmSetAll<1>(in3210)
			      + c7654 * mmSetAll<0>(tmp);
    out3210 = mmReplaceRMost(mmShiftElmL<1>(out3210), tmp);
    tmp = mmShiftElmR<1>(tmp) + c3210 * mmSetAll<0>(in3210)
			      + c7654 * mmSetAll<0>(tmp);
    return out3210;
}

template <class S> static void
mmForward4(const S*& src, float*& dst, mmFlt c0123, mmFlt c4567, mmFlt& tmp);

template <> inline void
mmForward4(const u_char*& src, float*& dst,
	   mmFlt c0123, mmFlt c4567, mmFlt& tmp)
{
    mmUInt8	in8 = mmLoadU(src);
    mmStoreU(dst, mmForward4(mmCvt<mmFlt>(in8), c0123, c4567, tmp));
    dst += mmFlt::NElms;
    mmStoreU(dst, mmForward4(mmCvt<mmFlt>(mmShiftElmR<mmFlt::NElms>(in8)),
			     c0123, c4567, tmp));
    dst += mmFlt::NElms;
    mmStoreU(dst, mmForward4(mmCvt<mmFlt>(mmShiftElmR<2*mmFlt::NElms>(in8)),
			     c0123, c4567, tmp));
    dst += mmFlt::NElms;
    mmStoreU(dst, mmForward4(mmCvt<mmFlt>(mmShiftElmR<3*mmFlt::NElms>(in8)),
			     c0123, c4567, tmp));
    dst += mmFlt::NElms;
    src += mmUInt8::NElms;
}

template <> inline void
mmForward4(const float*& src, float*& dst,
	   mmFlt c0123, mmFlt c4567, mmFlt& tmp)
{
    mmStoreU(dst, mmForward4(mmLoadU(src), c0123, c4567, tmp));
    dst += mmFlt::NElms;
    src += mmFlt::NElms;
}

template <class S> static void
mmBackward4(const S*& src, float*& dst, mmFlt c3210, mmFlt c7654, mmFlt& tmp);

template <> inline void
mmBackward4(const u_char*& src, float*& dst,
	    mmFlt c3210, mmFlt c7654, mmFlt& tmp)
{
    src -= mmUInt8::NElms;
    mmUInt8	in8 = mmLoadU(src);
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward4(mmCvt<mmFlt>(mmShiftElmR<3*mmFlt::NElms>(in8)),
			      c3210, c7654, tmp));
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward4(mmCvt<mmFlt>(mmShiftElmR<2*mmFlt::NElms>(in8)),
			      c3210, c7654, tmp));
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward4(mmCvt<mmFlt>(mmShiftElmR<mmFlt::NElms>(in8)),
			      c3210, c7654, tmp));
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward4(mmCvt<mmFlt>(in8), c3210, c7654, tmp));
}

template <> inline void
mmBackward4(const float*& src, float*& dst,
	    mmFlt c3210, mmFlt c7654, mmFlt& tmp)
{
    src -= mmFlt::NElms;
    dst -= mmFlt::NElms;
    mmStoreU(dst, mmBackward4(mmLoadU(src), c3210, c7654, tmp));
}
#endif

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
    const mmFlt	c0123 = mmSet(_c[0], _c[1], _c[2], _c[3]);
    mmFlt	tmp = mmSet(_c[0]*src[1], _c[0]*src[0] + _c[1]*src[1],
			    _c[1]*src[0], 0.0);
    src += 2;			// mmForward2()のために2つ前進．
    for (const float* const tail2 = out.end() - mmNBytes/sizeof(T) - 2;
	 dst <= tail2; )	// srcがdstよりも2つ前進しているのでoverrunに注意．
	mmForward2(src, dst, c0123, tmp);
    mmEmpty();
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
    const mmFlt	c1032 = mmSet(_c[1], _c[0], _c[3], _c[2]);
    mmFlt	tmp   = mmSet(_c[1]*src[-2], _c[1]*src[-1] + _c[0]*src[-2],
			      _c[0]*src[-1], 0.0);
    src -= 2;			// mmBackward2()のために2つ後退．
    for (const float* const head2 = out.begin() + mmNBytes/sizeof(T) + 2;
	 dst >= head2; )	// srcがdstよりも2つ後退しているのでoverrunに注意．
	mmBackward2(src, dst, c1032, tmp);
    mmEmpty();
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
    const mmFlt	c0123 = mmSet(_c[0], _c[1], _c[2], _c[3]),
		c4567 = mmSet(_c[4], _c[5], _c[6], _c[7]);
    mmFlt	tmp   = mmZero<mmFlt>();
    for (const float* const tail2 = out.end() - mmNBytes/sizeof(T);
	 dst <= tail2; )
	mmForward4(src, dst, c0123, c4567, tmp);
    mmEmpty();
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
    const mmFlt	c3210 = mmSet(_c[3], _c[2], _c[1], _c[0]),
		c7654 = mmSet(_c[7], _c[6], _c[5], _c[4]);
    mmFlt	tmp   = mmZero<mmFlt>();
    for (const float* const head2 = out.begin() + mmNBytes/sizeof(T);
	 dst >= head2; )
	mmBackward4(src, dst, c3210, c7654, tmp);
    mmEmpty();
#else
    --src;
    --dst;
    *dst = 0.0;
    --src;
    --dst;
    *dst = _c[0]*src[1];
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
  \param limit1F	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
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
  \param limit1B	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
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
*  class BilateralIIRFilterBase						*
************************************************************************/
//! 両側Infinite Inpulse Response Filterのベースとなるクラス
class BilateralIIRFilterBase
{
  public:
  //! 微分の階数
    enum Order
    {
	Zeroth,						//!< 0階微分
	First,						//!< 1階微分
	Second						//!< 2階微分
    };
};

/************************************************************************
*  class BilateralIIRFilter<D>						*
************************************************************************/
//! 両側Infinite Inpulse Response Filterを表すクラス
template <u_int D> class BilateralIIRFilter : public BilateralIIRFilterBase
{
  public:
    typedef IIRFilter<D>	IIR;
    
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
			    int is=0, int ie=0)			const	;
    u_int	dim()						const	;
    float	operator [](int i)				const	;
    void	limits(float& limit0,
		       float& limit1, float& limit2)		const	;
    
  private:
    IIR				_iirF;
    mutable Array<float>	_bufF;
    IIR				_iirB;
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
#ifdef DEBUG
    float	limit0, limit1, limit2;
    limits(limit0, limit1, limit2);
    std::cerr << "limit0 = " << limit0 << ", limit1 = " << limit1
	      << ", limit2 = " << limit2 << std::endl;
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
  \param order	フィルタの微分階数. #Zerothまたは#Secondならば対称フィルタ
		として，#Firstならば反対称フィルタとして自動的に後退方向の
		z変換係数を計算する．#Zeroth, #First, #Secondのときに，それ
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
    
//! フィルタによる畳み込みを行う. 出力は operator [](int) で取り出す
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
				   int is, int ie) const
{
    out.resize(in.ncol(), in.nrow());
    if (ie == 0)
	ie = in.nrow();

    for (int i = is; i < ie; ++i)
    {
	convolve(in[i]);
	T2*	col = &out[0];
	for (int j = 0; j < dim(); ++j)
	    (*col++)[i] = (*this)[j];
    }

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

/************************************************************************
*  class BilateralIIRFilter2						*
************************************************************************/
//! 2次元両側Infinite Inpulse Response Filterを表すクラス
template <class BIIRH, class BIIRV=BIIRH> class BilateralIIRFilter2
{
  public:
    typedef typename BIIRH::Order	Order;
    
    BilateralIIRFilter2()	:_biirH(), _biirV()			{}
    BilateralIIRFilter2(u_int nthreads)
    	:_biirH(nthreads), _biirV(nthreads)				{}

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
    BIIRH				_biirH;
    BIIRV				_biirV;
    mutable Array2<Array<float> >	_buf;
};
    
//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数
  \param cHB	横方向後退z変換係数
  \param cHV	縦方向前進z変換係数
  \param cHV	縦方向後退z変換係数
  \return	このフィルタ自身
*/
template <class BIIRH, class BIIRV> inline BilateralIIRFilter2<BIIRH, BIIRV>&
BilateralIIRFilter2<BIIRH, BIIRV>::initialize(float cHF[], float cHB[],
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
  \param cHV	縦方向前進z変換係数
  \param orderV	縦方向微分階数
  \return	このフィルタ自身
*/
template <class BIIRH, class BIIRV> inline BilateralIIRFilter2<BIIRH, BIIRV>&
BilateralIIRFilter2<BIIRH, BIIRV>::initialize(float cHF[], Order orderH,
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
template <class BIIRH, class BIIRV>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline const BilateralIIRFilter2<BIIRH, BIIRV>&
BilateralIIRFilter2<BIIRH, BIIRV>::convolve(const Array2<T1, B1, R1>& in,
					    Array2<T2, B2, R2>& out) const
{
    _biirH(in, _buf);
    _biirV(_buf, out);

    return *this;
}

}

#endif	/* !__TUIIRFilterPP_h */
