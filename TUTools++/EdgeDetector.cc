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
 *  $Id: EdgeDetector.cc,v 1.15 2009-09-09 07:06:30 ueshiba Exp $
 */
#include "TU/EdgeDetector.h"
#include "TU/mmInstructions.h"

namespace TU
{
static const float	slant = 0.414214;	// tan(M_PI/8)
    
/************************************************************************
*  static functions							*
************************************************************************/
#if defined(SSE2)
static inline mmInt32
mmDir4(mmFlt eH, mmFlt eV)
{
    mmInt32	l0 = mmCast<mmInt32>(eH < eV),
		l1 = mmCast<mmInt32>(eH < -eV);
    return ((l0 ^ l1) & mmSetAll<mmInt32>(0x2)) |
	   (l1 & mmSetAll<mmInt32>(0x4));
}

static inline mmInt32
mmDir8(mmFlt eH, mmFlt eV)
{
    mmFlt	sH = mmSetAll<mmFlt>(slant) * eH,
		sV = mmSetAll<mmFlt>(slant) * eV;
    mmInt32	l0 = mmCast<mmInt32>(sH < eV),
		l1 = mmCast<mmInt32>(eH < sV),
		l2 = mmCast<mmInt32>(eH < -sV),
		l3 = mmCast<mmInt32>(sH < -eV);
    return (((l0 ^ l1) | (l2 ^ l3)) & mmSetAll<mmInt32>(0x1)) |
	   ((l1 ^ l3) & mmSetAll<mmInt32>(0x2)) |
	   (l3 & mmSetAll<mmInt32>(0x4));
}
#endif

//! あるエッジ点と指定された方向の近傍点が接続しているか調べる
/*!
  \param edge	エッジ画像
  \param p	エッジ点
  \param dir	近傍点の方向
  \return	接続していればtrue，そうでなければfalse
*/
static inline bool
isLink(const Image<u_char>& edge, const Point2i& p, int dir)
{
  // (1) 近傍点が少なくとも強/弱エッジ点であり，かつ，(2a) 4近傍点であるか，
  // (2b) 両隣の近傍点が強/弱エッジ点でない場合に接続していると判定する．
    return (edge(p.neighbor(dir)) &&
	    (!(dir & 0x1) ||
	     (!edge(p.neighbor(dir-1)) && !edge(p.neighbor(dir+1)))));
}
    
//! あるエッジ点を起点にして，接続するエッジ点を追跡する
/*!
  \param edge	エッジ画像
  \param p	エッジ点
*/
static void
trace(Image<u_char>& edge, const Point2i& p)
{
    u_char&	e = edge(p);		// この点pの画素値
    
    if (e & EdgeDetector::TRACED)	// 既にこの点が訪問済みならば，
	return;				// 直ちに戻る．

    e |= (EdgeDetector::TRACED | EdgeDetector::EDGE);	// 訪問済みかつエッジ点
    for (int dir = 0; dir < 8; ++dir)	// pの8つの近傍点それぞれについて
	if (isLink(edge, p, dir))	// pと接続していれば
	    trace(edge, p.neighbor(dir));	// さらに追跡を続ける．
}

//! ある点を打てばEDGEラベルが付いている点とそうでない点を結べるか調べる
/*!
  \param edge	エッジ画像
  \param p	打とうとする点
  \return	結べるのであればtrue，そうでなければfalse
*/
static bool
canInterpolate(const Image<u_char>& edge, const Point2i& p)
{
    int	nedges = 0, nweaks = 0;
    
    for (int dir = 0; dir < 8; ++dir)	// pの8つの近傍点それぞれについて
    {
	u_char	e = edge(p.neighbor(dir));
	
	if (e & EdgeDetector::EDGE)
	    ++nedges;			// EDGEラベルが付いている点
	else if (e & EdgeDetector::WEAK)
	    ++nweaks;			// EDGEラベルが付いていない弱いエッジ点
    }

  // pの近傍に，既にEDGEラベルが付いている点が少なくとも1つ，および
  // 付いていない弱いエッジ点が1つだけあれば，trueを返す．
    return (nedges != 0 && nweaks == 1);
}
    
/************************************************************************
*  class EdgeDetector							*
************************************************************************/
//! エッジ強度を求める
/*!
  \param edgeH	横方向1階微分入力画像
  \param edgeV	縦方向1階微分入力画像
  \param out	エッジ強度出力画像
  \return	このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::strength(const Image<float>& edgeH,
		       const Image<float>& edgeV, Image<float>& out) const
{
    out.resize(edgeH.height(), edgeH.width());
    for (u_int v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v], *eV = edgeV[v];
	float*			dst = out[v];
	const float* const	end = dst + out.width();
#if defined(SSE)
	for (const float* const end2 = dst + 4*(out.width()/4); dst < end2; )
	{
	    const mmFlt	fH = mmLoadU(eH), fV = mmLoadU(eV);
	    
	    mmStoreU(dst, mmSqrt(fH * fH + fV * fV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    dst += mmFlt::NElms;
	}
#endif
	while (dst < end)
	{
	    *dst++ = sqrtf(*eH * *eH + *eV * *eV);
	    ++eH;
	    ++eV;
	}
    }

    return *this;
}
    
//! 4近傍によるエッジ方向を求める
/*!
  \param edgeH	横方向1階微分入力画像
  \param edgeV	縦方向1階微分入力画像
  \param out	エッジ方向出力画像
  \return	このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::direction4(const Image<float>& edgeH,
			 const Image<float>& edgeV, Image<u_char>& out) const
{
    out.resize(edgeH.height(), edgeH.width());
    for (u_int v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v], *eV = edgeV[v];
	u_char*			dst = out[v];
	const u_char* const	end = dst + out.width();
#if defined(SSE2)
	for (const u_char* const end2 = dst + mmUInt8::floor(out.width());
	     dst < end2; dst += mmUInt8::NElms)
	{
	    const mmInt32	d0 = mmDir4(mmLoadU(eH), mmLoadU(eV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    const mmInt32	d1 = mmDir4(mmLoadU(eH), mmLoadU(eV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    const mmInt32	d2 = mmDir4(mmLoadU(eH), mmLoadU(eV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    const mmInt32	d3 = mmDir4(mmLoadU(eH), mmLoadU(eV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    mmStoreU(dst, mmCvt<mmUInt8>(mmCvt<mmInt16>(d0, d1),
					 mmCvt<mmInt16>(d2, d3)));
	}
#endif
	while (dst < end)
	{
	    *dst++ = (*eH <= *eV ? (*eH <= -*eV ? 4 : 2)
				 : (*eH <= -*eV ? 6 : 0));
	    ++eH;
	    ++eV;
	}
    }
    
    return *this;
}
    
//! 8近傍によるエッジ方向を求める
/*!
  \param edgeH	横方向1階微分入力画像
  \param edgeV	縦方向1階微分入力画像
  \param out	エッジ方向出力画像
  \return	このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::direction8(const Image<float>& edgeH,
			 const Image<float>& edgeV, Image<u_char>& out) const
{
    out.resize(edgeH.height(), edgeH.width());
    for (u_int v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v], *eV = edgeV[v];
	u_char*			dst = out[v];
	const u_char* const	end = dst + out.width();
#if defined(SSE2)
	for (const u_char* const end2 = dst + mmUInt8::floor(out.width());
	     dst < end2; dst += mmUInt8::NElms)
	{
	    const mmInt32	d0 = mmDir8(mmLoadU(eH), mmLoadU(eV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    const mmInt32	d1 = mmDir8(mmLoadU(eH), mmLoadU(eV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    const mmInt32	d2 = mmDir8(mmLoadU(eH), mmLoadU(eV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    const mmInt32	d3 = mmDir8(mmLoadU(eH), mmLoadU(eV));
	    eH  += mmFlt::NElms;
	    eV  += mmFlt::NElms;
	    mmStoreU(dst, mmCvt<mmUInt8>(mmCvt<mmInt16>(d0, d1),
					 mmCvt<mmInt16>(d2, d3)));
	}
#endif
	while (dst < end)
	{
	    const float	sH = slant * *eH, sV = slant * *eV;
	    
	    *dst++ = (sH <= *eV ?
		      (*eH <= sV ?
		       (*eH <= -sV ?
			(sH <= -*eV ? 4 : 3) : 2) : 1) :
		      (sH <= -*eV ?
		       (*eH <= -sV ?
			(*eH <=  sV ? 5 : 6) : 7) : 0));
	    ++eH;
	    ++eV;
	}
    }
    
    return *this;
}
    
//! 非極大値抑制処理により細線化を行う
/*!
  \param strength	エッジ強度入力画像
  \param direction	エッジ方向入力画像
  \param out		強いエッジ点と弱いエッジ点にそれぞれ#EDGEラベルと
			#WEAKラベルを付けた出力画像
  \return		このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::suppressNonmaxima(const Image<float>& strength,
				const Image<u_char>& direction,
				Image<u_char>& out) const
{
    out.resize(strength.height(), strength.width());

  // 出力画像の外周を0にする．
    if (out.height() > 0)
	for (u_int u = 0; u < out.width(); ++u)
	    out[0][u] = out[out.height()-1][u] = 0;
    if (out.width() > 0)
	for (u_int v = 0; v < out.height(); ++v)
	    out[v][0] = out[v][out.width()-1] = 0;

  // 各点のエッジ強度が (1) その点のエッジ方向に沿った両隣と比較して極大に
  // なっており，かつ，(2a) 強い閾値以上ならばEDGEラベルを，(2b) 弱い閾値
  // 以上ならばWEAKラベルをそれぞれ書き込む．そうでなければ0を書き込む．
    for (u_int v = 0; ++v < out.height() - 1; )
    {
	const float		*prv = strength[v-1],
				*str = strength[v],
				*nxt = strength[v+1];
	const u_char		*dir = direction[v];
	const u_char* const	end  = &out[v][out.width() - 1];
	for (u_char* dst = out[v]; ++dst < end; )
	{
	    ++prv;
	    ++str;
	    ++nxt;
	    ++dir;
	    
	    if (*str >= _th_low)	// 弱い閾値以上なら
		switch (*dir)		// エッジ方向を見る．
		{
		  case 0:
		  case 4:
		    *dst = (*str > *(str-1) && *str > *(str+1) ?
			    (*str >= _th_high ? EDGE : WEAK) : 0);
		    break;
		  case 1:
		  case 5:
		    *dst = (*str > *(prv-1) && *str > *(nxt+1) ?
			    (*str >= _th_high ? EDGE : WEAK) : 0);
		    break;
		  case 2:
		  case 6:
		    *dst = (*str > *prv && *str > *nxt ?
			    (*str >= _th_high ? EDGE : WEAK) : 0);
		    break;
		  default:
		    *dst = (*str > *(prv+1) && *str > *(nxt-1) ?
			    (*str >= _th_high ? EDGE : WEAK) : 0);
		    break;
		}
	    else
		*dst = 0;		// 弱い閾値未満なら 0
	}
    }

    return *this;
}

//! 2次微分画像のゼロ交差点を検出する
/*!
  \param in		入力2次微分画像
  \param out		ゼロ交差点を255，そうでない点を0とした出力画像
  \return		このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::zeroCrossing(const Image<float>& in, Image<u_char>& out) const
{
    out.resize(in.height(), in.width());

  // 出力画像の下端と右端を0にする．
    if (out.height() > 0)
	for (u_int u = 0; u < out.width(); ++u)
	    out[out.height()-1][u] = 0;
    if (out.width() > 0)
	for (u_int v = 0; v < out.height(); ++v)
	    out[v][out.width()-1] = 0;

  // 現在点を左上隅とする2x2ウィンドウ中の画素が異符号ならエッジ点とする．
    for (u_int v = 0; v < out.height() - 1; ++v)
    {
	const float		*cur = in[v],
				*nxt = in[v+1];
	const u_char* const	end  = &out[v][out.width() - 1];
	for (u_char* dst = out[v]; dst < end; )
	{
	    if ((*cur >= 0.0 && *(cur+1) >= 0.0 &&
		 *nxt >= 0.0 && *(nxt+1) >= 0.0) ||
		(*cur <= 0.0 && *(cur+1) <= 0.0 &&
		 *nxt <= 0.0 && *(nxt+1) <= 0.0))
		*dst++ = 0;
	    else
		*dst++ = 255;
	    ++cur;
	    ++nxt;
	}
    }

    return *this;
}
    
//! 2次微分画像のゼロ交差点を検出し，エッジ強度によって分類する
/*!
  \param in		入力2次微分画像
  \param strength	入力エッジ強度画像
  \param out		強いエッジ点と弱いエッジ点にそれぞれ#EDGEラベルと
			#WEAKラベルを付けた出力画像
  \return		このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::zeroCrossing(const Image<float>& in, const Image<float>& strength,
			   Image<u_char>& out) const
{
    out.resize(in.height(), in.width());

  // 出力画像の外周を0にする．
    if (out.height() > 0)
	for (u_int u = 0; u < out.width(); ++u)
	    out[0][u] = out[out.height()-1][u] = 0;
    if (out.width() > 0)
	for (u_int v = 0; v < out.height(); ++v)
	    out[v][0] = out[v][out.width()-1] = 0;

  // 現在点を左上隅とする2x2ウィンドウ中の画素が異符号ならエッジ点とする．
    for (u_int v = 0; ++v < out.height() - 1; )
    {
	const float		*cur = in[v],
				*nxt = in[v+1],
				*str = strength[v];
	const u_char* const	end  = &out[v][out.width() - 1];
	for (u_char* dst = out[v]; ++dst < end; )
	{
	    ++cur;
	    ++nxt;
	    ++str;

	    if ((*str < _th_low) ||
		(*cur >= 0.0 && *(cur+1) >= 0.0 &&
		 *nxt >= 0.0 && *(nxt+1) >= 0.0) ||
		(*cur <= 0.0 && *(cur+1) <= 0.0 &&
		 *nxt <= 0.0 && *(nxt+1) <= 0.0))
		*dst = 0;
	    else
		*dst = (*str >= _th_high ? EDGE : WEAK);
	}
    }

    return *this;
}

//! 強いエッジ点を起点に弱いエッジを追跡することによりヒステリシス閾値処理を行う
/*!
  \param edge		強いエッジ点と弱いエッジ点にそれぞれ#EDGEラベルと
			#WEAKラベルを付けた画像．処理が終わると最終的なエッジ
			点に255を，そうでない点には0を書き込んで返される．
  \return		このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::hysteresisThresholding(Image<u_char>& edge) const
{
  // 強いエッジ点を起点にして，接続する弱いエッジ点を追跡しEDGEラベルを付ける．
    for (u_int v = 0; ++v < edge.height() - 1; )
	for (u_int u = 0; ++u < edge.width() - 1; )
	    if (edge[v][u] & EDGE)
		trace(edge, Point2i(u, v));

  // EDGEラベルが付いておらず，かつ付いている点と付いていない弱いエッジ点の
  // 橋渡しになれる点に新たにEDGEラベルを付けて追跡を行う．
    for (u_int v = 0; ++v < edge.height() - 1; )
	for (u_int u = 0; ++u < edge.width() - 1; )
	{
	    Point2i	p(u, v);
	    
	    if (!(edge(p) & EDGE) && canInterpolate(edge, p))
		trace(edge, p);
	}

  // EDGE点には255を，そうでない点には0を書き込む．
    for (u_int v = 0; v < edge.height(); )
    {
	u_char*	dst = edge[v++];
	for (const u_char* const end = dst + edge.width(); dst < end; ++dst)
	    *dst = (*dst & EDGE ? 255 : 0);
    }
 
    return *this;
}
    
}
