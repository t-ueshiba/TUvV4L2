/*
 *  $Id: EdgeDetector.cc,v 1.1 2006-11-07 01:15:06 ueshiba Exp $
 */
#include "TU/Image++.h"
#ifdef __INTEL_COMPILER
#  define SSE2
#  include "TU/mmInstructions.h"
#endif

namespace TU
{
static const float	slant = 0.414214;	// tan(M_PI/8)
    
/************************************************************************
*  static functions							*
************************************************************************/
#ifdef SSE2
static inline mmInt	mmDir4(mmFlt eH, mmFlt eV)
			{
			    mmInt  l0 = mmCastToInt(mmCmpLEF(eH, eV)),
				   l1 = mmCastToInt(
					  mmCmpLEF(eH, mmSubF(mmZeroF(), eV)));
			    return mmOr(mmAnd(mmXor(l0, l1), mmSet32(0x2)),
					mmAnd(l1, mmSet32(0x4)));
			}
static inline mmInt	mmDir8(mmFlt eH, mmFlt eV)
			{
			    mmFlt  sH = mmMulF(mmSetF(slant), eH),
				   sV = mmMulF(mmSetF(slant), eV);
			    mmInt  l0 = mmCastToInt(mmCmpLEF(sH, eV)),
				   l1 = mmCastToInt(mmCmpLEF(eH, sV)),
				   l2 = mmCastToInt(
					  mmCmpLEF(eH, mmSubF(mmZeroF(), sV))),
				   l3 = mmCastToInt(
					  mmCmpLEF(sH, mmSubF(mmZeroF(), eV)));
			    return mmOr(mmOr(mmAnd(mmOr(mmXor(l0, l1),
							mmXor(l2, l3)),
						   mmSet32(0x1)),
					     mmAnd(mmXor(l1, l3),
						   mmSet32(0x2))),
					mmAnd(l3, mmSet32(0x4)));
			}
#endif

static inline bool
isLink(const Image<u_char>& edge, const Point2<int>& p, int dir)
{
    return (edge(p.neighbor(dir)) &&
	    (!(dir & 0x1) ||
	     (!edge(p.neighbor(dir-1)) && !edge(p.neighbor(dir+1)))));
}
    
static void
trace(Image<u_char>& edge, const Point2<int>& p)
{
    u_char&	e = edge(p);
    
    if (e & EdgeDetector::TRACED)
	return;

    e |= (EdgeDetector::TRACED | EdgeDetector::EDGE);
    for (int dir = 0; dir < 8; ++dir)
	if (isLink(edge, p, dir))
	    trace(edge, p.neighbor(dir));
}

static bool
canInterpolate(const Image<u_char>& edge, const Point2<int>& p)
{
    int	nedges = 0, nweaks = 0;
    
    for (int dir = 0; dir < 8; ++dir)
    {
	u_char	e = edge(p.neighbor(dir));
	
	if (e & EdgeDetector::EDGE)
	    ++nedges;
	else if (e & EdgeDetector::WEAK)
	    ++nweaks;
    }

    return (nedges != 0 && nweaks == 1);
}
    
/************************************************************************
*  class EdgeDetector							*
************************************************************************/
//! エッジ強度を求める
/*!
  \param edgeH	横方向1次微分入力画像.
  \param edgeV	縦方向1次微分入力画像.
  \param out	エッジ強度出力画像.
  \return	このエッジ検出器自身.
*/
const EdgeDetector&
EdgeDetector::strength(const Image<float>& edgeH,
		       const Image<float>& edgeV, Image<float>& out) const
{
    out.resize(edgeH.height(), edgeH.width());
    for (int v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v], *eV = edgeV[v];
	float*			dst = out[v];
	const float* const	end = dst + out.width();
#ifdef SSE
	for (const float* const end2 = dst + 4*(out.width()/4); dst < end2; )
	{
	    const mmFlt	fH = mmLoadFU(eH), fV = mmLoadFU(eV);
	    
	    mmStoreFU(dst, mmSqrtF(mmAddF(mmMulF(fH, fH), mmMulF(fV, fV))));
	    eH  += 4;
	    eV  += 4;
	    dst += 4;
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
  \param edgeH	横方向1次微分入力画像.
  \param edgeV	縦方向1次微分入力画像.
  \param out	エッジ方向出力画像.
  \return	このエッジ検出器自身.
*/
const EdgeDetector&
EdgeDetector::direction4(const Image<float>& edgeH,
			 const Image<float>& edgeV, Image<u_char>& out) const
{
    out.resize(edgeH.height(), edgeH.width());
    for (int v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v], *eV = edgeV[v];
	u_char*			dst = out[v];
	const u_char* const	end = dst + out.width();
#ifdef SSE2
	for (const u_char* const end2 = dst + mmNBytes*(out.width()/mmNBytes);
	     dst < end2; dst += mmNBytes)
	{
	    const mmInt	d0 = mmDir4(mmLoadFU(eH), mmLoadFU(eV));
	    eH  += 4;
	    eV  += 4;
	    const mmInt	d1 = mmDir4(mmLoadFU(eH), mmLoadFU(eV));
	    eH  += 4;
	    eV  += 4;
	    const mmInt	d2 = mmDir4(mmLoadFU(eH), mmLoadFU(eV));
	    eH  += 4;
	    eV  += 4;
	    const mmInt	d3 = mmDir4(mmLoadFU(eH), mmLoadFU(eV));
	    eH  += 4;
	    eV  += 4;
	    mmStoreU((mmInt*)dst, mmPack(mmPack32(d0, d1), mmPack32(d2, d3)));
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
  \param edgeH	横方向1次微分入力画像.
  \param edgeV	縦方向1次微分入力画像.
  \param out	エッジ方向出力画像.
  \return	このエッジ検出器自身.
*/
const EdgeDetector&
EdgeDetector::direction8(const Image<float>& edgeH,
			 const Image<float>& edgeV, Image<u_char>& out) const
{
    out.resize(edgeH.height(), edgeH.width());
    for (int v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v], *eV = edgeV[v];
	u_char*			dst = out[v];
	const u_char* const	end = dst + out.width();
#ifdef SSE2
	for (const u_char* const end2 = dst + mmNBytes*(out.width()/mmNBytes);
	     dst < end2; dst += mmNBytes)
	{
	    const mmInt	d0 = mmDir8(mmLoadFU(eH), mmLoadFU(eV));
	    eH  += 4;
	    eV  += 4;
	    const mmInt	d1 = mmDir8(mmLoadFU(eH), mmLoadFU(eV));
	    eH  += 4;
	    eV  += 4;
	    const mmInt	d2 = mmDir8(mmLoadFU(eH), mmLoadFU(eV));
	    eH  += 4;
	    eV  += 4;
	    const mmInt	d3 = mmDir8(mmLoadFU(eH), mmLoadFU(eV));
	    eH  += 4;
	    eV  += 4;
	    mmStoreU((mmInt*)dst, mmPack(mmPack32(d0, d1), mmPack32(d2, d3)));
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
  \param strength	エッジ強度入力画像.
  \param direction	エッジ方向入力画像.
  \param out		細線化されたエッジ画像.
  \return		このエッジ検出器自身.
*/
const EdgeDetector&
EdgeDetector::suppressNonmaxima(const Image<float>& strength,
				const Image<u_char>& direction,
				Image<u_char>& out) const
{
    out.resize(strength.height(), strength.width());

    for (int u = 0; u < out.width(); ++u)
	out[0][u] = out[out.height()-1][u] = 0;
    for (int v = 0; v < out.height(); ++v)
	out[v][0] = out[v][out.width()-1] = 0;
    
    for (int v = 0; ++v < out.height() - 1; )
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
	    
	    if (*str >= _th_low)
		switch (*dir)
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
		*dst = 0;
	}
    }

    for (int v = 0; ++v < out.height() - 1; )
	for (int u = 0; ++u < out.width() - 1; )
	    if (out[v][u] & EDGE)
		trace(out, Point2<int>(u, v));
    
    for (int v = 0; ++v < out.height() - 1; )
	for (int u = 0; ++u < out.width() - 1; )
	{
	    Point2<int>	p(u, v);
	    
	    if (!(out(p) & EDGE) && canInterpolate(out, p))
		trace(out, p);
	}
    
    for (int v = 0; v < out.height(); )
    {
	u_char*	dst = out[v++];
	for (const u_char* const end = dst + out.width(); dst < end; ++dst)
	    *dst = (*dst & EDGE ? 255 : 0);
    }
    
    return *this;
}

}
