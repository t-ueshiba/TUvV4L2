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
#include "TU/EdgeDetector.h"
#include "TU/simd/simd.h"

/************************************************************************
*  static functions							*
************************************************************************/
namespace TU
{
static const float	slant = 0.41421356;	// tan(M_PI/8)

#if defined(SIMD)
namespace simd
{
static inline Is32vec
dir4(F32vec u, F32vec v)
{
    const Is32vec	l4 = cast<int>(u < -v);
    return (l4 & Is32vec(0x4)) | ((cast<int>(u < v) ^ l4) & Is32vec(0x2));
}

static inline Is32vec
dir4x(F32vec u, F32vec v)
{
    const Is32vec	l4 = cast<int>(v < zero<float>());
    return (l4				        & Is32vec(0x4)) |
	   ((cast<int>(u < zero<float>()) ^ l4) & Is32vec(0x2)) |
						  Is32vec(0x1);
}

static inline Is32vec
dir4x(F32vec u, F32vec v, F32vec lambda)
{
    const Is32vec	l = cast<int>((u < zero<float>()) ^
				      (v < zero<float>()));
    return ((l ^ cast<int>(lambda < zero<float>())) & Is32vec(0x4)) |
	   (l					    & Is32vec(0x2)) |
						      Is32vec(0x1);
}

static inline Is32vec
dir8(F32vec u, F32vec v)
{
    const F32vec	su = F32vec(slant) * u,
			sv = F32vec(slant) * v;
    const Is32vec	l2 = cast<int>( u <  sv),
			l4 = cast<int>(su <  -v);
    return (l4				  & Is32vec(0x4)) |
	   ((l2 ^ l4)			  & Is32vec(0x2)) |
	   (((cast<int>(su <   v) ^ l2) |
	     (cast<int>( u < -sv) ^ l4))  & Is32vec(0x1));
}

static inline Is32vec
dir8x(F32vec u, F32vec v)
{
    const Is32vec	l2 = cast<int>(u < zero<float>()),
			l4 = cast<int>(v < zero<float>()),
			l  = l2 ^ l4;
    return (l4				& Is32vec(0x4)) |
	   (l				& Is32vec(0x2)) |
	   (((cast<int>(u <  v) ^ l2) |
	     (cast<int>(u < -v) ^ l4))  & Is32vec(0x1));
}

static inline Is32vec
dir8x(F32vec u, F32vec v, F32vec lambda)
{
    const Is32vec	l2 = cast<int>(u < zero<float>()),
			l4 = cast<int>(v < zero<float>()),
			l  = l2 ^ l4;
    return ((cast<int>(lambda < zero<float>()) ^ l) & Is32vec(0x4)) |
	   (l					    & Is32vec(0x2)) |
	   (((cast<int>(u <  v) ^ l2) |
	     (cast<int>(u < -v) ^ l4))		    & Is32vec(0x1));
}

template <class T> static inline vec<T>
eigen(vec<T> a, vec<T> b, vec<T> c)
{
    const vec<T>	avrg = avg(a, c), diff = sub_avg(a, c),
			frac = sqrt(diff*diff + b*b);
    return select(avrg > zero<T>(), avrg + frac, avrg - frac);
}

}	// namespace simd
#endif

template <class T> static inline u_int
dir4(T u, T v)
{
    return (u < -v ? (u < v ? 4 : 6) : (u < v ? 2 : 0));
}
    
template <class T> static inline u_int
dir4x(T u, T v)
{
    return (v < 0 ? (u < 0 ? 5 : 7) : (u < 0 ? 3 : 1));
}
    
template <class T> static inline u_int
dir4x(T u, T v, T lambda)
{
    const u_int	l = (u < 0) ^ (v < 0);
    return (((lambda < 0) ^ l) << 2) | (l << 1) | 0x1;
}
    
template <class T> static inline u_int
dir8(T u, T v)
{
    const T	su = slant * u, sv = slant * v;
    return (su < -v ?
	    ( u < -sv ? (u < sv ? (su <   v ? 4 : 5) : 6) : 7) :
	    (su <   v ? (u < sv ? ( u < -sv ? 3 : 2) : 1) : 0));
}
    
template <class T> static inline u_int
dir8x(T u, T v)
{
    return (v < 0 ?
	    (u < -v ? (u < 0 ? (u <  v ? 4 : 5) : 6) : 7) :
	    (u <  v ? (u < 0 ? (u < -v ? 3 : 2) : 1) : 0));
}
    
template <class T> static inline u_int
dir8x(T u, T v, T lambda)
{
    const u_int	l2 = (u < 0), l4 = (v < 0), l = l2 ^ l4;
    return (((lambda < 0) ^ l) << 2) |
	   (l		       << 1) |
	   ((u < v) ^ l2) | ((u < -v) ^ l4);
}
    
template <class T> static inline T
eigen(T a, T b, T c)
{
    const T	avrg = T(0.5)*(a + c), diff = T(0.5)*(a - c),
		frac = std::sqrt(diff*diff + b*b);
    return (avrg > T(0) ? avrg + frac : avrg - frac);
}
    
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
#ifdef _DEBUG
    static int	depth = 0;
    std::cerr << "depth: " << depth << std::endl;
#endif
    u_char&	e = edge(p);		// この点pの画素値
    
    if (e & EdgeDetector::TRACED)	// 既にこの点が訪問済みならば，
	return;				// 直ちに戻る．

    e |= (EdgeDetector::TRACED | EdgeDetector::EDGE);	// 訪問済みかつエッジ点
    for (int dir = 0; dir < 8; ++dir)	// pの8つの近傍点それぞれについて
	if (isLink(edge, p, dir))	// pと接続していれば
	{
#ifdef _DEBUG
	    ++depth;
#endif
	    trace(edge, p.neighbor(dir));	// さらに追跡を続ける．
#ifdef _DEBUG
	    --depth;
#endif
	}
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
    for (size_t v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v].data(), *eV = edgeV[v].data();
	float*			dst = out[v].data();
	const float* const	end = dst + out.width();
#if defined(SSE)
	using namespace		simd;
	
	const size_t		nelms = F32vec::size;
	for (const float* const end2 = dst + F32vec::floor(out.width());
	     dst < end2; )
	{
	    const F32vec	fH = load<false>(eH), fV = load<false>(eV);
	    
	    store<false>(dst, sqrt(fH * fH + fV * fV));
	    eH  += nelms;
	    eV  += nelms;
	    dst += nelms;
	}
#endif
	while (dst < end)
	{
	    *dst++ = std::sqrt(*eH * *eH + *eV * *eV);
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
    for (size_t v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v].data(), *eV = edgeV[v].data();
	u_char*			dst = out[v].data();
	const u_char* const	end = dst + out.width();
#if defined(SIMD)
	using namespace		simd;

	const size_t		nelms = F32vec::size;
	for (const u_char* const end2 = dst + Iu8vec::floor(out.width());
	     dst < end2; dst += Iu8vec::size)
	{
	    const Is32vec	d0 = dir4(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d1 = dir4(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d2 = dir4(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d3 = dir4(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    store<false>(dst,
			 cvt<u_char>(cvt<short>(d0, d1), cvt<short>(d2, d3)));
	}
#endif
	while (dst < end)
	    *dst++ = dir4(*eH++, *eV++);
    }
    
    return *this;
}
    
const EdgeDetector&
EdgeDetector::direction4x(const Image<float>& edgeH,
			  const Image<float>& edgeV, Image<u_char>& out) const
{
    out.resize(edgeH.height(), edgeH.width());
    for (size_t v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v].data(), *eV = edgeV[v].data();
	u_char*			dst = out[v].data();
	const u_char* const	end = dst + out.width();
#if defined(SIMD)
	using namespace		simd;

	const size_t		nelms = F32vec::size;
	for (const u_char* const end2 = dst + Iu8vec::floor(out.width());
	     dst < end2; dst += Iu8vec::size)
	{
	    const Is32vec	d0 = dir4x(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d1 = dir4x(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d2 = dir4x(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d3 = dir4x(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    store<false>(dst,
			 cvt<u_char>(cvt<short>(d0, d1), cvt<short>(d2, d3)));
	}
#endif
	while (dst < end)
	    *dst++ = dir4x(*eH++, *eV++);
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
    for (size_t v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v].data(), *eV = edgeV[v].data();
	u_char*			dst = out[v].data();
	const u_char* const	end = dst + out.width();
#if defined(SIMD)
	using namespace		simd;

	const size_t		nelms = F32vec::size;
	for (const u_char* const end2 = dst + Iu8vec::floor(out.width());
	     dst < end2; dst += Iu8vec::size)
	{
	    const Is32vec	d0 = dir8(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d1 = dir8(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d2 = dir8(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d3 = dir8(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    store<false>(dst,
			 cvt<u_char>(cvt<short>(d0, d1), cvt<short>(d2, d3)));
	}
#endif
	while (dst < end)
	    *dst++ = dir8(*eH++, *eV++);
    }
    
    return *this;
}
    
const EdgeDetector&
EdgeDetector::direction8x(const Image<float>& edgeH,
			  const Image<float>& edgeV, Image<u_char>& out) const
{
    out.resize(edgeH.height(), edgeH.width());
    for (size_t v = 0; v < out.height(); ++v)
    {
	const float		*eH = edgeH[v].data(), *eV = edgeV[v].data();
	u_char*			dst = out[v].data();
	const u_char* const	end = dst + out.width();
#if defined(SIMD)
	using namespace		simd;

	const size_t		nelms = F32vec::size;
	for (const u_char* const end2 = dst + Iu8vec::floor(out.width());
	     dst < end2; dst += Iu8vec::size)
	{
	    const Is32vec	d0 = dir8x(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d1 = dir8x(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d2 = dir8x(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    const Is32vec	d3 = dir8x(load<false>(eH), load<false>(eV));
	    eH += nelms;
	    eV += nelms;
	    store<false>(dst,
			 cvt<u_char>(cvt<short>(d0, d1), cvt<short>(d2, d3)));
	}
#endif
	while (dst < end)
	    *dst++ = dir8x(*eH++, *eV++);
    }
    
    return *this;
}
    
const EdgeDetector&
EdgeDetector::ridge(const Image<float>& edgeHH,
		    const Image<float>& edgeHV,
		    const Image<float>& edgeVV,
		    Image<float>& strength, Image<u_char>& direction) const
{
    typedef ImageLine<float>::const_iterator	const_fiterator;
    typedef ImageLine<float>::iterator		fiterator;
    typedef ImageLine<u_char>::iterator		citerator;
    
    strength.resize(edgeHH.height(), edgeHH.width());
    direction.resize(edgeHH.height(), edgeHH.width());
    for (size_t v = 0; v < strength.height(); ++v)
    {
	const_fiterator	eHH = edgeHH[v].begin(), eHV = edgeHV[v].begin(),
			eVV = edgeVV[v].begin();
	fiterator	str = strength[v].begin();
	citerator	dir = direction[v].begin();
#if defined(SSE)
	using namespace	simd;

	const size_t	nelms = F32vec::size;
	for (citerator end2 = dir + Iu8vec::floor(direction.width());
	     dir != end2; dir += Iu8vec::size)
	{
	    F32vec	fHH = load<false>(eHH), fHV = load<false>(eHV);
	    F32vec	lambda = eigen(fHH, fHV, load<false>(eVV));
	    Is32vec	d0 = dir8x(fHV, lambda - fHH, lambda);
	    store<false>(str, abs(lambda));
	    str += nelms;
	    eHH += nelms;
	    eHV += nelms;
	    eVV += nelms;

	    fHH = load<false>(eHH);
	    fHV = load<false>(eHV);
	    lambda = eigen(fHH, fHV, load<false>(eVV));
	    Is32vec	d1 = dir8x(fHV, lambda - fHH, lambda);
	    store<false>(str, abs(lambda));
	    str += nelms;
	    eHH += nelms;
	    eHV += nelms;
	    eVV += nelms;

	    fHH = load<false>(eHH);
	    fHV = load<false>(eHV);
	    lambda = eigen(fHH, fHV, load<false>(eVV));
	    Is32vec	d2 = dir8x(fHV, lambda - fHH, lambda);
	    store<false>(str, abs(lambda));
	    str += nelms;
	    eHH += nelms;
	    eHV += nelms;
	    eVV += nelms;

	    fHH = load<false>(eHH);
	    fHV = load<false>(eHV);
	    lambda = eigen(fHH, fHV, load<false>(eVV));
	    Is32vec	d3 = dir8x(fHV, lambda - fHH, lambda);
	    store<false>(str, abs(lambda));
	    str += nelms;
	    eHH += nelms;
	    eHV += nelms;
	    eVV += nelms;

	    store<false>(dir,
			 cvt<u_char>(cvt<short>(d0, d1), cvt<short>(d2, d3)));
	}
#endif
	for (citerator end = direction[v].end(); dir != end; ++dir)
	{
	    const float	lambda = eigen(*eHH, *eHV, *eVV);
	    *str = std::abs(lambda);
	    *dir = dir8x(*eHV, lambda - *eHH, lambda);
	    
	    ++eHH;
	    ++eHV;
	    ++eVV;
	    ++str;
	}
    }

    return *this;
}
    
//! 非極大値抑制処理により細線化を行う
/*!
  \param strength	エッジ強度入力画像
  \param direction	エッジ方向入力画像
  \param out		強いエッジ点と弱いエッジ点にそれぞれ #EDGE ラベルと
			#WEAK ラベルを付けた出力画像
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
	for (size_t u = 0; u < out.width(); ++u)
	    out[0][u] = out[out.height()-1][u] = 0;
    if (out.width() > 0)
	for (size_t v = 0; v < out.height(); ++v)
	    out[v][0] = out[v][out.width()-1] = 0;

  // 各点のエッジ強度が (1) その点のエッジ方向に沿った両隣と比較して極大に
  // なっており，かつ，(2a) 強い閾値以上ならばEDGEラベルを，(2b) 弱い閾値
  // 以上ならばWEAKラベルをそれぞれ書き込む．そうでなければ0を書き込む．
    for (size_t v = 0; ++v < out.height() - 1; )
    {
	const float		*prv = strength[v-1].data(),
				*str = strength[v].data(),
				*nxt = strength[v+1].data();
	const u_char		*dir = direction[v].data();
	const u_char* const	end  = &out[v][out.width() - 1];
	for (u_char* dst = out[v].data(); ++dst < end; )
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
	for (size_t u = 0; u < out.width(); ++u)
	    out[out.height()-1][u] = 0;
    if (out.width() > 0)
	for (size_t v = 0; v < out.height(); ++v)
	    out[v][out.width()-1] = 0;

  // 現在点を左上隅とする2x2ウィンドウ中の画素が異符号ならエッジ点とする．
    for (size_t v = 0; v < out.height() - 1; ++v)
    {
	const float		*cur = in[v].data(),
				*nxt = in[v+1].data();
	const u_char* const	end  = &out[v][out.width() - 1];
	for (u_char* dst = out[v].data(); dst < end; )
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
  \param out		強いエッジ点と弱いエッジ点にそれぞれ #EDGE ラベルと
			#WEAK ラベルを付けた出力画像
  \return		このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::zeroCrossing(const Image<float>& in, const Image<float>& strength,
			   Image<u_char>& out) const
{
    out.resize(in.height(), in.width());

  // 出力画像の外周を0にする．
    if (out.height() > 0)
	for (size_t u = 0; u < out.width(); ++u)
	    out[0][u] = out[out.height()-1][u] = 0;
    if (out.width() > 0)
	for (size_t v = 0; v < out.height(); ++v)
	    out[v][0] = out[v][out.width()-1] = 0;

  // 現在点を左上隅とする2x2ウィンドウ中の画素が異符号ならエッジ点とする．
    for (size_t v = 0; ++v < out.height() - 1; )
    {
	const float		*cur = in[v].data(),
				*nxt = in[v+1].data(),
				*str = strength[v].data();
	const u_char* const	end  = &out[v][out.width() - 1];
	for (u_char* dst = out[v].data(); ++dst < end; )
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
  \param edge		強いエッジ点と弱いエッジ点にそれぞれ #EDGE ラベルと
			#WEAK ラベルを付けた画像．処理が終わると最終的なエッジ
			点に255を，そうでない点には0を書き込んで返される．
  \return		このエッジ検出器自身
*/
const EdgeDetector&
EdgeDetector::hysteresisThresholding(Image<u_char>& edge) const
{
  // 強いエッジ点を起点にして，接続する弱いエッジ点を追跡しEDGEラベルを付ける．
    for (size_t v = 0; ++v < edge.height() - 1; )
	for (size_t u = 0; ++u < edge.width() - 1; )
	    if (edge[v][u] & EDGE)
		trace(edge, Point2i(u, v));

  // EDGEラベルが付いておらず，かつ付いている点と付いていない弱いエッジ点の
  // 橋渡しになれる点に新たにEDGEラベルを付けて追跡を行う．
    for (size_t v = 0; ++v < edge.height() - 1; )
	for (size_t u = 0; ++u < edge.width() - 1; )
	{
	    Point2i	p(u, v);
	    
	    if (!(edge(p) & EDGE) && canInterpolate(edge, p))
		trace(edge, p);
	}

  // EDGE点には255を，そうでない点には0を書き込む．
    for (size_t v = 0; v < edge.height(); )
    {
	u_char*	dst = edge[v++].data();
	for (const u_char* const end = dst + edge.width(); dst < end; ++dst)
	    *dst = (*dst & EDGE ? 255 : 0);
    }
 
    return *this;
}
    
}
