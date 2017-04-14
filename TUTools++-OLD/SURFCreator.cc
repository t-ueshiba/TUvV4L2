/*
 *  平成21-22年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2009-2010.
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
 *  $Id: SURFCreator.cc,v 1.15 2012-01-24 02:44:54 ueshiba Exp $
 */
#include "TU/SURFCreator.h"
#include <algorithm>

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class T> static bool
SolveLinearSystem33(T* solution, T sq[3][3])
{
    const int	size = 3;
    int		pivot = 0;

  /* Triangularize the matrix. */
    for (int col = 0; col < size - 1; col++) 
    {
      /* Pivot row with largest coefficient to top. */
	T	maxc = -1.0;
	for (int row = col; row < size; row++) 
	{
	    T	coef = sq[row][col];
	    coef = (coef < 0.0 ? - coef : coef);
	    if (coef > maxc)
	    {
		maxc = coef;
		pivot = row;
	    }
	}
	if (pivot != col)
	{
	  /* Exchange "pivot" with "col" row (this is no less efficient
	     than having to perform all array accesses indirectly). */
	    for (int i = 0; i < size; i++) 
	    {
		T	temp = sq[pivot][i];
		sq[pivot][i] = sq[col][i];
		sq[col][i] = temp;
	    }
	    T	temp = solution[pivot];
	    solution[pivot] = solution[col];
	    solution[col] = temp;
	}
      /* Do reduction for this column. */
	for (int row = col + 1; row < size; row++) 
	{
	    T	mult = sq[row][col] / sq[col][col];
	    for (int c = col; c < size; c++)	/* Could start with c=col+1. */
		sq[row][c] -= mult * sq[col][c];
	    solution[row] -= mult * solution[col];
	}
    }

  /* Do back substitution.  Pivoting does not affect solution order. */
    for (int row = size - 1; row >= 0; row--)
    {
	T	val = solution[row];
	for (int col = size - 1; col > row; col--)
	    val -= solution[col] * sq[row][col];
	solution[row] = val / sq[row][row];
    }
    return true;
}
    
/************************************************************************
*  class LUT								*
************************************************************************/
template <int LB, int UB>  
class LUT 
{ 
  public:
    typedef SURFCreator::value_type				value_type;
    
    explicit		LUT(value_type (*f) (value_type),
			    value_type coeffadd = 0, value_type coeffmul = 1)
			    :_lut(_lut_array - LB)
			{
			    for (int i = LB; i <= UB; ++i) 
				_lut[i] = f(coeffmul * (i + coeffadd)); 
			} 
	
    const value_type&	operator ()(int i)	const	{return _lut[i];} 

  private: 
    value_type		_lut_array[UB - LB + 1]; 
    value_type* const	_lut; 
};

static LUT<0, 83>	Exp1(std::exp, 0.5, -0.08);
static LUT<0, 40>	Exp2(std::exp, 0.5, -0.125);

/************************************************************************
*  struct HarrResponse							*
************************************************************************/
struct HarrResponse
{
    typedef SURFCreator::value_type				value_type;

    HarrResponse() :orientation(0), ponderation(0)			{}
    HarrResponse(value_type ori, value_type pon)
	:orientation(ori), ponderation(pon)				{}
    
    value_type	orientation;
    value_type	ponderation;
};

bool
operator <(const HarrResponse& resp0, const HarrResponse& resp1)
{
    return resp0.orientation < resp1.orientation;
}
    
/************************************************************************
*   class SURFCreator::BoxFilter					*
************************************************************************/
inline
SURFCreator::BoxFilter::BoxFilter(
    const IntegralImage<value_type>& integralImage, value_type baseSize)
    :_integralImage(integralImage),
     _sqCorrectFactor(0), _lc(0), _lm(0), _lr(0), _lb(0), _w1(0)
{
    setBaseSize(baseSize);
}

inline SURFCreator::BoxFilter&
SURFCreator::BoxFilter::setBaseSize(value_type baseSize)
{
    _sqCorrectFactor = 81.0 / (baseSize * baseSize * baseSize * baseSize);
    _lc = (int(baseSize * 3) - 1) / 2 - 1;

    int	tmp = int(baseSize + 0.5);	// rounding
    _lm = tmp / 2;
    _lr = _lm + tmp;
    _lb = _lm * 2;
    _w1 = int(baseSize);
    
    return *this;
}

inline SURFCreator::value_type
SURFCreator::BoxFilter::getDxx(size_t x) const
{
    return _integralImage.crop2(x - _lr, x + _lr, _y_lb_min, _y_lb_max)
   - 3.0 * _integralImage.crop2(x - _lm, x + _lm, _y_lb_min, _y_lb_max);
}

inline SURFCreator::value_type
SURFCreator::BoxFilter::getDyy(size_t x) const
{
    return _integralImage.crop2(x - _lb, x + _lb, _y_lr_min, _y_lr_max)
   - 3.0 * _integralImage.crop2(x - _lb, x + _lb, _y_lm_min, _y_lm_max);
}

inline SURFCreator::value_type
SURFCreator::BoxFilter::getDxy(size_t x) const
{
    return _integralImage.crop2(x,	 x + _lc, _y,	     _y_lc_max)
	 + _integralImage.crop2(x - _lc, x,	  _y_lc_min, _y       )
	 - _integralImage.crop2(x,	 x + _lc, _y_lc_min, _y       )
	 - _integralImage.crop2(x - _lc, x,	  _y,	     _y_lc_max);
}

inline SURFCreator::value_type
SURFCreator::BoxFilter::getDet(size_t x) const
{
    const value_type	dxx = getDxx(x),
			dyy = getDyy(x),
			dxy = getDxy(x) * 0.9 * 2 / 3.0;
    return ((dxx * dyy) - (dxy * dxy)) * _sqCorrectFactor;
}

  //#if defined(SIMD)
#if defined(SSE)
namespace simd
{
template <size_t OCTAVE>
static F32vec	octave_load(const float* p)				;

template <> inline F32vec
octave_load<0>(const float* p)
{
    return load(p);
}

template <> inline F32vec
octave_load<1>(const float* p)
{
    return shuffle<2, 0, 2, 0>(load(p), load(p + 4));
}

}	// namespace simd

template <size_t OCTAVE> inline simd::F32vec
SURFCreator::BoxFilter::simdCrop2(int umin, int umax, int vmin, int vmax) const
{
    using namespace	simd;
    
    ++umax;
    ++vmax;
    return octave_load<OCTAVE>(&_integralImage[vmax][umax])
	 + octave_load<OCTAVE>(&_integralImage[vmin][umin])
	 - octave_load<OCTAVE>(&_integralImage[vmin][umax])
	 - octave_load<OCTAVE>(&_integralImage[vmax][umin]);
}

template <size_t OCTAVE> inline simd::F32vec
SURFCreator::BoxFilter::simdGetDxx(size_t x) const
{
    using namespace	simd;

    return simdCrop2<OCTAVE>(x - _lr, x + _lr, _y_lb_min, _y_lb_max)
      - F32vec(3.0) * simdCrop2<OCTAVE>(x - _lm, x + _lm, _y_lb_min, _y_lb_max);
}

template <size_t OCTAVE> inline simd::F32vec
SURFCreator::BoxFilter::simdGetDyy(size_t x) const
{
    using namespace	simd;

    return simdCrop2<OCTAVE>(x - _lb, x + _lb, _y_lr_min, _y_lr_max)
      - F32vec(3.0) * simdCrop2<OCTAVE>(x - _lb, x + _lb, _y_lm_min, _y_lm_max);
}

template <size_t OCTAVE> inline simd::F32vec
SURFCreator::BoxFilter::simdGetDxy(size_t x) const
{
    return simdCrop2<OCTAVE>(x,	      x + _lc, _y,	  _y_lc_max)
	 + simdCrop2<OCTAVE>(x - _lc, x,       _y_lc_min, _y       )
	 - simdCrop2<OCTAVE>(x,	      x + _lc, _y_lc_min, _y       )
	 - simdCrop2<OCTAVE>(x - _lc, x,       _y,	  _y_lc_max);
}

template <size_t OCTAVE> inline simd::F32vec
SURFCreator::BoxFilter::simdGetDet(size_t x) const
{
    using namespace	simd;

    const F32vec	dxx = simdGetDxx<OCTAVE>(x),
			dyy = simdGetDyy<OCTAVE>(x),
			dxy = simdGetDxy<OCTAVE>(x) * F32vec(0.9 * 2 / 3.0);
    return ((dxx * dyy) - (dxy * dxy)) * F32vec(_sqCorrectFactor);
}
#endif

inline SURFCreator::value_type
SURFCreator::BoxFilter::getWx(size_t x, size_t y)
{
    return _integralImage.crop2(x,	 x + _w1, y - _w1, y + _w1)
	 - _integralImage.crop2(x - _w1, x,	  y - _w1, y + _w1);
}

inline SURFCreator::value_type
SURFCreator::BoxFilter::getWy(size_t x, size_t y)
{
    return _integralImage.crop2(x - _w1, x + _w1, y - _w1, y	  )
	 - _integralImage.crop2(x - _w1, x + _w1, y,	   y + _w1);
}

inline SURFCreator::BoxFilter&
SURFCreator::BoxFilter::setY(size_t y)
{
    _y_lb_min = y - _lb;
    _y_lb_max = y + _lb;
    _y_lr_min = y - _lr;
    _y_lr_max = y + _lr;
    _y_lm_min = y - _lm;
    _y_lm_max = y + _lm;
    _y_lc_min = y - _lc;
    _y_lc_max = y + _lc;
    _y	      = y;

    return *this;
}

inline bool
SURFCreator::BoxFilter::checkBounds(int x, int y) const
{
    return (x > _lr && x + _lr < int(_integralImage.width())  - 1 &&
	    y > _lr && y + _lr < int(_integralImage.height()) - 1);
}

inline bool
SURFCreator::BoxFilter::checkBoundsW(int x, int y) const
{
    return (x > _w1 && x + _w1 < int(_integralImage.width())  - 1 &&
	    y > _w1 && y + _w1 < int(_integralImage.height()) - 1);
}

/************************************************************************
*  class SURFCreator							*
************************************************************************/
const SURFCreator::value_type	SURFCreator::_magFactor
					= 12.0 / SURFCreator::NSubRegions;
const SURFCreator::value_type	SURFCreator::_baseSigma = 1.2;

/*
 *  static member functions
 */
template <> inline void
SURFCreator::vote<64u>(value_type uIdx, value_type vIdx,
		       value_type wu,   value_type wv, Histogram bins)
{
    const int		u = int(uIdx + 1.0);
    const int		v = int(vIdx + 1.0);
    const value_type	weightU0 = u - uIdx;
    const value_type	weightU1 = 1.0 - weightU0;
    value_type		weightV  = v - vIdx;
    const value_type	weight00 = weightV * weightU0;
    const value_type	weight01 = weightV * weightU1;
    weightV = 1.0 - weightV;
    const value_type	weight10 = weightV * weightU0;
    const value_type	weight11 = weightV * weightU1;

    HistSlice		*q   = bins + v;
    value_type		*p00 = (*  q)[u], *p01 = (*q)[u+1],
			*p10 = (*++q)[u], *p11 = (*q)[u+1];
    int			 bin = (wu <= 0 ? 0 : 1);
    p00[bin] += weight00 * wu;
    p01[bin] += weight01 * wu;
    p10[bin] += weight10 * wu;
    p11[bin] += weight11 * wu;
		    
    bin = (wv <= 0 ? 2 : 3);
    p00[bin] += weight00 * wv;
    p01[bin] += weight01 * wv;
    p10[bin] += weight10 * wv;
    p11[bin] += weight11 * wv;
}

template <> inline void
SURFCreator::vote<128u>(value_type uIdx, value_type vIdx,
			value_type wu,   value_type wv, Histogram bins)
{
    const int		u = int(uIdx + 1.0);
    const int		v = int(vIdx + 1.0);

    const value_type	weightU0 = u - uIdx;
    const value_type	weightU1 = 1.0 - weightU0;
    value_type		weightV  = v - vIdx;
    const value_type	weight00 = weightV * weightU0;
    const value_type	weight01 = weightV * weightU1;
    weightV = 1.0 - weightV;
    const value_type	weight10 = weightV * weightU0;
    const value_type	weight11 = weightV * weightU1;

    HistSlice		*q   = bins + v;
    value_type		*p00 = (*  q)[u], *p01 = (*q)[u + 1],
			*p10 = (*++q)[u], *p11 = (*q)[u + 1];
    int			 bin = (wv <= 0 ? 0 : 1);
    p00[bin] += weight00 * wu;
    p01[bin] += weight01 * wu;
    p10[bin] += weight10 * wu;
    p11[bin] += weight11 * wu;

    bin += 2;
    value_type	val = std::abs(wu);
    p00[bin] += weight00 * val;
    p01[bin] += weight01 * val;
    p10[bin] += weight10 * val;
    p11[bin] += weight11 * val;
		    
    bin = (wu <= 0 ? 4 : 5);
    p00[bin] += weight00 * wv;
    p01[bin] += weight01 * wv;
    p10[bin] += weight10 * wv;
    p11[bin] += weight11 * wv;

    bin += 2;
    val = std::abs(wv);
    p00[bin] += weight00 * val;
    p01[bin] += weight01 * val;
    p10[bin] += weight10 * val;
    p11[bin] += weight11 * val;
}

/*
 *  private member functions
 */
template <class T, size_t D> void
SURFCreator::assignOrientation(Feature<T, D>& feature) const
{
    using namespace	std;

  // 特徴点の位置とスケール変化のステップを取り出す．
    const int		xc   = int(feature[0] + 0.5),
			yc   = int(feature[1] + 0.5);
    const int		step = int(feature.sigma + 0.8);
    HarrResponse	resp[19*19*2];
    BoxFilter		boxFilter(_integralImage, 2.0 * feature.sigma + 1.6);

  // 半径6*sigmaの円内におけるHarr wavelet値を計算する．
    HarrResponse*	respEnd = resp;
    for (int j = -9; j <= 9; ++j)
    { 
	const int	y = yc + j * step;
	
	for (int i = -9; i <= 9; ++i)
	{
	    const int	x = xc + i * step;

	  // keep points in a circular region of diameter 6s
	    const int	sqrad = i * i + j * j;
	    if (sqrad <= 81 && boxFilter.checkBoundsW(x, y))
	    {
		value_type	wx  = boxFilter.getWx(x, y),
				wy  = boxFilter.getWy(x, y);
		value_type	len = sqrt(wx * wx + wy * wy);
		if (len > 0)
		{
		    value_type	orientation = atan2(wy, wx),
				ponderation = len * Exp1(sqrad);
		    *respEnd++ = HarrResponse(orientation, ponderation);
		    *respEnd++ = HarrResponse(orientation + 2*M_PI,
					      ponderation);
		}
	    }
	}
    }
    std::sort(resp, respEnd);	// orientation によって昇順にソート

  // 幅が M_PI/3 のsliding windowを用いて特徴点の方向を推定する．
    value_type	meanOrientation	  = 0,
		arcPonderation	  = resp[0].ponderation,
		maxPonderation	  = 0,
		ponderedDirection = resp[0].orientation * arcPonderation;
    for (HarrResponse *winBegin = resp, *winEnd = resp;
	 winBegin->orientation < M_PI && winEnd != respEnd; )
    {
      // ウィンドウの幅が M_PI/3 に満たなければ右端を1つ拡げ，
      // そうでなければ左端を1つ縮める．
	if (winEnd->orientation - winBegin->orientation < M_PI/3)
        {
            if (maxPonderation < arcPonderation)
            {
		meanOrientation = ponderedDirection / arcPonderation;
		maxPonderation  = arcPonderation;              
            }

            winEnd++;
            arcPonderation    += winEnd->ponderation;
            ponderedDirection += winEnd->ponderation * winEnd->orientation;
        }
        else
        {
	    arcPonderation    -= winBegin->ponderation;          
	    ponderedDirection -= winBegin->ponderation * winBegin->orientation;
	    winBegin++;
	}
    }

  // 特徴点の angle が [0, 2*M_PI) に収まるように修正する．
    while (meanOrientation < 0)
	meanOrientation += 2*M_PI;
    while (meanOrientation >= 2*M_PI)
	meanOrientation -= 2*M_PI;
    
    feature.angle = meanOrientation;
}

template <class T, size_t D> void
SURFCreator::makeDescriptor(Feature<T, D>& feature) const
{
  // create the vector of features by analyzing a square patch around the point.
  // for this the current patch (x,y) will be translated in rotated coordinates
  // (u,v)

    value_type	aS   = feature.sigma * 1.65;	// multiply by this nice constant
    const int	step = std::max(int(aS / 2.0 + 0.5), 1);
    if (aS < 1)
	aS = 1;
    BoxFilter	boxFilter(_integralImage, int(aS));

    const value_type	cos = std::cos(feature.angle),	// 特徴点の方向の余弦
			sin = std::sin(feature.angle);	// 特徴点の方向の正弦
    const int		xc  = int(feature[0] + 0.5),	// 特徴点の整数横座標
			yc  = int(feature[1] + 0.5);	// 特徴点の整数縦座標
    const value_type	dx  = feature[0] - xc,		// 特徴点の端数横座標
			dy  = feature[1] - yc;		// 特徴点の端数縦座標
    const value_type	du  = cos * dx - sin * dy,	// 回転された端数横座標
			dv  = sin * dx + cos * dy;	// 回転された端数縦座標

  // make a bounding box around the rotated patch square. 
    const value_type	stepSample = aS * _magFactor;	// サンプリングステップ
    const int		radius = int((1.414 * stepSample)*(NSubRegions + 1) /
				     (2.0 * (step + 0.5)));

  // ヒストグラムを空に初期化する．
#if defined(SSE)
    alignas(16) Histogram	bins;
#else
    Histogram			bins;
#endif
    for (int y = 0; y < NSubRegions + 2; ++y)
	for (int x = 0; x < NSubRegions + 2; ++x)
	    for (int v = 0; v < VecLenMax; ++v)
		bins[y][x][v] = 0;
    
  // Go through all the pixels in the bounding box.
    for (int j = -radius; j <= radius; ++j)
    {
	for (int i = -radius; i <= radius; ++i)
	{
	  // Calculate U,V rotated values from X,Y taking subpixel correction
	  // into account. Divide by step sample to put in index units.
	    const value_type	u = ((cos * i - sin * j) * step - du)
				  / stepSample,
				v = ((sin * i + cos * j) * step - dv)
				  / stepSample;

	  // Compute location of sample in terms of real array coordinates.
	    const value_type	uIdx = NSubRegions / 2.0 - 0.5 + u,
				vIdx = NSubRegions / 2.0 - 0.5 + v;
	    
	  // test if some bins will be filled.
	    if (-1 <= uIdx && uIdx < NSubRegions &&
		-1 <= vIdx && vIdx < NSubRegions)
	    {
		const int	x = xc + i * step,
				y = yc + j * step;

		if (boxFilter.checkBoundsW(x, y))
		{
		    const value_type	ex = Exp2(int(u * u + v * v)); 
		    const value_type	wx = boxFilter.getWx(x, y) * ex,
					wy = boxFilter.getWy(x, y) * ex;
		    const value_type	wu = cos * wx + sin * wy,
					wv = sin * wx - cos * wy; // ?? pano-matic
		      //wv = -sin * wx + cos * wy;

		    vote<D>(uIdx, vIdx, wu, wv, bins);
		}
	    }		
	}
    }

  // Transform back to vector.
  // Fill the vector with the values of the square...
  // Remember the shift of 1 to drop outborders.
    const int	vecLen = D / (NSubRegions * NSubRegions);
    value_type*	p = feature.descriptor.data();
    for(int y = 1; y < NSubRegions + 1; ++y)
	for(int x = 1; x < NSubRegions + 1; ++x)
	    for(int v = 0; v < vecLen; ++v)
		*p++ = bins[y][x][v];
    feature.descriptor.normalize();
}

void
SURFCreator::calcDetsLine(size_t y, size_t o, size_t filterSize,
			  size_t borderSize, Matrix<value_type>& det) const
{

    const size_t	pixelStep   = 1 << o;			// 2^octave
    const size_t	octaveWidth = _integralImage.originalWidth()
				    / pixelStep;
    value_type*		p	    = &det[y][borderSize];
    value_type*		q	    = p + octaveWidth - 2 * borderSize;
    size_t		xx	    = borderSize * pixelStep;
    size_t		yy	    = y * pixelStep;
    BoxFilter		boxFilter(_integralImage, filterSize);
    boxFilter.setY(yy);
    
  //#if defined(SIMD)
#if defined(SSE2)
    using namespace	simd;

    if (o == 0)
    {
	const size_t	nelms = vec<value_type>::size;
	for (value_type* const r = q - nelms; p <= r; p += nelms)
	{
	    store(p, boxFilter.simdGetDet<0>(xx));
	    xx += pixelStep * nelms;
	}
	empty();
    }
    else if (o == 1)
    {
	const size_t	nelms = vec<value_type>::size;
	for (value_type* const r = q - nelms; p <= r; p += nelms)
	{
	    store(p, boxFilter.simdGetDet<1>(xx));
	    xx += pixelStep * nelms;
	}
	empty();
    }
#endif
    while (p < q)
    {
	*p++ = boxFilter.getDet(xx);
	xx += pixelStep;
    }
}

template <class F> void
SURFCreator::detectLine(size_t y, size_t s, size_t pixelStep,
			const Array<Matrix<value_type> >& det,
			const Array<size_t>& borderSizes,
			Inserter<F>& insert) const
{
    const size_t	borderSize   = borderSizes[s+1];
    const size_t	octaveHeight = _integralImage.originalHeight()
				     / pixelStep;
    const size_t	octaveWidth  = _integralImage.originalWidth()
				     / pixelStep;
    const size_t	xend	     = octaveWidth - borderSize - 1;
    for (size_t x = borderSize + 1; x < xend; x += 2)
    {
      // 2x2x2近傍における最大点を探す．
	value_type tab[8];
	tab[0] = det[s]  [y]  [x];
	tab[1] = det[s]  [y]  [x+1];
	tab[2] = det[s]  [y+1][x];
	tab[3] = det[s]  [y+1][x+1];
	tab[4] = det[s+1][y]  [x];
	tab[5] = det[s+1][y]  [x+1];
	tab[6] = det[s+1][y+1][x];
	tab[7] = det[s+1][y+1][x+1];
					
      // Find the max index without using a loop.
	size_t a04    = (tab[0]     > tab[4]     ? 0	: 4);
	size_t a15    = (tab[1]     > tab[5]     ? 1     : 5);
	size_t a26    = (tab[2]     > tab[6]     ? 2     : 6);
	size_t a37    = (tab[3]     > tab[7]     ? 3     : 7);
	size_t a0426  = (tab[a04]   > tab[a26]   ? a04   : a26);
	size_t a1537  = (tab[a15]   > tab[a37]   ? a15   : a37);
	size_t maxIdx = (tab[a0426] > tab[a1537] ? a0426 : a1537);

      // 最大点のスコアが近似しきい値を上回ることを確認する．
	value_type	score = tab[maxIdx];
	if (score < _params.scoreThresh * 0.8)
	    continue;

      // 他の3x3x3近傍点がすべてしきい値を下回ることを確認する．
	const int	xc = x + (maxIdx & 1);		//  x or x + 1
	const int	sx = 2 * (maxIdx & 1) - 1;	// -1 or 1
	maxIdx >>= 1;
	    
	const int	yc = y + (maxIdx & 1);		//  y or y + 1
	const int	sy = 2 * (maxIdx & 1) - 1;	// -1 or 1
	maxIdx >>= 1;
		    
	const int	sc = s + (maxIdx & 1);		//  s or s + 1
	const int	ss = 2 * (maxIdx & 1) - 1;	// -1 or 1

      // Skip too high scale ajusting.
	if (sc == int(_params.nScales) - 1)
	    continue;		

      // 極大点 (xc, yc, sc) の未調査の近傍点を調べる．
	const Matrix<value_type>*	slice = &det[sc+ss];
	const value_type*		p = &(*slice)[yc-1][xc];
	if ((*(p-1) > score) || (*p > score) || (*(p+1) > score))
	    continue;

	p = &(*slice)[yc][xc];
	if ((*(p-1) > score) || (*p > score) || (*(p+1) > score))
	    continue;

	p = &(*slice)[yc+1][xc];
	if ((*(p-1) > score) || (*p > score) || (*(p+1) > score))
	    continue;

	slice = &det[sc];
	p = &(*slice)[yc+sy][xc];
	if ((*(p-1) > score) || (*p > score) || (*(p+1) > score) ||
	    ((*slice)[yc   ][xc+sx] > score) ||
	    ((*slice)[yc-sy][xc+sx] > score))
	    continue;
		    
	slice = &det[sc-ss];
	p = &(*slice)[yc+sy][xc];
	if ((*(p-1) > score) || (*p > score) || (*(p+1) > score) ||
	    ((*slice)[yc   ][xc+sx] > score) ||
	    ((*slice)[yc-sy][xc+sx] > score))
	    continue;

      // 極大点の位置とそのスコアを微調整する．
	value_type	xf, yf, sf;
	if (!fineTuneExtrema(det, xc, yc, sc, xf, yf, sf, score,
			     octaveWidth, octaveHeight, borderSizes[sc]))
	    continue;
		    
      // 微調整されたスコアがしきい値を上回ることを確認する．
	if (score < _params.scoreThresh)
	    continue;
	    
      // Adjust the values.
	xf *= pixelStep;
	yf *= pixelStep;
      // This one was hard to guess...
	sf = ((2 * sf * pixelStep) +
	      _params.initialBoxFilterSize +
	      (pixelStep-1) * _params.nScales) / 3.0;

      // トレースを計算する．
	int	trace;
	if (!calcTrace(xf, yf, sf, trace))
	    continue;

      // 特徴点を生成し，挿入子によってユーザ側のコンテナに挿入する．
	insert(F(xf, yf, sf * _baseSigma, 0, score, trace));
    }
}

bool
SURFCreator::fineTuneExtrema(const Array<Matrix<value_type> >& det,
			     int x, int y, int s, 
			     value_type& xf, value_type& yf,
			     value_type& sf, value_type& score,
			     size_t octaveWidth, size_t octaveHeight,
			     size_t borderSize)
{
    const size_t	niterMax = 6;		// 最大繰り返し数
    value_type		dx = 0, dy = 0, ds = 0;	// current deviation
    value_type		vec[3];			// result vector: (x,y,s)

    for (size_t niter = 0; niter < niterMax; ++niter)
    {
      // Fill the result vector with gradient from pixels differences
      // (negate to prepare system solve).
	dx = (det[s  ][y  ][x+1] - det[s  ][y  ][x-1]) * 0.5;
	dy = (det[s  ][y+1][x  ] - det[s  ][y-1][x  ]) * 0.5;
	ds = (det[s+1][y  ][x  ] - det[s-1][y  ][x  ]) * 0.5;

	vec[0] = -dx; 
	vec[1] = -dy;
	vec[2] = -ds; 

      // Create the problem matrix.
	value_type	M[3][3];	//symetric, no ordering problem
	M[0][0] = det[s  ][y  ][x-1] - 2.0 * det[s][y][x] + det[s  ][y  ][x+1];
	M[1][1] = det[s  ][y-1][x  ] - 2.0 * det[s][y][x] + det[s  ][y+1][x  ];
	M[2][2] = det[s-1][y  ][x  ] - 2.0 * det[s][y][x] + det[s+1][y  ][x  ];

	M[0][1] = M[1][0]
	        = (det[s  ][y+1][x+1] + det[s  ][y-1][x-1] -
		   det[s  ][y+1][x-1] - det[s  ][y-1][x+1]) * 0.25; 
	M[0][2] = M[2][0]
	        = (det[s+1][y  ][x+1] + det[s-1][y  ][x-1] -
		   det[s+1][y  ][x-1] - det[s-1][y  ][x+1]) * 0.25; 
	M[1][2] = M[2][1]
	        = (det[s+1][y+1][x  ] + det[s-1][y-1][x  ] -
		   det[s+1][y-1][x  ] - det[s-1][y+1][x  ]) * 0.25; 

      // Solve the linear system. Results are in vec.
	if (!SolveLinearSystem33(vec, M))
	    return false;

      // Ajust the shifts with the results and stop if no significant change.
	if (niter < niterMax - 1)
	{
	    int	sx = 0, sy = 0;			// シフト量: -1, 0, 1

	    if (     vec[0] >  0.6 && x < int(octaveWidth - borderSize - 2))
		++sx;
	    else if (vec[0] < -0.6 && x > int(borderSize + 1))
		--sx;

	    if (     vec[1] >  0.6 && y < int(octaveHeight - borderSize - 2))
		++sy;
	    else if (vec[1] < -0.6 && y > int(borderSize + 1))
		--sy;
			
	    if (sx == 0 && sy == 0) 
		break;

	    x += sx;
	    y += sx;
	}
    }

  // Update the score.
    score = det[s][y][x] + 0.5 * (dx * vec[0] + dy * vec[1] + ds * vec[2]);

  // Reject too big deviation in last step (unfinished job).
    if (std::abs(vec[0]) > 1.5 || std::abs(vec[1]) > 1.5  ||
	std::abs(vec[2]) > 1.5)
	return false;

  // Put the last deviation (not integer :) to the output.
    xf = x + vec[0];
    yf = y + vec[1];
    sf = s + vec[2];

    return true;
}

size_t
SURFCreator::getFilterSize(size_t octave, size_t scale) const
{
    const size_t	scaleShift = 2 << octave;
    return _params.initialBoxFilterSize
	 + (scaleShift - 2) * (_params.nScales - _params.scaleOverlap)
	 + scaleShift * scale;
}

size_t
SURFCreator::getBorderSize(size_t octave, size_t scale) const
{
    const size_t	scaleShift = 2 << octave;
    if (scale <= 2)
    {
	size_t	mult = (octave == 0 ? 1 : 2);
	return (getFilterSize(octave, 1) + mult*scaleShift)*3/scaleShift + 1;
    }	
    return getFilterSize(octave, scale)*3/scaleShift + 1;
}

bool
SURFCreator::calcTrace(value_type x, value_type y,
		       value_type scale, int& trace) const
{
    BoxFilter	boxFilter(_integralImage, 3*scale);
    const int	xx = int(x + 0.5), yy = int(y + 0.5);

    if(!boxFilter.checkBounds(xx, yy))
	return false;

    boxFilter.setY(yy);
    trace = (boxFilter.getDxx(xx) + boxFilter.getDyy(xx) <= 0.0 ? -1 : 1);

    return true;
}

/************************************************************************
*  class SURFCreator::Parameters					*
************************************************************************/
SURFCreator::Parameters::Parameters()
    :nScales(5),
     nOctaves(4),
     scoreThresh(1000),
     initialBoxFilterSize(3),
     scaleOverlap(3)
{
}

/************************************************************************
*  template instantiations						*
************************************************************************/
template void	SURFCreator::detectLine(
		    size_t y, size_t s, size_t pixelStep,
		    const Array<Matrix<value_type> >& det,
		    const Array<size_t>& borderSizes,
		    Inserter<SURF>& insert)			 const	;
template void	SURFCreator::detectLine(
		    size_t y, size_t s, size_t pixelStep,
		    const Array<Matrix<value_type> >& det,
		    const Array<size_t>& borderSizes,
		    Inserter<SURF128>& insert)			 const	;
template void	SURFCreator::assignOrientation(SURF& feature)	 const	;
template void	SURFCreator::assignOrientation(SURF128& feature) const	;
template void	SURFCreator::makeDescriptor(SURF& feature)	 const	;
template void	SURFCreator::makeDescriptor(SURF128& feature)	 const	;
}
