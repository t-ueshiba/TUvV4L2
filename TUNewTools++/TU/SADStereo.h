/*!
  \file		SADStereo.h
  \brief	SADステレオマッチングクラスの定義と実装
*/
#ifndef __TU_SADSTEREO_H
#define __TU_SADSTEREO_H

#include "TU/StereoBase.h"
#include "TU/Array++.h"
#include "TU/BoxFilter.h"
#include <boost/tuple/tuple_io.hpp>

namespace TU
{
/************************************************************************
*  class SADStereo<SCORE, DISP>						*
************************************************************************/
template <class SCORE, class DISP>
class SADStereo : public StereoBase<SADStereo<SCORE, DISP> >
{
  public:
    using Score			= SCORE;
    using Disparity		= DISP;

  private:
    using super			= StereoBase<SADStereo<Score, Disparity> >;
#if defined(SIMD)
    using ScoreVec		= simd::vec<Score>;
    using DisparityVec		= simd::vec<Disparity>;
#else
    using ScoreVec		= Score;
    using DisparityVec		= Disparity;
#endif
    using ScoreVecArray		= Array<ScoreVec>;
    using ScoreVecArray2	= Array2<ScoreVec>;
    using col_siterator		= typename ScoreVecArray2::iterator;
    using const_col_siterator	= typename ScoreVecArray2::const_iterator;
    using const_reverse_col_siterator
			= typename ScoreVecArray2::const_reverse_iterator;
    using const_reverse_col_sbox= box_filter_iterator<
				    const_reverse_col_siterator, ScoreVecArray>;
    using DisparityArray	= Array<Disparity>;
    using DisparityArray2	= Array2<Disparity>;
    using reverse_col_diterator	= typename DisparityArray::reverse_iterator;
    using FloatArray		= Array<float>;
    using reverse_col_fiterator	= FloatArray::reverse_iterator;

    struct ScoreUpdate
    {
	using result_type	= ScoreVec;
	using argument_type	= std::tuple<ScoreVec, ScoreVec,
					     ScoreVec, ScoreVec>;
	
	ScoreUpdate(Score blend)	:_blend(blend)	{}

	result_type	operator ()(const argument_type& args) const
			{
			    using 	std::get;

			    return _blend(get<0>(args), get<1>(args))
				 - _blend(get<2>(args), get<3>(args));
			}
	
      private:
	const Blend<ScoreVec>	_blend;
    };
    
    struct Buffers
    {
	void	initialize(size_t N, size_t D, size_t W)		;
	void	initialize(size_t N, size_t D, size_t W, size_t H)	;
	
	ScoreVecArray2		Q;	// W x D
	DisparityArray		dminL;	// 1 x (W - N + 1)
	FloatArray		delta;	// 1 x (W - N + 1)
	DisparityArray		dminR;	// 1 x (W + D - 1)
	ScoreVecArray		RminR;	// 1 x D
	DisparityArray2		dminV;	// (W - N + 1) x (H + D - 1)
	ScoreVecArray2		RminV;	// (W - N + 1) x D
    };

  public:
    struct Parameters : public super::Parameters
    {
	Parameters()	:windowSize(11), intensityDiffMax(20),
			 derivativeDiffMax(20), blend(0)		{}

	std::istream&	get(std::istream& in)
			{
			    super::Parameters::get(in);
			    in >> windowSize >> intensityDiffMax;

			    return in;
			}
	std::ostream&	put(std::ostream& out) const
			{
			    using namespace	std;

			    super::Parameters::put(out);
			    cerr << "  window size:                        ";
			    out << windowSize << endl;
			    cerr << "  maximum intensity difference:       ";
			    out << intensityDiffMax << endl;
			    cerr << "  maximum derivative difference:      ";
			    out << derivativeDiffMax << endl;
			    cerr << "  blend ratio:                        ";
			    out << blend << endl;
			    
			    return out;
			}
			    
	size_t	windowSize;		//!< ウィンドウのサイズ
	size_t	intensityDiffMax;	//!< 輝度差の最大値
	size_t	derivativeDiffMax;	//!< 輝度勾配差の最大値
	Score	blend;			//!< 輝度差と輝度勾配差の按分率
    };

  public:
    SADStereo()	:super(*this, 5), _params()				{}
    SADStereo(const Parameters& params)
	:super(*this, 5), _params(params)				{}

    const Parameters&
		getParameters()					const	;
    void	setParameters(const Parameters& params)			;
    size_t	getOverlap()					const	;
    template <class ROW, class ROW_D>
    void	match(ROW rowL, ROW rowLe, ROW rowR, ROW_D rowD)	;
    template <class ROW, class ROW_D>
    void	match(ROW rowL, ROW rowLe, ROW rowLlast,
		      ROW rowR, ROW rowV, ROW_D rowD)			;

  private:
    using	super::start;
    using	super::nextFrame;
    using	super::selectDisparities;
    using	super::pruneDisparities;
    
    template <class COL, class COL_RV>
    void	initializeDissimilarities(COL colL, COL colLe,
					  COL_RV colRV,
					  col_siterator colQ)	const	;
    template <class COL, class COL_RV>
    void	updateDissimilarities(COL colL, COL colLe, COL_RV colRV,
				      COL colLp, COL_RV colRVp,
				      col_siterator colQ)	const	;
    template <class DMIN_RV, class RMIN_RV>
    void	computeDisparities(const_reverse_col_siterator colQ,
				   const_reverse_col_siterator colQe,
				   reverse_col_diterator dminL,
				   reverse_col_fiterator delta,
				   DMIN_RV dminRV, RMIN_RV RminRV) const;

  private:
    Parameters					_params;
    typename super::template Pool<Buffers>	_bufferPool;
};
    
template <class SCORE, class DISP>
inline const typename SADStereo<SCORE, DISP>::Parameters&
SADStereo<SCORE, DISP>::getParameters() const
{
    return _params;
}
    
template <class SCORE, class DISP> inline void
SADStereo<SCORE, DISP>::setParameters(const Parameters& params)
{
    _params = params;
#if defined(SIMD)
    _params.disparitySearchWidth
	= simd::vec<Disparity>::ceil(_params.disparitySearchWidth);
#endif
    if (_params.disparityMax < _params.disparitySearchWidth)
	_params.disparityMax = _params.disparitySearchWidth;
}

template <class SCORE, class DISP> inline size_t
SADStereo<SCORE, DISP>::getOverlap() const
{
    return _params.windowSize - 1;
}
    
template <class SCORE, class DISP> template <class ROW, class ROW_D> void
SADStereo<SCORE, DISP>::match(ROW rowL, ROW rowLe, ROW rowR, ROW_D rowD)
{
    start(0);
    const size_t	N = _params.windowSize,
			D = _params.disparitySearchWidth,
			H = std::distance(rowL, rowLe),
			W = (H != 0 ? rowL->size() : 0);
    if (H < N || W < N)				// 充分な行数／列数があるか確認
	return;

    auto* const	buffers = _bufferPool.get();	// 各種作業領域を確保
    buffers->initialize(N, D, W);

    std::advance(rowD, N/2);	// 出力行をウィンドウサイズの半分だけ進める
    auto	rowLp = rowL;
    auto	rowRp = rowR;

  // 各行に対してステレオマッチングを行い視差を計算
    for (const auto rowL0 = rowL + N - 1; rowL != rowLe; ++rowL)
    {
	start(1);
	if (rowL <= rowL0)
	{
	    initializeDissimilarities(rowL->cbegin(), rowL->cend(),
				      rowR->cbegin(), buffers->Q.begin());
	}
	else
	{
	    updateDissimilarities(rowL->cbegin(), rowL->cend(), rowR->cbegin(),
				  rowLp->cbegin(), rowRp->cbegin(),
				  buffers->Q.begin());
	    ++rowLp;
	    ++rowRp;
	}

	if (rowL >= rowL0)
	{
	    start(2);
	    buffers->RminR = std::numeric_limits<Score>::max();
	    computeDisparities(buffers->Q.crbegin(), buffers->Q.crend(),
			       buffers->dminL.rbegin(),
			       buffers->delta.rbegin(),
			       buffers->dminR.end() - D + 1,
			       make_dummy_iterator(&(buffers->RminR)));
	    start(3);
	    selectDisparities(buffers->dminL.cbegin(), buffers->dminL.cend(),
			      buffers->dminR.cbegin(), buffers->delta.cbegin(),
			      rowD->begin() + N/2);
	    ++rowD;
	}

	++rowR;
    }

    _bufferPool.put(buffers);
    nextFrame();
}

template <class SCORE, class DISP> template <class ROW, class ROW_D> void
SADStereo<SCORE, DISP>::match(ROW rowL, ROW rowLe, ROW rowLlast,
			      ROW rowR, ROW rowV, ROW_D rowD)
{
    start(0);
    const size_t	N = _params.windowSize,
			D = _params.disparitySearchWidth,
			H = std::distance(rowL, rowLe),
			W = (H != 0 ? rowL->size() : 0);
    if (H < N || W < N)				// 充分な行数／列数があるか確認
	return;

    auto* const	buffers = _bufferPool.get();	// 各種作業領域を確保
    buffers->initialize(N, D, W, H);
    
    std::advance(rowD, N/2);	// 出力行をウィンドウサイズの半分だけ進める

    const auto	rowD0 = rowD;
    auto	v  = H;
    size_t	cV = std::distance(rowL, rowLlast);
    auto	rowLp = rowL;
    auto	rowRp = rowR;
    auto	cVp = cV;

  // 各行に対してステレオマッチングを行い視差を計算
    for (const auto rowL0 = rowL + N - 1; rowL != rowLe; ++rowL)
    {
	--v;
	--cV;
	
	start(1);
	if (rowL <= rowL0)
	    initializeDissimilarities(rowL->cbegin(), rowL->cend(),
				      make_zip_iterator(
					  std::make_tuple(
					      rowR->cbegin(),
					      make_vertical_iterator(rowV,
								     cV))),
				      buffers->Q.begin());
	else
	{
	    updateDissimilarities(rowL->cbegin(), rowL->cend(),
				  make_zip_iterator(
				      std::make_tuple(
					  rowR->cbegin(),
					  make_vertical_iterator(rowV, cV))),
				  rowLp->cbegin(),
				  make_zip_iterator(
				      std::make_tuple(
					  rowRp->cbegin(),
					  make_vertical_iterator(rowV, --cVp))),
				  buffers->Q.begin());
	    ++rowLp;
	    ++rowRp;
	}

	if (rowL >= rowL0)
	{
	    start(2);
	    buffers->RminR = std::numeric_limits<Score>::max();
	    computeDisparities(buffers->Q.crbegin(), buffers->Q.crend(),
			       buffers->dminL.rbegin(),
			       buffers->delta.rbegin(),
			       make_zip_iterator(
				   std::make_tuple(
				       buffers->dminR.end() - D + 1,
				       make_vertical_iterator(
					   buffers->dminV.end(), v))),
			       make_zip_iterator(
				   std::make_tuple(
				       make_dummy_iterator(&(buffers->RminR)),
				       buffers->RminV.rbegin())));
	    start(3);
	    selectDisparities(buffers->dminL.cbegin(), buffers->dminL.cend(),
			      buffers->dminR.cbegin(), buffers->delta.cbegin(),
			      rowD->begin() + N/2);

	    ++rowD;
	}

	++rowR;
    }

    if (_params.doVerticalBackMatch)
    {
      // 上画像からの逆方向視差探索により誤対応を除去する．マルチスレッドの
      // 場合は短冊を跨がる視差探索ができず各短冊毎に処理せねばならないので，
      // 結果はシングルスレッド時と異なる．
	start(4);
	rowD = rowD0;
	for (v = H - N + 1; v-- != 0; )
	{
	    pruneDisparities(make_vertical_iterator(buffers->dminV.cbegin(), v),
			     make_vertical_iterator(buffers->dminV.cend(),   v),
			     rowD->begin() + N/2);
	    ++rowD;
	}
    }

    _bufferPool.put(buffers);
    nextFrame();
}

template <class SCORE, class DISP>
template <class COL, class COL_RV> void
SADStereo<SCORE, DISP>::initializeDissimilarities(COL colL, COL colLe,
						  COL_RV colRV,
						  col_siterator colQ) const
{
    using left_t	= iterator_value<COL>;
#if defined(SIMD)
    using diff_t	= Diff<simd::vec<left_t> >;
#else
    using diff_t	= Diff<left_t>;
#endif
    if (_params.blend > 0)
    {
	using blend_t	= Blend<ScoreVec>;
#if defined(SIMD)
	using qiterator	= simd::cvtup_iterator<
			      assignment_iterator<
				  blend_t, subiterator<col_siterator> > >;
	using ddiff_t	= Diff<simd::vec<std::make_signed_t<left_t> > >;
#else
	using qiterator	= assignment_iterator<blend_t,
					      subiterator<col_siterator> >;
	using ddiff_t	= Diff<std::make_signed_t<left_t> >;
#endif
	while (++colL != colLe - 1)
	{
	    ++colRV;
	    ++colQ;

	    auto	P = make_zip_iterator(
				std::make_tuple(
				    boost::make_transform_iterator(
					make_col_load_iterator(colRV),
					diff_t(*colL,
					       _params.intensityDiffMax)),
				    boost::make_transform_iterator(
					make_transform_iterator2(
					    make_col_load_iterator(colRV) + 1,
					    make_col_load_iterator(colRV) - 1,
					    Minus()),
					ddiff_t(*(colL + 1) - *(colL - 1),
						_params.derivativeDiffMax))));
	    for (qiterator Q( make_assignment_iterator(
				  colQ->begin(), blend_t(_params.blend))),
			   Qe(make_assignment_iterator(
				  colQ->end(), blend_t(_params.blend)));
		 Q != Qe; ++Q, ++P)
		*Q += *P;
	}
    }
    else
    {
#if defined(SIMD)
	using qiterator	= simd::cvtup_iterator<subiterator<col_siterator> >;
#else
	using qiterator	= subiterator<col_siterator>;
#endif
	for (; colL != colLe; ++colL)
	{
	    const auto	diff = diff_t(*colL, _params.intensityDiffMax);
	    auto	in   = make_col_load_iterator(colRV);
	    
	    for (qiterator Q(colQ->begin()), Qe(colQ->end());
		 Q != Qe; ++Q, ++in)
		*Q += diff(*in);
	    
	    ++colRV;
	    ++colQ;
	}
    }
}
    
template <class SCORE, class DISP> template <class COL, class COL_RV> void
SADStereo<SCORE, DISP>::updateDissimilarities(COL colL,  COL colLe,
					      COL_RV colRV,
					      COL colLp, COL_RV colRVp,
					      col_siterator colQ) const
{
    using left_t	= iterator_value<COL>;
#if defined(SIMD)
    using diff_t	= Diff<simd::vec<left_t> >;
#else
    using diff_t	= Diff<left_t>;
#endif

    if (_params.blend > 0)
    {
	using blend_t	= Blend<ScoreVec>;
#if defined(SIMD)
	using qiterator	= simd::cvtup_iterator<
			      assignment_iterator<
				  ScoreUpdate, subiterator<col_siterator> > >;
	using ddiff_t	= Diff<simd::vec<std::make_signed_t<left_t> > >;
#else
	using qiterator	= assignment_iterator<ScoreUpdate,
					      subiterator<col_siterator> >;
	using ddiff_t	= Diff<std::make_signed_t<left_t> >;
#endif
	while (++colL != colLe - 1)
	{
	    ++colRV;
	    ++colLp;
	    ++colRVp;
	    ++colQ;

	    const Minus	minus{};
	    auto	P = make_zip_iterator(
				std::make_tuple(
				    boost::make_transform_iterator(
					make_col_load_iterator(colRV),
					diff_t(*colL,
					       _params.intensityDiffMax)),
				    boost::make_transform_iterator(
					make_transform_iterator2(
					    make_col_load_iterator(colRV) + 1,
					    make_col_load_iterator(colRV) - 1,
					    Minus()),
					ddiff_t(*(colL + 1) - *(colL - 1),
						_params.derivativeDiffMax)),
				    boost::make_transform_iterator(
					make_col_load_iterator(colRVp),
					diff_t(*colLp,
					       _params.intensityDiffMax)),
				    boost::make_transform_iterator(
					make_transform_iterator2(
					    make_col_load_iterator(colRVp) + 1,
					    make_col_load_iterator(colRVp) - 1,
					    Minus()),
					ddiff_t(*(colLp + 1) - *(colLp - 1),
						_params.derivativeDiffMax))));
	    for (qiterator Q( make_assignment_iterator(
				  colQ->begin(), ScoreUpdate(_params.blend))),
			   Qe(make_assignment_iterator(
				  colQ->end(), ScoreUpdate(_params.blend)));
		 Q != Qe; ++Q, ++P)
		*Q += *P;
	}
    }
    else
    {
#if defined(SIMD)
	using qiterator	= simd::cvtup_iterator<subiterator<col_siterator> >;
#else
	using qiterator	= subiterator<col_siterator>;
#endif
	for (; colL != colLe; ++colL)
	{
	    const auto	diff_p = diff_t(*colLp, _params.intensityDiffMax);
	    const auto	diff_n = diff_t(*colL,  _params.intensityDiffMax);
	    auto	in_p   = make_col_load_iterator(colRVp);
	    auto	in_n   = make_col_load_iterator(colRV);
	
	    for (qiterator Q(colQ->begin()), Qe(colQ->end());
		 Q != Qe; ++Q, ++in_p, ++in_n)
		*Q += (diff_n(*in_n) - diff_p(*in_p));

	    ++colRV;
	    ++colLp;
	    ++colRVp;
	    ++colQ;
	}
    }
}

template <class SCORE, class DISP> template <class DMIN_RV, class RMIN_RV> void
SADStereo<SCORE, DISP>::computeDisparities(const_reverse_col_siterator colQ,
					   const_reverse_col_siterator colQe,
					   reverse_col_diterator dminL,
					   reverse_col_fiterator delta,
					   DMIN_RV dminRV,
					   RMIN_RV RminRV) const
{
    const size_t	dsw1 = _params.disparitySearchWidth - 1;

  // 評価値を横方向に積算し，最小値を与える視差を双方向に探索する．
    for (const_reverse_col_sbox boxR(colQ, _params.windowSize), boxRe(colQe);
	 boxR != boxRe; ++boxR)
    {
#if defined(SIMD)
	using mask_type	= simd::mask_type<Disparity>;
#  if defined(WITHOUT_CVTDOWN)
	using miterator	= simd::cvtdown_mask_iterator<
			      mask_type,
			      simd::mask_iterator<
				  subiterator<const_col_siterator>,
				  subiterator<RMIN_RV> > >;
#  else
	using miterator	= simd::mask_iterator<
			      mask_type,
			      subiterator<const_col_siterator>,
			      subiterator<RMIN_RV> >;
#  endif
#else
	using miterator	= mask_iterator<subiterator<const_col_siterator>,
					subiterator<RMIN_RV> >;
#endif

	Idx<DisparityVec>	index;
	auto			dminRVt = make_col_store_iterator(--dminRV);
#if defined(SIMD) && defined(WITHOUT_CVTDOWN)
	miterator	maskRV(make_mask_iterator(boxR->cbegin(),
						  std::begin(*RminRV)));
	for (miterator maskRVe(make_mask_iterator(boxR->cend(),
						  std::end(*RminRV)));
	     maskRV != maskRVe; ++maskRV)
#else
	miterator	maskRV(boxR->cbegin(), std::begin(*RminRV));
	for (miterator maskRVe(boxR->cend(), std::end(*RminRV));
	     maskRV != maskRVe; ++maskRV)
#endif
	{
	    using dvalue_t = decayed_iterator_value<decltype(dminRVt)>;

	  //*dminRVt = select(*maskRV, index, dvalue_t(*dminRVt));
	    *dminRVt = fast_select(*maskRV, index, dvalue_t(*dminRVt));

	    ++dminRVt;
	    ++index;
	}
#if defined(SIMD) && defined(WITHOUT_CVTDOWN)
      	const int	dL = maskRV.base().dL();	// 左画像から見た視差
#else
      	const int	dL = maskRV.dL();		// 左画像から見た視差
#endif
#if defined(SIMD)
	const Score*	R  = boxR->cbegin().base();
#else
	const Score*	R  = boxR->cbegin();
#endif
	*dminL = dL;
	*delta = (dL == 0 || dL == dsw1 ? 0 :
		  0.5f * float(R[dL-1] - R[dL+1]) /
		  float(std::max(R[dL-1] - R[dL], R[dL+1] - R[dL]) + 1));
	++delta;
	++dminL;
	++RminRV;
    }
}

/************************************************************************
*  class SADStereo<SCORE, DISP>::Buffers				*
************************************************************************/
template <class SCORE, class DISP> void
SADStereo<SCORE, DISP>::Buffers::initialize(size_t N, size_t D, size_t W)
{
#if defined(SIMD)
    const size_t	DD = D / ScoreVec::size;
#else
    const size_t	DD = D;
#endif
    Q.resize(W, DD);			// Q(u, *; d)
    Q = 0;

    dminL.resize(W - N + 1);
    delta.resize(dminL.size());
    dminR.resize(dminL.size() + D - 1);
    RminR.resize(DD);
}

template <class SCORE, class DISP> void
SADStereo<SCORE, DISP>::Buffers::initialize(size_t N, size_t D,
					    size_t W, size_t H)
{
    initialize(N, D, W);
    
    dminV.resize(dminL.size(), H + D - 1);
    RminV.resize(dminL.size(), RminR.size());
    RminV = std::numeric_limits<SCORE>::max();
}

}
#endif	// !__TU_SADSTEREO_H
