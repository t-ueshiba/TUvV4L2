/*
 *  $Id$
 */
#ifndef __TU_SADSTEREO_H
#define __TU_SADSTEREO_H

#include "TU/StereoBase.h"
#include "TU/Array++.h"
#include "TU/BoxFilter.h"

namespace TU
{
/************************************************************************
*  class SADStereo<SCORE, DISP>						*
************************************************************************/
template <class SCORE, class DISP>
class SADStereo : public StereoBase<SADStereo<SCORE, DISP> >
{
  public:
    typedef SCORE					Score;
    typedef DISP					Disparity;

  private:
    typedef StereoBase<SADStereo<Score, Disparity> >	super;
#if defined(SSE)
    typedef mm::vec<Score>				ScoreVec;
    typedef mm::vec<Disparity>				DisparityVec;
#else
    typedef Score					ScoreVec;
    typedef Disparity					DisparityVec;
#endif
    typedef Array<ScoreVec,
		  Buf<ScoreVec,
		      typename super::template
		      Allocator<ScoreVec>::type> >	ScoreVecArray;
    typedef Array2<ScoreVecArray,
		   Buf<ScoreVec,
		       typename super::template
		       Allocator<ScoreVec>::type>,
		   Buf<ScoreVecArray,
		       typename super::template
		       Allocator<ScoreVecArray>::type> >
							ScoreVecArray2;
    typedef Array<ScoreVecArray2,
		  Buf<ScoreVecArray2,
		      typename super::template
		      Allocator<ScoreVecArray2>::type> >
							ScoreVecArray2Array;
    typedef typename ScoreVecArray2::iterator		col_siterator;
    typedef typename ScoreVecArray2::const_iterator
							const_col_siterator;
    typedef typename ScoreVecArray2::const_reverse_iterator
						const_reverse_col_siterator;
    typedef ring_iterator<typename ScoreVecArray2Array::iterator>
							ScoreVecArray2Ring;
    typedef box_filter_iterator<ScoreVecArray2Ring>	ScoreVecArray2Box;
    typedef box_filter_iterator<
		typename ScoreVecArray2::const_reverse_iterator>
							ScoreVecArrayBox;
    typedef Array<Disparity,
		  Buf<Disparity,
		      typename super::template
		      Allocator<Disparity>::type> >	DisparityArray;
    typedef Array2<DisparityArray,
		   Buf<Disparity,
		       typename super::template
		       Allocator<Disparity>::type> >	DisparityArray2;
    typedef typename DisparityArray::reverse_iterator	reverse_col_diterator;
    typedef Array<float,
		  Buf<float,
		      typename super::template
		      Allocator<float>::type> >		FloatArray;
    typedef typename FloatArray::reverse_iterator	reverse_col_fiterator;

    struct Buffers
    {
	void	initialize(size_t N, size_t D, size_t W)		;
	void	initialize(size_t N, size_t D, size_t W, size_t H)	;
	
#if defined(RING)
	ScoreVecArray2Array	P;	// (N + 1) x W x D
#else
	ScoreVecArray2		Q;	// W x D
#endif
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
	Parameters()	:windowSize(11), intensityDiffMax(20)		{}

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
			    
			    return out;
			}
			    
	size_t	windowSize;			//!< ウィンドウのサイズ
	size_t	intensityDiffMax;		//!< 輝度差の最大値
    };

  public:
    SADStereo()	:super(*this, 6), _params()				{}
    SADStereo(const Parameters& params)
	:super(*this, 6), _params(params)				{}

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
    
    template <template <class, class> class ASSIGN, class COL, class COL_RV>
    void	initializeDissimilarities(COL colL, COL colLe,
					  COL_RV colRV,
					  col_siterator colP)	const	;
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
#if defined(SSE)
    _params.disparitySearchWidth
	= mm::vec<Disparity>::ceil(_params.disparitySearchWidth);
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

    Buffers*	buffers = _bufferPool.get();	// 各種作業領域を確保
    buffers->initialize(N, D, W);

    std::advance(rowD, N/2);	// 出力行をウィンドウサイズの半分だけ進める
#if defined(RING)
    ScoreVecArray2Ring	rowP(buffers->P.begin(), buffers->P.end());
    ScoreVecArray2Box	boxQ;
#else
    ROW			rowLp = rowL, rowRp = rowR;
#endif
  // 各行に対してステレオマッチングを行い視差を計算
    for (ROW rowL0 = rowL + N - 1; rowL != rowLe; ++rowL)
    {
	start(1);
#if defined(RING)
	initializeDissimilarities<assign>(
	    rowL->cbegin(), rowL->cend(),
	    make_rvcolumn_iterator(rowR->cbegin()), rowP->begin());
	++rowP;
#else
	if (rowL <= rowL0)
	    initializeDissimilarities<plus_assign>(
		rowL->cbegin(), rowL->cend(),
		make_rvcolumn_iterator(rowR->cbegin()), buffers->Q.begin());
	else
	{
	    updateDissimilarities(rowL->cbegin(), rowL->cend(),
				  make_rvcolumn_iterator(rowR->cbegin()),
				  rowLp->cbegin(),
				  make_rvcolumn_iterator(rowRp->cbegin()),
				  buffers->Q.begin());
	    ++rowLp;
	    ++rowRp;
	}
#endif
	if (rowL >= rowL0)
	{
#if defined(RING)
	    start(2);
	    if (rowL == rowL0)
		boxQ.initialize(rowP - N, N);
	    const ScoreVecArray2&	Q = *boxQ;
	    ++boxQ;
#else
	    const ScoreVecArray2&	Q = buffers->Q;
#endif
	    start(3);
	    buffers->RminR.fill(std::numeric_limits<Score>::max());
	    computeDisparities(Q.crbegin(), Q.crend(),
			       buffers->dminL.rbegin(),
			       buffers->delta.rbegin(),
			       make_rvcolumn_iterator(
				   buffers->dminR.end() - D + 1),
			       make_dummy_iterator(&(buffers->RminR)));
	    start(4);
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

    Buffers*	buffers = _bufferPool.get();	// 各種作業領域を確保
    buffers->initialize(N, D, W, H);
    
    std::advance(rowD, N/2);	// 出力行をウィンドウサイズの半分だけ進める

    const ROW_D		rowD0 = rowD;
    size_t		v = H, cV = std::distance(rowL, rowLlast);
#if defined(RING)
    ScoreVecArray2Ring	rowP(buffers->P.begin(), buffers->P.end());
    ScoreVecArray2Box	boxQ;
#else
    ROW			rowLp = rowL, rowRp = rowR;
    size_t		cVp = cV;
#endif
  // 各行に対してステレオマッチングを行い視差を計算
    for (const ROW rowL0 = rowL + N - 1; rowL != rowLe; ++rowL)
    {
	--v;
	--cV;
	
	start(1);
#if defined(RING)
	initializeDissimilarities<assign>(
	    rowL->cbegin(), rowL->cend(),
	    make_rvcolumn_iterator(
		make_fast_zip_iterator(
		    std::make_tuple(rowR->cbegin(),
				    make_vertical_iterator(rowV, cV)))),
	    rowP->begin());
	++rowP;
#else
	if (rowL <= rowL0)
	    initializeDissimilarities<plus_assign>(
		rowL->cbegin(), rowL->cend(),
		make_rvcolumn_iterator(
		    make_fast_zip_iterator(
			std::make_tuple(rowR->cbegin(),
					make_vertical_iterator(rowV, cV)))),
		buffers->Q.begin());
	else
	{
	    updateDissimilarities(rowL->cbegin(), rowL->cend(),
				  make_rvcolumn_iterator(
				      make_fast_zip_iterator(
					  std::make_tuple(
					      rowR->cbegin(),
					      make_vertical_iterator(rowV,
								     cV)))),
				  rowLp->cbegin(),
				  make_rvcolumn_iterator(
				      make_fast_zip_iterator(
					  std::make_tuple(
					      rowRp->cbegin(),
					      make_vertical_iterator(rowV,
								     --cVp)))),
				  buffers->Q.begin());
	    ++rowLp;
	    ++rowRp;
	}
#endif
	if (rowL >= rowL0)
	{
#if defined(RING)
	    start(2);
	    if (rowL == rowL0)
		boxQ.initialize(rowP - N, N);
	    const ScoreVecArray2&	Q = *boxQ;
	    ++boxQ;
#else
	    const ScoreVecArray2&	Q = buffers->Q;
#endif
	    start(3);
	    buffers->RminR.fill(std::numeric_limits<Score>::max());
	    computeDisparities(Q.crbegin(), Q.crend(),
			       buffers->dminL.rbegin(),
			       buffers->delta.rbegin(),
			       make_rvcolumn_iterator(
				   make_fast_zip_iterator(
				       std::make_tuple(
					   buffers->dminR.end() - D + 1,
					   make_vertical_iterator(
					       buffers->dminV.end(), v)))),
			       make_row_iterator(
				   make_fast_zip_iterator(
				       std::make_tuple(
					   make_dummy_iterator(
					       &(buffers->RminR)),
					   buffers->RminV.rbegin()))));
	    start(4);
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
	start(5);
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
template <template <class, class> class ASSIGN, class COL, class COL_RV> void
SADStereo<SCORE, DISP>::initializeDissimilarities(COL colL, COL colLe,
						  COL_RV colRV,
						  col_siterator colP) const
{
#if defined(SSE)
    typedef mm::load_iterator<typename iterator_value<COL_RV>::iterator>
								in_iterator;
    typedef mm::cvtup_iterator<typename ScoreVecArray::iterator>
								qiterator;
#else
    typedef typename iterator_value<COL_RV>::iterator		in_iterator;
    typedef typename ScoreVecArray::iterator			qiterator;
#endif
    typedef Diff<iterator_value<in_iterator> >			diff_type;

    for (; colL != colLe; ++colL)
    {
	using namespace	std::placeholders;
	
	auto	P = boost::make_transform_iterator(
			in_iterator(colRV->begin()),
			std::bind(diff_type(_params.intensityDiffMax),
				  *colL, _1));
	for (qiterator Q(colP->begin()), Qe(colP->end()); Q != Qe; ++Q, ++P)
	    exec_assignment<ASSIGN>(*P, *Q);
	
	++colRV;
	++colP;
    }
}
    
template <class SCORE, class DISP> template <class COL, class COL_RV> void
SADStereo<SCORE, DISP>::updateDissimilarities(COL colL,  COL colLe,
					      COL_RV colRV,
					      COL colLp, COL_RV colRVp,
					      col_siterator colQ) const
{
#if defined(SSE)
    typedef mm::load_iterator<typename iterator_value<COL_RV>::iterator>
								in_iterator;
    typedef mm::cvtup_iterator<typename ScoreVecArray::iterator>
								qiterator;
#else
    typedef typename iterator_value<COL_RV>::iterator		in_iterator;
    typedef typename ScoreVecArray::iterator			qiterator;
#endif
    typedef Diff<iterator_value<in_iterator> >			diff_type;

    for (; colL != colLe; ++colL)
    {
	using namespace	std::placeholders;

	auto	Pp = boost::make_transform_iterator(
			 in_iterator(colRVp->begin()),
			 std::bind(diff_type(_params.intensityDiffMax),
				   *colLp, _1));
	auto	Pn = boost::make_transform_iterator(
			 in_iterator(colRV->begin()),
			 std::bind(diff_type(_params.intensityDiffMax),
				   *colL, _1));
	for (qiterator Q(colQ->begin()), Qe(colQ->end());
	     Q != Qe; ++Q, ++Pp, ++Pn)
	    *Q += (*Pn - *Pp);

	++colRV;
	++colLp;
	++colRVp;
	++colQ;
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
    for (ScoreVecArrayBox boxR(colQ, _params.windowSize), boxRe(colQe);
	 boxR != boxRe; ++boxR)
    {
#if defined(SSE)
	typedef mm::store_iterator<
	    typename iterator_value<DMIN_RV>::iterator>		diterator;
#  if defined(WITHOUT_CVTDOWN)
	typedef mm::cvtdown_mask_iterator<
	    Disparity,
	    mm::mask_iterator<
		typename ScoreVecArray::const_iterator,
		typename iterator_value<RMIN_RV>::iterator> >	miterator;
#  else
	typedef mm::mask_iterator<
	    Disparity,
	    typename ScoreVecArray::const_iterator,
	    typename iterator_value<RMIN_RV>::iterator>		miterator;
#  endif
#else
	typedef typename iterator_value<DMIN_RV>::iterator	diterator;
	typedef mask_iterator<
	    typename ScoreVecArray::const_iterator,
	    typename iterator_value<RMIN_RV>::iterator>		miterator;
#endif
	typedef iterator_value<diterator>			dvalue_type;

	Idx<DisparityVec>	index;
	diterator		dminRVt((--dminRV)->begin());
#if defined(SSE) && defined(WITHOUT_CVTDOWN)
	miterator	maskRV(make_mask_iterator(boxR->cbegin(),
						  RminRV->begin()));
	for (miterator maskRVe(make_mask_iterator(boxR->cend(),
						  RminRV->end()));
	     maskRV != maskRVe; ++maskRV)
#else
	miterator	maskRV(boxR->cbegin(), RminRV->begin());
	for (miterator maskRVe(boxR->cend(), RminRV->end());
	     maskRV != maskRVe; ++maskRV)
#endif
	{
	    *dminRVt = select(*maskRV, index, dvalue_type(*dminRVt));

	    ++dminRVt;
	    ++index;
	}
#if defined(SSE) && defined(WITHOUT_CVTDOWN)
      	const int	dL = maskRV.base().dL();	// 左画像から見た視差
#else
      	const int	dL = maskRV.dL();		// 左画像から見た視差
#endif
#if defined(SSE)
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
#if defined(SSE)
    const size_t	DD = D / ScoreVec::size;
#else
    const size_t	DD = D;
#endif
#if defined(RING)
    P.resize(N + 1);
    for (row_siterator rowP = P.begin(); rowP != P.end(); ++rowP)
	if (!rowP->resize(W, DD))
	    break;
#else
    Q.resize(W, DD);			// Q(u, *; d)
    Q.fill(0);
#endif

    if (dminL.resize(W - N + 1))
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
    RminV.fill(std::numeric_limits<SCORE>::max());
}

}
#endif	// !__TU_SADSTEREO_H
