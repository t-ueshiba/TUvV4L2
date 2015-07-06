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
 *  $Id: Array++.h 1785 2015-02-14 05:43:15Z ueshiba $
 */
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
    typedef Array<ScoreVec>				ScoreVecArray;
    typedef Array2<ScoreVecArray>			ScoreVecArray2;
    typedef typename ScoreVecArray2::iterator		col_siterator;
    typedef typename ScoreVecArray2::const_iterator	const_col_siterator;
    typedef typename ScoreVecArray2::const_reverse_iterator
						const_reverse_col_siterator;
    typedef box_filter_iterator<const_reverse_col_siterator>
							const_reverse_col_sbox;
    typedef Array<Disparity>				DisparityArray;
    typedef Array2<DisparityArray>			DisparityArray2;
    typedef typename DisparityArray::reverse_iterator	reverse_col_diterator;
    typedef Array<float>				FloatArray;
    typedef FloatArray::reverse_iterator		reverse_col_fiterator;

    struct ScoreUpdate
    {
	typedef ScoreVec				result_type;
	typedef boost::tuple<ScoreVec, ScoreVec,
			     ScoreVec, ScoreVec>	argument_type;
	
	ScoreUpdate(Score blend)	:_blend(blend)	{}

	result_type	operator ()(const argument_type& args) const
			{
			    using namespace	boost;

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
    ROW		rowLp = rowL, rowRp = rowR;

  // 各行に対してステレオマッチングを行い視差を計算
    for (ROW rowL0 = rowL + N - 1; rowL != rowLe; ++rowL)
    {
	start(1);
	if (rowL <= rowL0)
	    initializeDissimilarities(rowL->cbegin(), rowL->cend(),
				      rowR->cbegin(), buffers->Q.begin());
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

    Buffers*	buffers = _bufferPool.get();	// 各種作業領域を確保
    buffers->initialize(N, D, W, H);
    
    std::advance(rowD, N/2);	// 出力行をウィンドウサイズの半分だけ進める

    const ROW_D		rowD0 = rowD;
    size_t		v = H, cV = std::distance(rowL, rowLlast);
    ROW			rowLp = rowL, rowRp = rowR;
    size_t		cVp = cV;

  // 各行に対してステレオマッチングを行い視差を計算
    for (const ROW rowL0 = rowL + N - 1; rowL != rowLe; ++rowL)
    {
	--v;
	--cV;
	
	start(1);
	if (rowL <= rowL0)
	    initializeDissimilarities(rowL->cbegin(), rowL->cend(),
				      make_fast_zip_iterator(
					  boost::make_tuple(
					      rowR->cbegin(),
					      make_vertical_iterator(rowV,
								     cV))),
				      buffers->Q.begin());
	else
	{
	    updateDissimilarities(rowL->cbegin(), rowL->cend(),
				  make_fast_zip_iterator(
				      boost::make_tuple(
					  rowR->cbegin(),
					  make_vertical_iterator(rowV, cV))),
				  rowLp->cbegin(),
				  make_fast_zip_iterator(
				      boost::make_tuple(
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
			       make_fast_zip_iterator(
				   boost::make_tuple(
				       buffers->dminR.end() - D + 1,
				       make_vertical_iterator(
					   buffers->dminV.end(), v))),
			       make_fast_zip_iterator(
				   boost::make_tuple(
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
#if defined(SSE)
    typedef decltype(mm::make_load_iterator(col2ptr(colRV)))	in_iterator;
#else
    typedef decltype(col2ptr(colRV))				in_iterator;
#endif
    typedef Diff<tuple_head<iterator_value<in_iterator> > >	diff_type;

    if (_params.blend > 0)
    {
	typedef Blend<ScoreVec>					blend_type;
#if defined(SSE)
	typedef mm::cvtup_iterator<
	    assignment_iterator<blend_type,
				subiterator<col_siterator> > >	qiterator;
#else
	typedef assignment_iterator<
	    blend_type, subiterator<col_siterator> >		qiterator;
#endif
	typedef Minus<iterator_value<in_iterator> >		minus_type;
	typedef Diff<
	    tuple_head<typename minus_type::result_type> >	ddiff_type;

	while (++colL != colLe - 1)
	{
	    ++colRV;
	    ++colQ;
	
	    auto	P = make_fast_zip_iterator(
				boost::make_tuple(
				    boost::make_transform_iterator(
					in_iterator(col2ptr(colRV)),
					diff_type(*colL,
						  _params.intensityDiffMax)),
				    boost::make_transform_iterator(
					boost::make_transform_iterator(
					    make_fast_zip_iterator(
						boost::make_tuple(
						    in_iterator(
							col2ptr(colRV) + 1),
						    in_iterator(
							col2ptr(colRV) - 1))),
					    make_unarizer(minus_type())),
					ddiff_type(
					    *(colL + 1) - *(colL - 1),
					    _params.derivativeDiffMax))));
	    for (qiterator Q( make_assignment_iterator(
				  colQ->begin(), blend_type(_params.blend))),
			   Qe(make_assignment_iterator(
				  colQ->end(), blend_type(_params.blend)));
		 Q != Qe; ++Q, ++P)
		*Q += *P;
	}
    }
    else
    {
#if defined(SSE)
	typedef mm::cvtup_iterator<subiterator<col_siterator> >	qiterator;
#else
	typedef subiterator<col_siterator>			qiterator;
#endif
	for (; colL != colLe; ++colL)
	{
	    const diff_type	diff(*colL, _params.intensityDiffMax);
	    in_iterator	in(col2ptr(colRV));
	    
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
#if defined(SSE)
    typedef decltype(mm::make_load_iterator(col2ptr(colRV)))	in_iterator;
#else
    typedef decltype(col2ptr(colRV))				in_iterator;
#endif
    typedef Diff<tuple_head<iterator_value<in_iterator> > >	diff_type;

    if (_params.blend > 0)
    {
#if defined(SSE)
	typedef mm::cvtup_iterator<
	    assignment_iterator<ScoreUpdate,
				subiterator<col_siterator> > >	qiterator;
#else
	typedef assignment_iterator<
	    ScoreUpdate, subiterator<col_siterator> >		qiterator;
#endif
	typedef Minus<iterator_value<in_iterator> >		minus_type;
	typedef Diff<
	    tuple_head<typename minus_type::result_type> >	ddiff_type;

	while (++colL != colLe - 1)
	{
	    ++colRV;
	    ++colLp;
	    ++colRVp;
	    ++colQ;
	    
	    auto	P = make_fast_zip_iterator(
				boost::make_tuple(
				    boost::make_transform_iterator(
					in_iterator(col2ptr(colRV)),
					diff_type(
					    *colL, _params.intensityDiffMax)),
				    boost::make_transform_iterator(
					boost::make_transform_iterator(
					    make_fast_zip_iterator(
						boost::make_tuple(
						    in_iterator(
							col2ptr(colRV) + 1),
						    in_iterator(
							col2ptr(colRV) - 1))),
					    make_unarizer(minus_type())),
					ddiff_type(*(colL + 1) - *(colL - 1),
						   _params.derivativeDiffMax)),
				    boost::make_transform_iterator(
					in_iterator(col2ptr(colRVp)),
					diff_type(*colLp,
						  _params.intensityDiffMax)),
				    boost::make_transform_iterator(
					boost::make_transform_iterator(
					    make_fast_zip_iterator(
						boost::make_tuple(
						    in_iterator(
							col2ptr(colRVp) + 1),
						    in_iterator(
							col2ptr(colRVp) - 1))),
					    make_unarizer(minus_type())),
					ddiff_type(
					    *(colLp + 1) - *(colLp - 1),
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
#if defined(SSE)
	typedef mm::cvtup_iterator<subiterator<col_siterator> >	qiterator;
#else
	typedef subiterator<col_siterator>			qiterator;
#endif
	for (; colL != colLe; ++colL)
	{
	    const diff_type	diff_p(*colLp, _params.intensityDiffMax),
				diff_n(*colL,  _params.intensityDiffMax);
	    in_iterator		in_p(col2ptr(colRVp)), in_n(col2ptr(colRV));
	
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
#if defined(SSE)
	typedef decltype(mm::make_store_iterator(col2ptr(dminRV)))
								diterator;
#  if defined(WITHOUT_CVTDOWN)
	typedef mm::cvtdown_mask_iterator<
	    Disparity,
	    mm::mask_iterator<subiterator<const_col_siterator>,
			      subiterator<RMIN_RV> > >		miterator;
#  else
	typedef mm::mask_iterator<Disparity,
				  subiterator<const_col_siterator>,
				  subiterator<RMIN_RV> >	miterator;
#  endif
#else
	typedef decltype(col2ptr(dminRV))			diterator;
	typedef mask_iterator<subiterator<const_col_siterator>,
			      subiterator<RMIN_RV> >		miterator;
#endif
	typedef iterator_value<diterator>			dvalue_type;

	Idx<DisparityVec>	index;
	diterator		dminRVt(col2ptr(--dminRV));
#if defined(SSE) && defined(WITHOUT_CVTDOWN)
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
	  //*dminRVt = select(*maskRV, index, dvalue_type(*dminRVt));
	    *dminRVt = fast_select(*maskRV, index, dvalue_type(*dminRVt));

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
