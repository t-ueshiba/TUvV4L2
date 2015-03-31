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
  \file		GFStereo.h
  \brief	Guided Filterステレオマッチングクラスの定義と実装
*/
#ifndef __TU_GFSTEREO_H
#define __TU_GFSTEREO_H

#include "TU/StereoBase.h"
#include "TU/Array++.h"
#include "TU/BoxFilter.h"

namespace TU
{
/************************************************************************
*  class GFStereo<SCORE, DISP>						*
************************************************************************/
template <class SCORE, class DISP>
class GFStereo : public StereoBase<GFStereo<SCORE, DISP> >
{
  public:
    typedef SCORE					Score;
    typedef DISP					Disparity;

  private:
    typedef StereoBase<GFStereo<Score, Disparity> >	super;
#if defined(SSE)
    typedef mm::vec<Score>				ScoreVec;
    typedef mm::vec<Disparity>				DisparityVec;
#else
    typedef Score					ScoreVec;
    typedef Disparity					DisparityVec;
#endif
    typedef boost::tuple<ScoreVec, ScoreVec>		ScoreVecTuple;

    class ScoreVecArray : public Array<ScoreVec>
    {
      private:
	typedef Array<ScoreVec>				array_t;
	    
      public:
	typedef typename array_t::iterator		iterator;
	typedef typename array_t::const_iterator	const_iterator;
	typedef typename array_t::reference		reference;
	
	class const_iterator2
	    : public boost::iterator_adaptor<const_iterator2,
					     const_iterator,
					     ScoreVecTuple,
					     boost::use_default,
					     ScoreVecTuple>
	{
	  private:
	    typedef boost::iterator_adaptor<const_iterator2,
					    const_iterator,
					    ScoreVecTuple,
					    boost::use_default,
					    ScoreVecTuple>	super;

	  public:
	    typedef typename super::difference_type	difference_type;
	    typedef typename super::reference		reference;

	    friend class	boost::iterator_core_access;
	    
	  public:
	    const_iterator2(const_iterator iter)	:super(iter)	{}

	  private:
	    reference		dereference() const
				{
				    return reference(*super::base(),
						     *(super::base() + 1));
				}
	    void		advance(difference_type n)
				{
				    super::base_reference() += (2*n);
				}
	    void		increment()
				{
				    super::base_reference() += 2;
				}
	    void		decrement()
				{
				    super::base_reference() -= 2;
				}
	    difference_type	distance_to(const_iterator2 iter) const
				{
				    return (iter.base() - super::base()) / 2;
				}
	};

	class iterator2_proxy
	{
	  public:
	    iterator2_proxy(const iterator& iter)	:_iter(iter)	{}

	    void		operator =(ScoreVecTuple x) const
				{
				    *_iter	 = boost::get<0>(x);
				    *(_iter + 1) = boost::get<1>(x);
				}
	    void		operator +=(ScoreVecTuple x) const
				{
				    *_iter	 += boost::get<0>(x);
				    *(_iter + 1) += boost::get<1>(x);
				}
		
	  private:
	    const iterator&	_iter;
	};

	class iterator2 : public boost::iterator_adaptor<iterator2,
							 iterator,
							 ScoreVecTuple,
							 boost::use_default,
							 iterator2_proxy>
	{
	  private:
	    typedef boost::iterator_adaptor<iterator2,
					    iterator,
					    ScoreVecTuple,
					    boost::use_default,
					    iterator2_proxy>	super;

	  public:
	    typedef typename super::difference_type	difference_type;
	    typedef typename super::reference		reference;

	    friend class	boost::iterator_core_access;
	    
	  public:
	    iterator2(iterator iter)	:super(iter)			{}

	  private:
	    reference		dereference() const
				{
				    return reference(super::base());
				}
	    void		advance(difference_type n)
				{
				    super::base_reference() += (2*n);
				}
	    void		increment()
				{
				    super::base_reference() += 2;
				}
	    void		decrement()
				{
				    super::base_reference() -= 2;
				}
	    difference_type	distance_to(iterator2 iter) const
				{
				    return (iter.base() - super::base()) / 2;
				}
	};

      public:
	ScoreVecArray()				:array_t()		{}
	explicit ScoreVecArray(size_t d)	:array_t(d)		{}
	
	using		array_t::operator =;
	using		array_t::begin;
	using		array_t::cbegin;
	using		array_t::end;
	using		array_t::cend;
	
	    
	const_iterator2	cbegin2() const
			{
			    return const_iterator2(cbegin());
			}
	const_iterator2	cend2() const
			{
			    return const_iterator2(cend());
			}
	iterator2	begin2()
			{
			    return iterator2(begin());
			}
	iterator2	end2()
			{
			    return iterator2(end());
			}
    };

    struct GuideElement : public boost::additive<GuideElement>
    {
	GuideElement()	:g_sum(0), g_sqsum(0)				{}

	GuideElement&	operator +=(const GuideElement& x)
			{
			    g_sum   += x.g_sum;
			    g_sqsum += x.g_sqsum;
			    return *this;
			}
	GuideElement&	operator -=(const GuideElement& x)
			{
			    g_sum   -= x.g_sum;
			    g_sqsum -= x.g_sqsum;
			    return *this;
			}
	
	Score	g_sum;		//!< ガイド画素の和
	Score	g_sqsum;	//!< ガイド画素の二乗和
    };

    class ParamInit
    {
      public:
	typedef ScoreVec	argument_type;
	typedef ScoreVecTuple	result_type;

      public:
	ParamInit(Score g)	:_g(g)					{}

	result_type	operator ()(argument_type p) const
			{
			    return result_type(p, _g * p);
			}

      private:
	const ScoreVec	_g;
    };

    class ParamUpdate
    {
      public:
	typedef ScoreVecTuple	argument_type;
	typedef ScoreVecTuple	result_type;

      public:
	ParamUpdate(Score gn, Score gp)	:_gn(gn), _gp(gp)		{}

	result_type	operator ()(const argument_type& p) const
			{
			    using namespace	boost;
			    
			    return result_type(get<0>(p) - get<1>(p),
					       _gn*get<0>(p) - _gp*get<1>(p));
			}

      private:
	const ScoreVec	_gn;
	const ScoreVec	_gp;
    };

    class CoeffInit
    {
      public:
	ScoreVecTuple	argument_type;
	typedef ScoreVecTuple	result_type;
	
      public:
	CoeffInit(Score g_avg, Score g_sqavg, Score e)
	    :_g_avg(g_avg), _g_rvar(1/(g_sqavg - g_avg*g_avg + e*e))	{}

	result_type	operator ()(boost::tuple<const ScoreVec&,
						 const ScoreVec&> params) const
			{
			    using namespace	boost;
			    
			    ScoreVec	a = (get<1>(params) -
					     get<0>(params)*_g_avg) * _g_rvar;
			    return result_type(a, get<0>(params) - a*_g_avg);
			}

      private:
	const ScoreVec	_g_avg;
	const ScoreVec	_g_rvar;
    };

    class CoeffTrans
    {
      public:
	typedef ScoreVecTuple	argument_type;
	typedef ScoreVec	result_type;
	
      public:
	CoeffTrans(Score g) :_g(g)					{}

	result_type	operator ()(boost::tuple<const ScoreVec&,
						 const ScoreVec&> coeffs) const
			{
			    return (boost::get<0>(coeffs) * _g +
				    boost::get<1>(coeffs));
			}
	
      private:
	const ScoreVec	_g;
    };

    typedef Array2<ScoreVecArray>			ScoreVecArray2;
    typedef Array<ScoreVecArray2>			ScoreVecArray2Array;
    typedef typename ScoreVecArray2::iterator		col_siterator;
    typedef typename ScoreVecArray2::const_iterator
							const_col_siterator;
    typedef typename ScoreVecArray2::const_reverse_iterator
						const_reverse_col_siterator;
    typedef typename ScoreVecArray2Array::iterator	row_siterator;
    typedef typename ScoreVecArray2Array::const_iterator
							const_row_siterator;
    typedef ring_iterator<row_siterator>		row_sring;
    typedef box_filter_iterator<row_sring>		row_sbox;
    typedef box_filter_iterator<const_col_siterator>	const_col_sbox;
    typedef box_filter_iterator<const_reverse_col_siterator>
							const_reverse_col_sbox;

    typedef Array<GuideElement>				GuideArray;
    typedef Array2<GuideArray>				GuideArray2;
    typedef typename GuideArray::iterator		col_giterator;
    typedef typename GuideArray::const_iterator		const_col_giterator;
    typedef box_filter_iterator<const_col_giterator>	const_col_gbox;

    typedef Array<Disparity>				DisparityArray;
    typedef Array2<DisparityArray>			DisparityArray2;
    typedef typename DisparityArray::reverse_iterator	reverse_col_diterator;

    typedef Array<float>				FloatArray;
    typedef FloatArray::reverse_iterator		reverse_col_fiterator;
    
    struct Buffers
    {
	void	initialize(size_t N, size_t D, size_t W)		;
	void	initialize(size_t N, size_t D, size_t W, size_t H)	;
	
	ScoreVecArray2		Q;	// W x 2D
	GuideArray		F;	// 1 x W
	ScoreVecArray2Array	A;
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
	Parameters()	:windowSize(11),
			 intensityDiffMax(20), epsilon(150)		{}

	std::istream&	get(std::istream& in)
			{
			    super::Parameters::get(in);
			    return in >> windowSize >> intensityDiffMax;
			}
	std::ostream&	put(std::ostream& out) const
			{
			    using namespace	std;

			    super::Parameters::put(out);
			    cerr << "  window size:                        ";
			    out << windowSize << endl;
			    cerr << "  maximum intensity difference:       ";
			    out << intensityDiffMax << endl;
			    cerr << "  epsilon for guided filtering:       ";
			    return out << epsilon << endl;
			}
			    
	size_t	windowSize;		//!< ウィンドウのサイズ
	size_t	intensityDiffMax;	//!< 輝度差の最大値
	Score	epsilon;		//!< guided filterの正則化パラメータ
    };

  public:
    GFStereo()	:super(*this, 7), _params()				{}
    GFStereo(const Parameters& params)
	:super(*this, 7), _params(params)				{}

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
    void	initializeFilterParameters(COL colL, COL colLe,
					   COL_RV colRV,
					   col_siterator colQ,
					   col_giterator colF)	  const	;
    template <class COL, class COL_RV>
    void	updateFilterParameters(COL colL, COL colLe, COL_RV colRV,
				       COL colLp, COL_RV colRVp,
				       col_siterator colQ,
				       col_giterator colF)	  const	;
    void	initializeFilterCoefficients(const_col_siterator colQ,
					     const_col_siterator colQe,
					     const_col_giterator colF,
					     col_siterator colA)  const	;
    template <class COL, class DMIN_RV, class RMIN_RV>
    void	computeDisparities(const_reverse_col_siterator colB,
				   const_reverse_col_siterator colBe,
				   COL colG,
				   reverse_col_diterator dminL,
				   reverse_col_fiterator delta,
				   DMIN_RV dminRV, RMIN_RV RminRV) const;

  private:
    Parameters					_params;
    typename super::template Pool<Buffers>	_bufferPool;
};

template <class SCORE, class DISP>
inline const typename GFStereo<SCORE, DISP>::Parameters&
GFStereo<SCORE, DISP>::getParameters() const
{
    return _params;
}
    
template <class SCORE, class DISP> inline void
GFStereo<SCORE, DISP>::setParameters(const Parameters& params)
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
GFStereo<SCORE, DISP>::getOverlap() const
{
    return 2*_params.windowSize - 2;
}
    
template <class SCORE, class DISP> template <class ROW, class ROW_D> void
GFStereo<SCORE, DISP>::match(ROW rowL, ROW rowLe, ROW rowR, ROW_D rowD)
{
    start(0);
    const size_t	H = std::distance(rowL, rowLe),
			W = (H != 0 ? rowL->size() : 0),
			N = _params.windowSize,
			D = _params.disparitySearchWidth;
    if (H < 2*N || W < 2*N)			// 充分な行数／列数があるか確認
	return;
    
    Buffers*	buffers = _bufferPool.get();	// 各種作業領域を確保
    buffers->initialize(N, D, W);
    
    ROW		rowLp = rowL, rowRp = rowR;
    row_sring	rowA(buffers->A.begin(), buffers->A.end());
    row_sbox	boxB;
    ROW		rowG  = rowL;
    const ROW	rowL0 = rowL + N - 1, rowL1 = rowL0 + N - 1;

    for (; rowL != rowLe; ++rowL)
    {
	start(1);
      // 各左画素に対して視差[0, D)の右画素のそれぞれとの間の相違度を計算し，
      // フィルタパラメータ(= 縦横両方向に積算された相違度(コスト)の総和
      // および画素毎のコストとガイド画素の積和)を初期化
	if (rowL <= rowL0)
	    initializeFilterParameters(rowL->cbegin(), rowL->cend(),
				       rowR->cbegin(),
				       buffers->Q.begin(), buffers->F.begin());
	else
	{
	    updateFilterParameters(rowL->cbegin(), rowL->cend(), rowR->cbegin(),
				   rowLp->cbegin(), rowRp->cbegin(),
				   buffers->Q.begin(), buffers->F.begin());
	    ++rowLp;
	    ++rowRp;
	}

	if (rowL >= rowL0)	// 最初のN行に対してコストPが計算済みならば...
	{
	    start(2);
	  // さらにコストを横方向に積算してフィルタパラメータを計算し，
	  // それを用いてフィルタ係数を初期化
	    initializeFilterCoefficients(buffers->Q.cbegin(), buffers->Q.cend(),
					 buffers->F.cbegin(), rowA->begin());
	    ++rowA;

	    if (rowL >= rowL1)		// rowL0からN行分のフィルタ係数が
	    {				// 計算済みならば...
		start(3);
		if (rowL == rowL1)
		    boxB.initialize(rowA - N, N);
		const auto&	B = *boxB;
		
		start(4);
	      // さらにフィルタ係数を横方向に積算して最終的な係数を求め，
	      // それにguide画像を適用してウィンドウコストを求め，それを
	      // 用いてそれぞれ左/右/上画像を基準とした最適視差を計算
	  	buffers->RminR.fill(std::numeric_limits<Score>::max());
		computeDisparities(B.crbegin(), B.crend(),
				   rowG->crbegin() + N - 1,
				   buffers->dminL.rbegin(),
				   buffers->delta.rbegin(),
				   buffers->dminR.end() - D + 1,
				   make_dummy_iterator(&(buffers->RminR)));
		++boxB;
		
		start(5);
	      // 左/右基準視差が一致する場合のみ，それをサブピクセル補間して
	      // 視差として書き出す
		selectDisparities(buffers->dminL.cbegin(),
				  buffers->dminL.cend(),
				  buffers->dminR.cbegin(),
				  buffers->delta.cbegin(),
				  rowD->begin() + N - 1);
	    }
	    
	    ++rowG;	// guide画像と視差画像は左画像よりもN-1行だけ遅れる
	    ++rowD;	// 同上
	}

	++rowR;
    }

    _bufferPool.put(buffers);
    nextFrame();
}
    
template <class SCORE, class DISP> template <class ROW, class ROW_D> void
GFStereo<SCORE, DISP>::match(ROW rowL, ROW rowLe, ROW rowLlast,
			     ROW rowR, ROW rowV, ROW_D rowD)
{
    start(0);
    const size_t	H = std::distance(rowL, rowLe),
			W = (H != 0 ? rowL->size() : 0),
			N = _params.windowSize,
			D = _params.disparitySearchWidth;
    if (H < 2*N || W < 2*N)			// 充分な行数／列数があるか確認
	return;

    Buffers*	buffers = _bufferPool.get();
    buffers->initialize(N, D, W, H);		// 各種作業領域を確保

    size_t	v = H, cV = std::distance(rowL, rowLlast);
    ROW		rowLp = rowL, rowRp = rowR;
    size_t	cVp = cV;
    row_sring	rowA(buffers->A.begin(), buffers->A.end());
    row_sbox	boxB;
    const ROW_D	rowD0 = rowD + N - 1;
    ROW		rowG  = rowL;
    const ROW 	rowL0 = rowL + N - 1, rowL1 = rowL0 + N - 1;

    for (; rowL != rowLe; ++rowL)
    {
	--v;
	--cV;
	
	start(1);
      // 各左画素に対して視差[0, D)の右画素のそれぞれとの間の相違度を計算し，
      // フィルタパラメータ(= 縦横両方向に積算された相違度(コスト)の総和
      // および画素毎のコストとガイド画素の積和)を初期化
	if (rowL <= rowL0)
	    initializeFilterParameters(rowL->cbegin(), rowL->cend(),
				       make_fast_zip_iterator(
					   boost::make_tuple(
					       rowR->cbegin(),
					       make_vertical_iterator(rowV,
								      cV))),
				       buffers->Q.begin(), buffers->F.begin());
	else
	{
	    updateFilterParameters(rowL->cbegin(), rowL->cend(),
				   make_fast_zip_iterator(
				       boost::make_tuple(
					   rowR->cbegin(),
					   make_vertical_iterator(rowV, cV))),
				   rowLp->cbegin(),
				   make_fast_zip_iterator(
				       boost::make_tuple(
					   rowRp->cbegin(),
					   make_vertical_iterator(rowV,
								  --cVp))),
				   buffers->Q.begin(), buffers->F.begin());
	    ++rowLp;
	    ++rowRp;
	}

	if (rowL >= rowL0)	// 最初のN行に対してコストPが計算済みならば...
	{
	    start(2);
	  // さらにコストを横方向に積算してフィルタパラメータを計算し，
	  // それを用いてフィルタ係数を初期化
	    initializeFilterCoefficients(buffers->Q.cbegin(), buffers->Q.cend(),
					 buffers->F.cbegin(), rowA->begin());
	    ++rowA;

	    if (rowL >= rowL1)		// rowL0からN行分のフィルタ係数が
	    {				// 計算済みならば...
		start(3);
	      // フィルタ係数を縦方向に積算
		if (rowL == rowL1)
		    boxB.initialize(rowA - N, N);
		const auto&	B = *boxB;
		
		start(4);
	      // さらにフィルタ係数を横方向に積算して最終的な係数を求め，
	      // それにguide画像を適用してウィンドウコストを求め，それを
	      // 用いてそれぞれ左/右/上画像を基準とした最適視差を計算
		buffers->RminR.fill(std::numeric_limits<Score>::max());
		computeDisparities(B.rbegin(), B.crend(),
				   rowG->crbegin() + N - 1,
				   buffers->dminL.rbegin(),
				   buffers->delta.rbegin(),
				   make_fast_zip_iterator(
				       boost::make_tuple(
					   buffers->dminR.end() - D + 1,
					   make_vertical_iterator(
					       buffers->dminV.end(), v))),
				   make_fast_zip_iterator(
				       boost::make_tuple(
					   make_dummy_iterator(
					       &(buffers->RminR)),
					   buffers->RminV.rbegin())));
		++boxB;

		start(5);
	      // 左/右基準視差が一致する場合のみ，それをサブピクセル補間して
	      // 視差として書き出す
		selectDisparities(buffers->dminL.cbegin(),
				  buffers->dminL.cend(),
				  buffers->dminR.cbegin(),
				  buffers->delta.cbegin(),
				  rowD->begin() + N - 1);
	    }

	    ++rowG;	// guide画像と視差画像は左画像よりもN-1行だけ遅れる
	    ++rowD;	// 同上
	}

	++rowR;
    }

    if (_params.doVerticalBackMatch)
    {
	start(6);
	rowD = rowD0;
	for (v = H - 2*(N - 1); v-- != 0; )
	{
	    pruneDisparities(make_vertical_iterator(buffers->dminV.cbegin(), v),
			     make_vertical_iterator(buffers->dminV.cend(),   v),
			     rowD->begin() + N - 1);
	    ++rowD;
	}
    }
    
    _bufferPool.put(buffers);
    nextFrame();
}

template <class SCORE, class DISP>
template <class COL, class COL_RV> void
GFStereo<SCORE, DISP>::initializeFilterParameters(COL colL, COL colLe,
						  COL_RV colRV,
						  col_siterator colQ,
						  col_giterator colF) const
{
#if defined(SSE)
    typedef decltype(mm::make_load_iterator(col2ptr(colRV)))	in_iterator;
    typedef mm::cvtup_iterator<
		assignment_iterator<
		    ParamInit,
		    typename ScoreVecArray::iterator2> >	qiterator;
#else
    typedef decltype(col2ptr(colRV))				in_iterator;
    typedef assignment_iterator<
		ParamInit, typename ScoreVecArray::iterator2>	qiterator;
#endif
    typedef Diff<tuple_head<iterator_value<in_iterator> > >	diff_type;

    for (; colL != colLe; ++colL)
    {
	const Score	pixL = *colL;
	const diff_type	diff(pixL, _params.intensityDiffMax);
	in_iterator	in(col2ptr(colRV));
	
	for (qiterator Q( make_assignment_iterator(colQ->begin2(),
						   ParamInit(pixL))),
		       Qe(make_assignment_iterator(colQ->end2(),
						   ParamInit(pixL)));
	     Q != Qe; ++Q, ++in)
	    *Q += diff(*in);

	colF->g_sum   += pixL;
	colF->g_sqsum += pixL * pixL;
	
	++colRV;
	++colQ;
	++colF;
    }
}

template <class SCORE, class DISP> template <class COL, class COL_RV> void
GFStereo<SCORE, DISP>::updateFilterParameters(COL colL, COL colLe, COL_RV colRV,
					      COL colLp, COL_RV colRVp,
					      col_siterator colQ,
					      col_giterator colF) const
{
#if defined(SSE)
    typedef decltype(mm::make_load_iterator(col2ptr(colRV)))	in_iterator;
    typedef mm::cvtup_iterator<
		assignment_iterator<
		    ParamUpdate,
		    typename ScoreVecArray::iterator2> >	qiterator;
#else
    typedef decltype(col2ptr(colRV))				in_iterator;
    typedef assignment_iterator<
		ParamUpdate, typename ScoreVecArray::iterator2>	qiterator;
#endif
    typedef Diff<tuple_head<iterator_value<in_iterator> > >	diff_type;

    for (; colL != colLe; ++colL)
    {
	const Score	pixLp = *colLp, pixL = *colL;
	const diff_type	diff_p(pixLp, _params.intensityDiffMax),
			diff_n(pixL,  _params.intensityDiffMax);
	in_iterator	in_p(col2ptr(colRVp)), in_n(col2ptr(colRV));
	
	for (qiterator Q( make_assignment_iterator(colQ->begin2(),
						   ParamUpdate(pixL, pixLp))),
		       Qe(make_assignment_iterator(colQ->end2(),
						   ParamUpdate(pixL, pixLp)));
	     Q != Qe; ++Q, ++in_p, ++in_n)
	    *Q += boost::make_tuple(diff_n(*in_n), diff_p(*in_p));

	colF->g_sum   += (pixL - pixLp);
	colF->g_sqsum += (pixL * pixL - pixLp * pixLp);

	++colRV;
	++colLp;
	++colRVp;
	++colQ;
	++colF;
    }
}

template <class SCORE, class DISP> void
GFStereo<SCORE, DISP>::initializeFilterCoefficients(const_col_siterator colQ,
						    const_col_siterator colQe,
						    const_col_giterator colF,
						    col_siterator colA) const
{
    const size_t	n = _params.windowSize * _params.windowSize;

  // 縦方向に積算したParamsを横方向に積算し，Coeffを初期化する．
    const_col_gbox	boxG(colF, _params.windowSize);
    for (const_col_sbox boxR(colQ, _params.windowSize), boxRe(colQe);
	 boxR != boxRe; ++boxR)
    {
	std::transform(boxR->cbegin2(), boxR->cend2(), colA->begin2(),
		       CoeffInit(boxG->g_sum/n, boxG->g_sqsum/n,
				 _params.epsilon));
	++boxG;
	++colA;
    }
}

template <class SCORE, class DISP>
template <class COL, class DMIN_RV, class RMIN_RV> void
GFStereo<SCORE, DISP>::computeDisparities(const_reverse_col_siterator colB,
					  const_reverse_col_siterator colBe,
					  COL colG,
					  reverse_col_diterator dminL,
					  reverse_col_fiterator delta,
					  DMIN_RV dminRV, RMIN_RV RminRV) const
{
    const size_t	n    = _params.windowSize * _params.windowSize;
    const size_t	dsw1 = _params.disparitySearchWidth - 1;
    ScoreVecArray	R(colB->size()/2);
    
  // 評価値を横方向に積算し，最小値を与える視差を双方向に探索する．
    for (const_reverse_col_sbox boxC(colB, _params.windowSize), boxCe(colBe);
	 boxC != boxCe; ++boxC)
    {
	std::transform(boxC->cbegin2(), boxC->cend2(),
		       R.begin(), CoeffTrans(*colG));
	++colG;

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
	miterator	maskRV(make_mask_iterator(R.cbegin(),
						  std::begin(*RminRV)));
	for (miterator maskRVe(make_mask_iterator(R.cend(),
						  std::end(*RminRV)));
	     maskRV != maskRVe; ++maskRV)
#else
	miterator	maskRV(R.cbegin(), std::begin(*RminRV));
	for (miterator maskRVe(R.cend(),   std::end(*RminRV));
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
	const Score*	Rb = R.cbegin().base();
#else
	const Score*	Rb = R.cbegin();
#endif
	*dminL = dL;
	*delta = (dL == 0 || dL == dsw1 ? 0 :
		  0.5f * float(Rb[dL-1] - Rb[dL+1]) /
		  float(std::max(Rb[dL-1] - Rb[dL], Rb[dL+1] - Rb[dL]) + 1));
	++delta;
	++dminL;
	++RminRV;
    }
}

/************************************************************************
*  class GFStereo<SCORE, DISP>::Buffers					*
************************************************************************/
template <class SCORE, class DISP> void
GFStereo<SCORE, DISP>::Buffers::initialize(size_t N, size_t D, size_t W)
{
#if defined(SSE)
    const size_t	DD = D / ScoreVec::size;
#else
    const size_t	DD = D;
#endif
    Q.resize(W, 2*DD);			// Q(u, *; d)
    Q.fill(0);
    F.resize(W);
    F.fill(GuideElement());

    A.resize(N + 1);
    for (auto& rowA : A)
	if (!rowA.resize(W - N + 1, 2*DD))
	    break;

    if (dminL.resize(W - 2*N + 2))
	delta.resize(dminL.size());
    dminR.resize(dminL.size() + D - 1);
    RminR.resize(DD);
}

template <class SCORE, class DISP> void
GFStereo<SCORE, DISP>::Buffers::initialize(size_t N, size_t D,
					   size_t W, size_t H)
{
    initialize(N, D, W);

    dminV.resize(dminL.size(), H + D - 1);
    RminV.resize(dminL.size(), RminR.size());
    RminV.fill(std::numeric_limits<SCORE>::max());
}

}
#endif	// !__TU_GFSTEREO_H
