/*
 *  $Id$
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

    class ScoreVecArray
	: public Array<ScoreVec,
		       Buf<ScoreVec,
			   typename super::template
			   Allocator<ScoreVec>::type> >
    {
      private:
	typedef Array<ScoreVec,
		      Buf<ScoreVec,
			  typename super::template
			  Allocator<ScoreVec>::type> >	array_t;
	    
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
	typedef ScoreVecTuple	first_argument_type;

      public:
	ParamUpdate(Score gn, Score gp)	:_gn(gn), _gp(gp)		{}

	result_type	operator ()(argument_type p) const
			{
			    using namespace	boost;
			    
			    return result_type(get<0>(p) - get<1>(p),
					       _gn*get<0>(p) - _gp*get<1>(p));
			}
	template <class TUPLE>
	void		operator ()(first_argument_type p, TUPLE t) const
			{
			    using namespace	boost;

			    get<1>(t) = first_argument_type(
				get<0>(get<0>(t)) + get<0>(p) - get<1>(p),
				get<1>(get<0>(t)) + _gn*get<0>(p)
						  - _gp*get<1>(p));
			}

      private:
	const ScoreVec	_gn;
	const ScoreVec	_gp;
    };

    class CoeffInit
    {
      public:
	typedef ScoreVecTuple	argument_type;
	typedef ScoreVecTuple	result_type;
	
      public:
	CoeffInit(Score g_avg, Score g_sqavg, Score e)
	    :_g_avg(g_avg), _g_rvar(1/(g_sqavg - g_avg*g_avg + e*e))	{}

	result_type	operator ()(argument_type params) const
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

	result_type	operator ()(argument_type  coeffs) const
			{
			    return (boost::get<0>(coeffs) * _g +
				    boost::get<1>(coeffs));
			}
	
      private:
	const ScoreVec	_g;
    };

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
    typedef typename ScoreVecArray2Array::iterator	row_siterator;
    typedef typename ScoreVecArray2Array::const_iterator
							const_row_siterator;
    typedef ring_iterator<row_siterator>		ScoreVecArray2Ring;
    typedef box_filter_iterator<ScoreVecArray2Ring>	ScoreVecArray2Box;
    typedef box_filter_iterator<const_col_siterator>	ScoreVecArrayBox;
    typedef box_filter_iterator<const_reverse_col_siterator>
							ScoreVecArrayRBox;

    typedef Array<GuideElement>				GuideArray;
    typedef Array2<GuideArray>				GuideArray2;
    typedef typename GuideArray::iterator		col_giterator;
    typedef typename GuideArray::const_iterator		const_col_giterator;
    typedef typename GuideArray2::iterator		row_giterator;
    typedef typename GuideArray2::const_iterator	const_row_giterator;
    typedef box_filter_iterator<const_col_giterator>	GuideBox;

    typedef Array<Disparity,
		  Buf<Disparity,
		      typename super::template
		      Allocator<Disparity>::type> >	DisparityArray;
    typedef Array2<DisparityArray,
		   Buf<Disparity,
		       typename super::template
		       Allocator<Disparity>::type> >	DisparityArray2;
    typedef typename DisparityArray::iterator		col_diterator;
    typedef typename DisparityArray::const_iterator	const_col_diterator;
    typedef typename DisparityArray::reverse_iterator	reverse_col_diterator;
    typedef typename DisparityArray2::iterator		row_diterator;
    typedef typename DisparityArray2::const_iterator	const_row_diterator;

    typedef Array<float,
		  Buf<float,
		      typename super::template
		      Allocator<float>::type> >		FloatArray;
    typedef typename FloatArray::reverse_iterator	reverse_col_fiterator;
    
    struct Buffers
    {
	void	initialize(size_t N, size_t D, size_t W)		;
	void	initialize(size_t N, size_t D, size_t W, size_t H)	;
	
#  if defined(RING)
	ScoreVecArray2Array	P;	// (N + 1) x W x 2D
	GuideArray2		E;	// (N + 1) x W
#  else
	ScoreVecArray2		Q;	// W x 2D
	GuideArray		F;	// 1 x W
#  endif
	ScoreVecArray2Array	A;
	DisparityArray		dminL;	// 1 x (W - N + 1)
	FloatArray		delta;	// 1 x (W - N + 1)
	DisparityArray		dminR;	// 1 x (W + D - 1)
	ScoreVecArray		RminR;	// 1 x D
	DisparityArray2		dminV;	// (W - N + 1) x (H + D - 1)
	ScoreVecArray2		RminV;	// (W - N + 1) x D
    };

    template <class ITER>
    class TupleIterator
    {
      public:
	typedef typename iterator_value<ITER>::type		element_type;
	typedef boost::tuple<element_type, element_type>	value_type;
	typedef value_type					reference;

      public:
	TupleIterator(const ITER& p, const ITER& q)	:_p(p), _q(q)	{}
    
	reference	operator *()	const	{ return reference(*_p, *_q); }
	TupleIterator&	operator ++()		{ ++_p; ++_q; return *this; }

      private:
	ITER	_p, _q;
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

    template <template <class, class> class ASSIGN, class COL, class COL_RV>
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
    buffers->initialize(N, D, W, H);
    
#if defined(RING)
    ScoreVecArray2Ring	rowP(buffers->P.begin(), buffers->P.end());
    ScoreVecArray2Box	boxQ;
    GuideArrayRing	rowE(buffers->E.begin(), buffers->E.end());
    GuideArrayBox	boxF;
#else
    ROW			rowLp = rowL, rowRp = rowR;
#endif
    ScoreVecArray2Ring	rowA(buffers->A.begin(), buffers->A.end());
    ScoreVecArray2Box	boxB;
    ROW			rowG  = rowL;
    ROW const		rowL0 = rowL + N - 1, rowL1 = rowL0 + N - 1;

    for (; rowL != rowLe; ++rowL)
    {
	start(1);
      // 各左画素に対して視差[0, D)の右画素のそれぞれとの間の相違度を計算し，
      // フィルタパラメータ(= 縦横両方向に積算された相違度(コスト)の総和
      // および画素毎のコストとガイド画素の積和)を初期化
#if defined(RING)
	initializeFilterParameters<assign>(
	    rowL->cbegin(), rowL->cend(),
	    make_rvcolumn_iterator(rowR->cbegin()),
	    rowP->begin(), rowE->begin());
	++rowP;
	++rowE;
#else
	if (rowL <= rowL0)
	    initializeFilterParameters<plus_assign>(
		rowL->cbegin(), rowL->cend(),
		make_rvcolumn_iterator(rowR->cbegin()),
		buffers->Q.begin(), buffers->F.begin());
	else
	{
	    updateFilterParameters(rowL->cbegin(), rowL->cend(),
				   make_rvcolumn_iterator(rowR->cbegin()),
				   rowLp->cbegin(),
				   make_rvcolumn_iterator(rowRp->cbegin()),
				   buffers->Q.begin(), buffers->F.begin());
	    ++rowLp;
	    ++rowRp;
	}
#endif
	if (rowL >= rowL0)	// 最初のN行に対してコストPが計算済みならば...
	{
#if defined(RING)
	  // コストを縦方向に積算
	    if (rowL == rowL0)
	    {
		boxQ.initialize(rowP - N, N);
		boxF.initialize(rowE - N, N);
	    }
	    const ScoreVecArray2&	Q = *boxQ;
	    const GuideArray&		F = *boxF;
	    ++boxQ;
	    ++boxF;
#else
	    const ScoreVecArray2&	Q = buffers->Q;
	    const GuideArray&		F = buffers->F;
#endif
	    start(2);
	  // さらにコストを横方向に積算してフィルタパラメータを計算し，
	  // それを用いてフィルタ係数を初期化
	    initializeFilterCoefficients(Q.cbegin(), Q.cend(),
					 F.cbegin(), rowA->begin());
	    ++rowA;

	    if (rowL >= rowL1)		// rowL0からN行分のフィルタ係数が
	    {				// 計算済みならば...
		start(3);
		if (rowL == rowL1)
		    boxB.initialize(rowA - N, N);
		const ScoreVecArray2&	B = *boxB;
		
		start(4);
	      // さらにフィルタ係数を横方向に積算して最終的な係数を求め，
	      // それにguide画像を適用してウィンドウコストを求め，それを
	      // 用いてそれぞれ左/右/上画像を基準とした最適視差を計算
	  	buffers->RminR = std::numeric_limits<Score>::max();
		computeDisparities(B.crbegin(), B.crend(), rowG->crbegin(),
				   buffers->dminL.rbegin(),
				   buffers->delta.rbegin(),
				   make_rvcolumn_iterator(
				       buffers->dminR.end() - D + 1),
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

    size_t		v = H, cV = std::distance(rowL, rowLlast);
#if defined(RING)
    ScoreVecArray2Ring	rowP(buffers->P.begin(), buffers->P.end());
    ScoreVecArray2Box	boxQ;
    GuideArrayRing	rowE(buffers->E.begin(), buffers->E.end());
    GuideArrayBox	boxF;
#else
    ROW			rowLp = rowL, rowRp = rowR;
    size_t		cVp = cV;
#endif
    ScoreVecArray2Ring	rowA(buffers->A.begin(), buffers->A.end());
    ScoreVecArray2Box	boxB;
    const ROW_D		rowD0 = rowD + N - 1;
    ROW			rowG  = rowL;
    const ROW 		rowL0 = rowL + N - 1, rowL1 = rowL0 + N - 1;

    for (; rowL != rowLe; ++rowL)
    {
	--v;
	--cV;
	
	start(1);
      // 各左画素に対して視差[0, D)の右画素のそれぞれとの間の相違度を計算し，
      // フィルタパラメータ(= 縦横両方向に積算された相違度(コスト)の総和
      // および画素毎のコストとガイド画素の積和)を初期化
#if defined(RING)
	initializeFilterParameters<assign>(
	    rowL->cbegin(), rowL->cend(),
	    make_rvcolumn_iterator(
		make_fast_zip_iterator(
		    boost::make_tuple(
			rowR->cbegin(), make_vertical_iterator(rowV, cV)))),
	    rowP->begin(), rowE->begin());
	++rowP;
	++rowE;
#else
	if (rowL <= rowL0)
	    initializeFilterParameters<plus_assign>(
		rowL->cbegin(), rowL->cend(),
		make_rvcolumn_iterator(
		    make_fast_zip_iterator(
			boost::make_tuple(
			    rowR->cbegin(), make_vertical_iterator(rowV, cV)))),
		buffers->Q.begin(), buffers->F.begin());
	else
	{
	    updateFilterParameters(rowL->cbegin(), rowL->cend(),
				   make_rvcolumn_iterator(
				       make_fast_zip_iterator(
					   boost::make_tuple(
					       rowR->cbegin(),
					       make_vertical_iterator(rowV,
								      cV)))),
				   rowLp->cbegin(),
				   make_rvcolumn_iterator(
				       make_fast_zip_iterator(
					   boost::make_tuple(
					       rowRp->cbegin(),
					       make_vertical_iterator(rowV,
								      --cVp)))),
				   buffers->Q.begin(), buffers->F.begin());
	    ++rowLp;
	    ++rowRp;
	}
#endif
	if (rowL >= rowL0)	// 最初のN行に対してコストPが計算済みならば...
	{
#if defined(RING)
	  // コストを縦方向に積算
	    if (rowL == rowL0)
	    {
		boxQ.initialize(rowP - N, N);
		boxF.initialize(rowE - N, N);
	    }
	    const ScoreVecArray2&	Q = *boxQ;
	    const GuideArray&		F = *boxF;
	    ++boxQ;
	    ++boxF;
#else
	    const ScoreVecArray2&	Q = buffers->Q;
	    const GuideArray&		F = buffers->F;
#endif
	    start(2);
	  // さらにコストを横方向に積算してフィルタパラメータを計算し，
	  // それを用いてフィルタ係数を初期化
	    initializeFilterCoefficients(Q.cbegin(), Q.cend(),
					 F.cbegin(), rowA->begin());
	    ++rowA;

	    if (rowL >= rowL1)		// rowL0からN行分のフィルタ係数が
	    {				// 計算済みならば...
		start(3);
	      // フィルタ係数を縦方向に積算
		if (rowL == rowL1)
		    boxB.initialize(rowA - N, N);
		const ScoreVecArray2&	B = *boxB;
		
		start(4);
	      // さらにフィルタ係数を横方向に積算して最終的な係数を求め，
	      // それにguide画像を適用してウィンドウコストを求め，それを
	      // 用いてそれぞれ左/右/上画像を基準とした最適視差を計算
		buffers->RminR = std::numeric_limits<Score>::max();
		computeDisparities(B.rbegin(), B.crend(), rowG->crbegin(),
				   buffers->dminL.rbegin(),
				   buffers->delta.rbegin(),
				   make_rvcolumn_iterator(
				       make_fast_zip_iterator(
					   boost::make_tuple(
					       buffers->dminR.end() - D + 1,
					       make_vertical_iterator(
						   buffers->dminV.end(), v)))),
				   make_row_iterator<boost::use_default>(
				       make_fast_zip_iterator(
					   boost::make_tuple(
					       make_dummy_iterator(
						   &(buffers->RminR)),
					       buffers->RminV.rbegin()))));
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
#if !defined(NO_VERTICAL_BM)
    start(6);
    rowD = rowD0;
    for (v = H - 2*(N - 1); v-- != 0; )
    {
	pruneDisparities(make_vertical_iterator(buffers->dminV.cbegin(), v),
			 make_vertical_iterator(buffers->dminV.cend(),   v),
			 rowD->begin() + N - 1);
	++rowD;
    }
#endif
    _bufferPool.put(buffers);
    nextFrame();
}

template <class SCORE, class DISP>
template <template <class, class> class ASSIGN, class COL, class COL_RV> void
GFStereo<SCORE, DISP>::initializeFilterParameters(COL colL, COL colLe,
						  COL_RV colRV,
						  col_siterator colQ,
						  col_giterator colF) const
{
    typedef typename iterator_value<COL>::type			pixel_type;
    typedef typename iterator_value<COL_RV>::type::iterator	in_iterator;
#if defined(SSE)
    typedef Diff<mm::vec<pixel_type> >				op_type;
    typedef boost::transform_iterator<
	Binder<op_type>, mm::load_iterator<in_iterator> >	piterator;
    typedef mm::cvtup_iterator<
	assignment_iterator<ParamInit,
			    typename ScoreVecArray::iterator2> >
								qiterator;
#else
    typedef Diff<pixel_type>					op_type;
    typedef boost::transform_iterator<Binder<op_type>, in_iterator>
								piterator;
    typedef assignment_iterator<ParamInit,
				typename ScoreVecArray::iterator2>
								qiterator;
#endif
    typedef ASSIGN<
	typename std::iterator_traits<piterator>::value_type,
	typename std::iterator_traits<qiterator>::reference>	assign_type;
    typedef ASSIGN<Score, Score&>				gassign_type;

    for (; colL != colLe; ++colL)
    {
	const Score	pixL = *colL;
	piterator	P(in_iterator(colRV->begin()),
			  makeBinder(op_type(_params.intensityDiffMax), pixL));
	for (qiterator Q( make_assignment_iterator(colQ->begin2(),
						   ParamInit(pixL))),
		       Qe(make_assignment_iterator(colQ->end2(),
						   ParamInit(pixL)));
	     Q != Qe; ++Q, ++P)
	    assign_type()(*P, *Q);

	gassign_type()(pixL,	    colF->g_sum);
	gassign_type()(pixL * pixL, colF->g_sqsum);
	
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
    typedef typename iterator_value<COL>::type			pixel_type;
    typedef typename iterator_value<COL_RV>::type::iterator	in_iterator;
#if defined(SSE)
    typedef Diff<mm::vec<pixel_type> >				op_type;
    typedef boost::transform_iterator<
	Binder<op_type>, mm::load_iterator<in_iterator> >	piterator;
    typedef mm::cvtup_iterator<
	assignment_iterator<ParamUpdate,
			    typename ScoreVecArray::iterator2> >
								qiterator;
#else
    typedef Diff<pixel_type>					op_type;
    typedef boost::transform_iterator<Binder<op_type>, in_iterator>
								piterator;
    typedef assignment_iterator<ParamUpdate,
				typename ScoreVecArray::iterator2>
								qiterator;
#endif
  /* 本来は fast_zip_iterator<boost::tuple<piterator, piterator> > として
   * 定義したいが，fast_zip_iterator を piterator と組み合わせると速度低下が
   * 著しいので，TupleIterator<ITER> を用いる．
   */
    typedef TupleIterator<piterator>				ppiterator;

    for (; colL != colLe; ++colL)
    {
	const Score	pixLp = *colLp, pixL = *colL;
	ppiterator	P(boost::make_transform_iterator(
			      colRV->begin(),
			      makeBinder(op_type(_params.intensityDiffMax),
					 pixL)),
			  boost::make_transform_iterator(
			      colRVp->begin(),
			      makeBinder(op_type(_params.intensityDiffMax),
					 pixLp)));
	for (qiterator Q( make_assignment_iterator(colQ->begin2(),
						   ParamUpdate(pixL, pixLp))),
		       Qe(make_assignment_iterator(colQ->end2(),
						   ParamUpdate(pixL, pixLp)));
	     Q != Qe; ++Q, ++P)
	    *Q += *P;

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
    GuideBox		boxG(colF, _params.windowSize);
    for (ScoreVecArrayBox boxR(colQ, _params.windowSize), boxRe(colQe);
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
    for (ScoreVecArrayRBox boxC(colB, _params.windowSize), boxCe(colBe);
	 boxC != boxCe; ++boxC)
    {
	std::transform(boxC->cbegin2(), boxC->cend2(),
		       R.begin(), CoeffTrans(*colG));
	++colG;

#if defined(SSE)
	typedef mm::store_iterator<
	    typename iterator_value<DMIN_RV>::type::iterator>	diterator;
#  if defined(WITHOUT_CVTDOWN)
	typedef mm::cvtdown_mask_iterator<
	    Disparity,
	    mm::mask_iterator<
		typename ScoreVecArray::const_iterator,
		typename iterator_value<RMIN_RV>::type::iterator> >
								miterator;
#  else
	typedef mm::mask_iterator<
	    Disparity,
	    typename ScoreVecArray::const_iterator,
	    typename iterator_value<RMIN_RV>::type::iterator>	miterator;
#  endif
#else
	typedef typename iterator_value<DMIN_RV>::type::iterator
								diterator;
	typedef mask_iterator<
	    typename ScoreVecArray::const_iterator,
	    typename iterator_value<RMIN_RV>::type::iterator>	miterator;
#endif
	typedef typename iterator_value<diterator>::type	dvalue_type;

	Idx<DisparityVec>	index;
	diterator		dminRVt((--dminRV)->begin());
#if defined(SSE) && defined(WITHOUT_CVTDOWN)
	miterator	maskRV(make_mask_iterator(R.cbegin(), RminRV->begin()));
	for (miterator maskRVe(make_mask_iterator(R.cend(),   RminRV->end()));
	     maskRV != maskRVe; ++maskRV)
#else
	miterator	maskRV(R.cbegin(), RminRV->begin());
	for (miterator maskRVe(R.cend(),   RminRV->end());
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
#if defined(RING)
    P.resize(N + 1);
    for (row_siterator rowP = P.begin(); rowP != P.end(); ++rowP)
	if (!rowP->resize(W, 2*DD))
	    break;
#else
    Q.resize(W, 2*DD);			// Q(u, *; d)
    Q = 0;
    F.resize(W);
    F = GuideElement();
#endif

    A.resize(N + 1);
    for (row_siterator rowA = A.begin(); rowA != A.end(); ++rowA)
	if (!rowA->resize(W - N + 1, 2*DD))
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
    RminV = std::numeric_limits<SCORE>::max();
}

}
#endif	// !__TU_GFSTEREO_H
