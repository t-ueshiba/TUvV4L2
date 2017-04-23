/*
 *  $Id$
 */
#ifndef __TU_WEIGHTEDMEDIANFILTER_H
#define __TU_WEIGHTEDMEDIANFILTER_H

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/set.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include "TU/Quantizer.h"
#include "TU/Vector++.h"
#include "TU/algorithm.h"	// diff(const T&, const T&)
#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif
#include "TU/Profiler.h"

namespace TU
{
/************************************************************************
*  class ExpDiff<S, T>							*
************************************************************************/
template <class S, class T>
class ExpDiff
{
  public:
    using argument_type	= S;
    using result_type	= T;

  public:
		ExpDiff(T sigma=1) :_nsigma(-sigma)	{}

    void	setSigma(result_type sigma)		{ _nsigma = -sigma; }
    T		operator ()(S x, S y) const
		{
		    return f(x, y, typename std::is_arithmetic<S>::type());
		}
    
  private:
    T		f(S x, S y, std::true_type) const
		{
		    return std::exp(diff(x, y) / _nsigma);
		}
    T		f(S x, S y, std::false_type) const
		{
		    Vector<T, 3>	v;
		    v[0] = T(x.r) - T(y.r);
		    v[1] = T(x.g) - T(y.g);
		    v[2] = T(x.b) - T(y.b);
		    return std::exp(length(v) / _nsigma);
		}

  private:
    T		_nsigma;
};
    
namespace detail
{
/************************************************************************
*  class WeightedMedianFilterBase<W>					*
************************************************************************/
template <class W>
class WeightedMedianFilterBase
{
  public:
    using weight_type	= typename W::result_type;
    using warray_type	= Array<weight_type>;
    
  protected:
    class Bin : public boost::intrusive::list_base_hook<>
    {
      public:
		Bin()	:_n(0)			{}
	    
		operator size_t()	const	{ return _n; }
	size_t	operator ++()			{ return ++_n; }
	size_t	operator --()			{ return --_n; }
	
      private:
	size_t	_n;
    };

    class Histogram : public Array<Bin>,
		      public boost::intrusive::set_base_hook<>
    {
      public:
			Histogram() :_n(0), _weighted_sum(0)	{}

			operator size_t()		const	{ return _n; }
	void		clear()
			{
			    Array<Bin>::operator =(Bin());
			    _nonempty_bins.clear();
			    _n = 0;
			    _weighted_sum = 0;
			}
	size_t		add(size_t idxG)
			{
			    auto&	bin = (*this)[idxG];
			    if (++bin == 1)
				_nonempty_bins.push_back(bin);
			    return ++_n;
			}
	size_t		remove(size_t idxG)
			{
			    auto&	bin = (*this)[idxG];
			    if (bin == 0)
				return _n;
			    else if (--bin == 0)
				_nonempty_bins.erase(
				    _nonempty_bins.iterator_to(bin));
			    return --_n;
			}
	weight_type	weighted_sum(const warray_type& weights) const
			{
			    _weighted_sum = 0;
			    for (const auto& bin : _nonempty_bins)
				_weighted_sum += bin * weights[idx(bin)];
			    return _weighted_sum;
			}
	weight_type	weighted_sum() const
			{
			    return _weighted_sum;
			}

	friend bool	operator <(const Histogram& x, const Histogram& y)
			{
			    return &x < &y;
			}
	friend std::ostream&
			operator <<(std::ostream& out,
				    const Histogram& histogram)
			{
			    out << histogram._n << '\t';
			    return histogram.put(out) << std::endl;
			}
	
      private:
	size_t		idx(const Bin& bin) const
			{
			    return &bin - &(*this)[0];
			}
    
      private:
	boost::intrusive::list<Bin>	_nonempty_bins;
	size_t				_n;		// total #points
	mutable weight_type		_weighted_sum;	// cache
    };

    class HistogramArray : public Array<Histogram>
    {
      public:
		HistogramArray()	:_median(0)	{}
		~HistogramArray()			{ clear(); }
	
	void	resize(size_t nbinsI, size_t nbinsG)
		{
		    Array<Histogram>::resize(nbinsI);
		    for (auto& h : *this)
			h.resize(nbinsG);
		    clear();
		}
	void	add(size_t idxI, size_t idxG)
		{
		    auto&	h = (*this)[idxI];
		    if (h.add(idxG) == 1)
			_nonempty_histograms.insert(h);
		}
	void	remove(size_t idxI, size_t idxG)
		{
		    auto&	h = (*this)[idxI];
		    if (h.remove(idxG) == 0)
			_nonempty_histograms.erase(
			    _nonempty_histograms.iterator_to(h));
		}
	template <class IDX_I, class IDX_G>
	IDX_G	add(IDX_I idxI, IDX_I idxIe, IDX_G idxG)
		{
		    for (; idxI != idxIe; ++idxI, ++idxG)
			add(*idxI, *idxG);
		    return idxG;
		}
	template <class IDX_I, class IDX_G>
	IDX_G	remove(IDX_I idxI, IDX_I idxIe, IDX_G idxG)
		{
		    for (; idxI != idxIe; ++idxI, ++idxG)
			remove(*idxI, *idxG);
		    return idxG;
		}
	size_t	median(const warray_type& weights) const
		{
		  // medianの現在値に対してその前後の重み和の差(balance)を計算
		    weight_type	balance = 0;
		    auto	p = _nonempty_histograms.begin();
		    for (; idx(*p) < _median; ++p)
			balance += p->weighted_sum(weights);
		    for (auto q = p; q != _nonempty_histograms.end(); ++q)
			balance -= q->weighted_sum(weights);

		  // balance >= 0 となる最左位置を探索
		    if (balance >= 0)	// balance < 0 となるまで左にシフト
			while ((balance -= 2 * (--p)->weighted_sum()) >= 0)
			    ;
		    else		// balance >= 0 となるまで右にシフト
			while ((balance += 2 * p->weighted_sum()) < 0)
			    ++p;

		    return (_median = idx(*p));
		}
	
      private:
	void	clear()
		{
		    for (auto& h : *this)
			h.clear();
		    _nonempty_histograms.clear();
		    _median = 0;
		}
	size_t	idx(const Histogram& h) const
		{
		    return &h - &(*this)[0];
		}

      private:
	boost::intrusive::set<Histogram>	_nonempty_histograms;
	mutable size_t				_median;	// cache
    };
    
  public:
    WeightedMedianFilterBase(const W& wfunc, size_t winSize,
			     size_t nbinsI, size_t nbinsG)		;

    size_t	winSize()		const	{ return _winSize; }
    size_t	nbinsI()		const	{ return _nbinsI; }
    size_t	nbinsG()		const	{ return _nbinsG; }
    void	setWinSize(size_t w)		{ _winSize = w; }
    void	setNBinsI(size_t nbins)		{ _nbinsI = nbins; }
    void	setNBinsG(size_t nbins)		{ _nbinsG = nbins; }
    void	refreshWeights()		{ _initialized = false; }
    
  protected:
    template <class T>
    void	setWeights(const QuantizerBase<T>& quantizer)		;
    auto	weights(size_t i)	const	{ return _weights[i]; }
    
  private:
    const W&		_wfunc;
    size_t		_winSize;
    size_t		_nbinsI;
    size_t		_nbinsG;
    bool		_initialized;
    Array2<weight_type>	_weights;	// weight function
};

template <class W> inline
WeightedMedianFilterBase<W>::WeightedMedianFilterBase(const W& wfunc,
						      size_t winSize,
						      size_t nbinsI,
						      size_t nbinsG)
    :_wfunc(wfunc), _winSize(winSize),
     _nbinsI(nbinsI), _nbinsG(nbinsG), _initialized(false)
{
}

template <class W> template <class T> void
WeightedMedianFilterBase<W>::setWeights(const QuantizerBase<T>& quantizer)
{
    if (_initialized)
	return;

    _weights.resize(quantizer.size(), quantizer.size());
    for (size_t i = 0; i < _weights.nrow(); ++i)
	for (size_t j = i; j < _weights.ncol(); ++j)
	    _weights[i][j] = _weights[j][i]
			   = _wfunc(quantizer[i], quantizer[j]);

    _initialized = std::is_unsigned<T>::value;
}

}
/************************************************************************
*  class WeightedMedianFilter<T, W>					*
************************************************************************/
template <class T, class W>
class WeightedMedianFilter : public detail::WeightedMedianFilterBase<W>
{
  private:
    using value_type		= T;
    using guide_type		= typename W::argument_type;
    using super			= detail::WeightedMedianFilterBase<W>;
    using HistogramArray	= typename super::HistogramArray;

  public:
    WeightedMedianFilter(const W& wfunc=W(), size_t winSize=3,
			 size_t nbinsI=256, size_t nbinsG=256)
	:super(wfunc, winSize, nbinsI, nbinsG)				{}

    using	super::winSize;
    using	super::nbinsI;
    using	super::nbinsG;
    
    template <class IN, class GUIDE, class OUT>
    void	convolve(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out)	;

  private:
    Quantizer<value_type>	_quantizerI;
    Quantizer<guide_type>	_quantizerG;
    HistogramArray		_histograms;	// 2D histogram
};
    
template <class T, class W>
template <class IN, class GUIDE, class OUT> void
WeightedMedianFilter<T, W>::convolve(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out)
{
    if (std::distance(ib, ie) < winSize())
	return;

    const auto&	indicesI = _quantizerI(ib, ie, nbinsI());  // 入力を量子化
    const auto&	indicesG = _quantizerG(gb, ge, nbinsG());  // ガイドを量子化

    super::setWeights(_quantizerG);	// 重みの2次元lookup tableをセット

    auto	headI = indicesI.begin();
    auto	headG = indicesG.begin();
    auto	tailI = headI;
    auto	idxG  = headG;
    std::advance(tailI, winSize()-1);
    std::advance(idxG,  winSize()/2);

  // ウィンドウ初期位置におけるヒストグラムをセット
    _histograms.resize(_quantizerI.size(), _quantizerG.size());
    auto	tailG = _histograms.add(headI, tailI, headG);

  // median点を探索し，その値を出力
    for (; tailI != indicesI.end(); ++tailI)
    {
	_histograms.add(*tailI, *tailG);	// tail点をヒストグラムに追加
	*out = _quantizerI[_histograms.median(super::weights(*idxG))];
	_histograms.remove(*headI, *headG);	// head点をヒストグラムから除去

	++tailG;
	++headI;
	++headG;
	++idxG;
	++out;
    }
}

/************************************************************************
*  class WeightedMedianFilter2<T, W, PF>				*
************************************************************************/
template <class T, class W, class CLOCK=void>
class WeightedMedianFilter2 : public detail::WeightedMedianFilterBase<W>,
			      public Profiler<CLOCK>
{
  private:
    using value_type		= T;
    using guide_type		= typename W::argument_type;
    using super			= detail::WeightedMedianFilterBase<W>;
    using pf_type		= Profiler<CLOCK>;
    using HistogramArray	= typename super::HistogramArray;
#if defined(USE_TBB)
    template <class ROW_I, class ROW_G, class ROW_O>
    class Filter
    {
      public:
	Filter(const WeightedMedianFilter2<T, W, CLOCK>& wmf,
	       ROW_I rowI, ROW_G rowG, ROW_O rowO)
	    :_wmf(wmf), _rowI(rowI), _rowG(rowG), _rowO(rowO)		{}
	    
	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    _wmf.filter(_rowI + r.begin(), _rowI + r.end(),
				_rowG + r.begin(), _rowO + r.begin());
		}

      private:
	const WeightedMedianFilter2<T, W, CLOCK>&	_wmf;
	ROW_I						_rowI;
	ROW_G						_rowG;
	ROW_O						_rowO;
    };

    template <class ROW_I, class ROW_G, class ROW_O>
    Filter<ROW_I, ROW_G, ROW_O>
		makeFilter(ROW_I rowI, ROW_G rowG, ROW_O rowO) const
		{
		    return Filter<ROW_I, ROW_G, ROW_O>(*this, rowI, rowG, rowO);
		}
#endif
  // std::reverse_iterator<ITER> はITERが指すオブジェクトへの参照を返すため
  // counting_iterator に対しては使えないので，独自に定義
    template <class ITER>
    class reverse_iterator
	: public boost::iterator_adaptor<reverse_iterator<ITER>,
					 ITER,
					 boost::use_default,
					 boost::use_default,
					 iterator_value<ITER> >
    {
      private:
	using super	= boost::iterator_adaptor<reverse_iterator,
						  ITER,
						  boost::use_default,
						  boost::use_default,
						  iterator_value<ITER> >;

      public:
	using		typename super::reference;
	using		typename super::difference_type;
	friend class	boost::iterator_core_access;

      public:
	reverse_iterator(const ITER& iter)	:super(iter)	{}

      private:
	reference	dereference() const
			{
			    auto	tmp = super::base();
			    return *--tmp;
			}
	void		advance(difference_type n)
			{
			    std::advance(super::base_reference(), -n);
			}
	void		increment()
			{
			    --super::base_reference();
			}
	void		decrement()
			{
			    ++super::base_reference();
			}
	difference_type	distance_to(const reverse_iterator& iter) const
			{
			    return std::distance(iter.base(), super::base());
			}
    };

  public:
    WeightedMedianFilter2(const W& wfunc=W(), size_t winSize=3,
			  size_t nbinsI=256, size_t nbinsG=256)
	:super(wfunc, winSize, nbinsI, nbinsG), pf_type(4),
	 _grainSize(100)						{}

    using	super::winSize;
    using	super::nbinsI;
    using	super::nbinsG;

    template <class IN, class GUIDE, class OUT>
    void	convolve(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out)	;
    size_t	grainSize()			const	{ return _grainSize; }
    void	setGrainSize(size_t gs)			{ _grainSize = gs; }
    
  private:
    template <class ROW_I, class ROW_G, class ROW_O>
    void	filter(ROW_I rowI, ROW_I rowIe,
		       ROW_G rowG, ROW_O out)			const	;
    template <class ROW_I, class ROW_G, class COL_C, class COL_G, class COL_O>
    void	filterRow(HistogramArray& histograms,
			  ROW_I rowI, ROW_G rowG, COL_C c,
			  COL_G colG, COL_O colO)		const	;
    
  private:
    size_t			_grainSize;
    Quantizer2<value_type>	_quantizerI;
    Quantizer2<guide_type>	_quantizerG;
};

template <class T, class W, class CLOCK>
template <class IN, class GUIDE, class OUT> void
WeightedMedianFilter2<T, W, CLOCK>::convolve(IN ib, IN ie,
					  GUIDE gb, GUIDE ge, OUT out)
{
    if (std::distance(ib, ie) < winSize() || ib->size() < winSize())
	return;

    pf_type::start(0);
    const auto&	indicesI = _quantizerI(ib, ie, nbinsI());  // 入力を量子化
    const auto&	indicesG = _quantizerG(gb, ge, nbinsG());  // ガイドを量子化

    pf_type::start(1);
    super::setWeights(_quantizerG);	// 重みの2次元lookup tableをセット

#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(
			  0, indicesI.size() + 1 - winSize(), _grainSize),
		      makeFilter(indicesI.begin(), indicesG.begin(), out));
#else
    filter(indicesI.begin(), indicesI.end() + 1 - winSize(),
	   indicesG.begin(), out);
#endif
    pf_type::nextFrame();
}

template <class T, class W, class CLOCK>
template <class ROW_I, class ROW_G, class ROW_O> void
WeightedMedianFilter2<T, W, CLOCK>::filter(ROW_I rowI, ROW_I rowIe,
					ROW_G rowG, ROW_O rowO) const
{
    using col_iterator	= boost::counting_iterator<size_t>;
    using rcol_iterator	= reverse_iterator<col_iterator>;

    pf_type::start(2);
    auto		endI = rowI;
    auto		midG = rowG;
    const size_t	mid = winSize()/2, rmid = (winSize()-1)/2;
    std::advance(endI, winSize() - 1);	// ウィンドウの最下行
    std::advance(midG, mid);		// ウィンドウの中央行
    std::advance(rowO, mid);		// 出力行をウィンドウの中央に合わせる

  // ウィンドウ初期位置におけるヒストグラムをセット
    HistogramArray	histograms;
    histograms.resize(_quantizerI.size(), _quantizerG.size());
    for (size_t c = 0; c < winSize() - 1; ++c)
	histograms.add(make_vertical_iterator(rowI, c),
		       make_vertical_iterator(endI, c),
		       make_vertical_iterator(rowG, c));
    
    pf_type::start(3);
  // 左から右／右から左に交互に走査してmedian点を探索
    for (bool reverse = false; rowI != rowIe; ++rowI)
    {
	if (!reverse)
	{
	    filterRow(histograms, rowI, rowG, col_iterator(0),
		      midG->begin() + mid, rowO->begin() + mid);
	    reverse = true;
	}
	else
	{
	    filterRow(histograms, rowI, rowG,
		      rcol_iterator(col_iterator(rowI->size())),
		      midG->rbegin() + rmid, rowO->rbegin() + rmid);
	    reverse = false;
	}

	++rowG;
	++midG;
	++rowO;
    }
}

template <class T, class W, class CLOCK>
template <class ROW_I, class ROW_G, class COL_C, class COL_G, class COL_O> void
WeightedMedianFilter2<T, W, CLOCK>::filterRow(HistogramArray& histograms,
					      ROW_I rowI, ROW_G rowG,
					      COL_C head,
					      COL_G colG, COL_O colO) const
{
    auto	endI = rowI;
    std::advance(endI, winSize() - 1);		// ウィンドウ最下行
    auto	endG = rowG;
    std::advance(endG, winSize() - 1);		// ウィンドウ最下行
    auto	tail = head;			// ウィンドウ最左／最右列
    auto	end  = head + winSize() - 1;	// ウィンドウ最右／最左列

  // ウィンドウ最下行の点をヒストグラムに追加
    for (; tail != end; ++tail)
	histograms.add(*(endI->begin() + *tail), *(endG->begin() + *tail));
    
    ++endI;					// 最下行の次
    end = head + rowI->size();			// 列の右端／左端
    
    for (; tail != end; ++head, ++tail)
    {
      // tail点をヒストグラムに追加
	histograms.add(make_vertical_iterator(rowI, *tail),
		       make_vertical_iterator(endI, *tail),
		       make_vertical_iterator(rowG, *tail));

      // median点を検出してその値を出力
	*colO = _quantizerI[histograms.median(super::weights(*colG))];

      // head点をヒストグラムから除去
	histograms.remove(make_vertical_iterator(rowI, *head),
			  make_vertical_iterator(endI, *head),
			  make_vertical_iterator(rowG, *head));

	++colG;
	++colO;
    }

  // ウィンドウ最上行の点をヒストグラムから除去
    for (; head != end; ++head)
	histograms.remove(*(rowI->begin() + *head), *(rowG->begin() + *head));
}
    
}
#endif
