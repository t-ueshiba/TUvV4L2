/*
 *  $Id$
 */
#ifndef __TU_WEIGHTEDMEDIANFILTER_H
#define __TU_WEIGHTEDMEDIANFILTER_H

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/set.hpp>
#include "TU/Quantizer.h"
#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif

#if defined(PROFILE)
#  include "TU/Profiler.h"
#else
struct Profiler
{
    Profiler(size_t)				{}

    void	reset()			const	{}
    void	print(std::ostream&)	const	{}
    void	start(int)		const	{}
    void	nextFrame()		const	{}
};
#endif

namespace TU
{
namespace detail
{
/************************************************************************
*  class WeightedMedianFilterBase<W>					*
************************************************************************/
template <class W>
class WeightedMedianFilterBase
{
  public:
    typedef typename W::result_type	weight_type;
    typedef Array<weight_type>		warray_type;
    
  protected:
    class Bin : public boost::intrusive::list_base_hook<>
    {
      public:
	Bin()	:_n(0)				{}
	    
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
	Histogram()	:Array<Bin>(), _n(0), _weighted_sum(0)	{}

			operator size_t()		const	{ return _n; }
	void		clear()
			{
			    Array<Bin>::fill(Bin());
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

    class HistogramArray : public Array2<Histogram>
    {
      public:
		~HistogramArray()
		{
		    clear();
		}
	void	resize(size_t nbinsI, size_t nbinsG)
		{
		    Array2<Histogram>::resize(nbinsI, nbinsG);
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
	size_t	updateMedian(size_t median, const warray_type& weights) const
		{
		  // medianの現在値に対してその前後の重み和の差(balance)を計算
		    weight_type	balance = 0;
		    auto	p = _nonempty_histograms.begin();
		    for (; idx(*p) < median; ++p)
			balance += p->weighted_sum(weights);
		    for (auto q = p; q != _nonempty_histograms.end(); ++q)
			balance -= q->weighted_sum(weights);

		  // medianを1つずつ増減しながら balance >= 0 となる
		  // 最小のmedian値を探索
		    if (balance >= 0)	// balance < 0 となるまでmedianを左にシフト
			while ((balance -= 2 * (--p)->weighted_sum()) >= 0)
			    ;
		    else	// balance >= 0 となるまでmedianを右にシフト
			while ((balance += 2 * p->weighted_sum()) < 0)
			    ++p;

		    return idx(*p);
		}
	
      private:
	void	clear()
		{
		    for (auto& h : *this)
			h.clear();
		    _nonempty_histograms.clear();
		}
	size_t	idx(const Histogram& h) const
		{
		    return &h - &(*this)[0];
		}

      private:
	boost::intrusive::set<Histogram>	_nonempty_histograms;
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
    void		setWeights(const QuantizerBase<T>& quantizer)	;
    const warray_type&	weights(size_t i)	const	{ return _weights[i]; }
    
  private:
    const W&		_wfunc;
    size_t		_winSize;
    size_t		_nbinsI;
    size_t		_nbinsG;
    bool		_initialized;
    Array2<warray_type>	_weights;	// weight function
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
    typedef T					value_type;
    typedef typename W::argument_type		guide_type;
    typedef detail::WeightedMedianFilterBase<W>	super;
    typedef typename super::HistogramArray	HistogramArray;

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

    auto	currI = indicesI.begin();
    auto	headI = currI;
    auto	currG = indicesG.begin();
    auto	headG = currG;
    auto	centG = currG;
    std::advance(centG, winSize()/2);

  // ヒストグラムを空にする
    _histograms.resize(_quantizerI.size(), _quantizerG.size());

  // ウィンドウ初期位置におけるヒストグラムをセット
    for (size_t n = winSize(); --n > 0; )
    {
	_histograms.add(*currI, *currG);
	++currI;
	++currG;
    }

  // median点を探索し，その値を出力
    size_t	median = 0;
    for (; currI != indicesI.end(); ++currI)
    {
	_histograms.add(*currI, *currG);	// current点をヒストグラムに追加

	median = _histograms.updateMedian(median, super::weights(*centG));
	*out = _quantizerI[median];		// median点の値を出力

	_histograms.remove(*headI, *headG);	// head点をヒストグラムから除去

	++currG;
	++headI;
	++headG;
	++centG;
	++out;
    }
}

/************************************************************************
*  class WeightedMedianFilter2<T, W>					*
************************************************************************/
template <class T, class W>
class WeightedMedianFilter2 : public detail::WeightedMedianFilterBase<W>,
			      public Profiler
{
  private:
    typedef T					value_type;
    typedef typename W::argument_type		guide_type;
    typedef detail::WeightedMedianFilterBase<W>	super;
    typedef typename super::HistogramArray	HistogramArray;
#if defined(USE_TBB)
    template <class ROW_I, class ROW_G, class OUT>
    class Filter
    {
      public:
	Filter(const WeightedMedianFilter2<T, W>& wmf,
	       ROW_I rowI, ROW_G rowG, OUT out)
	    :_wmf(wmf), _rowI(rowI), _rowG(rowG), _out(out)		{}
	    
	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    _wmf.filter(_rowI + r.begin(), _rowI + r.end(),
				_rowG + r.begin(), _out + r.begin());
		}

      private:
	const WeightedMedianFilter2<T, W>&	_wmf;
	ROW_I					_rowI;
	ROW_G					_rowG;
	OUT					_out;
    };

    template <class ROW_I, class ROW_G, class OUT>
    Filter<ROW_I, ROW_G, OUT>
		makeFilter(ROW_I rowI, ROW_G rowG, OUT out) const
		{
		    return Filter<ROW_I, ROW_G, OUT>(*this, rowI, rowG, out);
		}
#endif

  public:
    WeightedMedianFilter2(const W& wfunc=W(), size_t winSize=3,
			  size_t nbinsI=256, size_t nbinsG=256)
	:super(wfunc, winSize, nbinsI, nbinsG),
	 _grainSize(100), Profiler(3)					{}

    using	super::winSize;
    using	super::nbinsI;
    using	super::nbinsG;

    template <class IN, class GUIDE, class OUT>
    void	convolve(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out)	;
    size_t	grainSize()			const	{ return _grainSize; }
    void	setGrainSize(size_t gs)			{ _grainSize = gs; }
    
  private:
    template <class ROW_I, class ROW_G, class OUT>
    void	filter(ROW_I rowI, ROW_I rowIe,
		       ROW_G rowG, OUT out)	const	;
    
  private:
    size_t			_grainSize;
    Quantizer2<value_type>	_quantizerI;
    Quantizer2<guide_type>	_quantizerG;
};

template <class T, class W>
template <class IN, class GUIDE, class OUT> void
WeightedMedianFilter2<T, W>::convolve(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out)
{
    if (std::distance(ib, ie) < winSize() || ib->size() < winSize())
	return;

    start(0);
    const auto&	indicesI = _quantizerI(ib, ie, nbinsI());	// 入力を量子化
    const auto&	indicesG = _quantizerG(gb, ge, nbinsG());	// ガイドを量子化

    start(1);
    super::setWeights(_quantizerG);	// 重みの2次元lookup tableをセット

    start(2);
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(
			  0, std::distance(indicesI.begin(),
					   indicesI.end() + 1 - winSize()),
			  _grainSize),
		      makeFilter(indicesI.begin(), indicesG.begin(), out));
#else
    filter(indicesI.begin(), indicesI.end() + 1 - winSize(),
	   indicesG.begin(), out);
#endif
    nextFrame();
}

template <class T, class W>
template <class ROW_I, class ROW_G, class OUT> void
WeightedMedianFilter2<T, W>::filter(ROW_I rowI, ROW_I rowIe,
				    ROW_G rowG, OUT out) const
{
    HistogramArray	histograms;
    auto		guide = rowG;
    std::advance(guide, winSize()/2);
    std::advance(out,   winSize()/2);

    for (; rowI != rowIe; ++rowI)
    {
      // ヒストグラムを空にする
	histograms.resize(_quantizerI.size(), _quantizerG.size());
	
      // ウィンドウ初期位置におけるヒストグラムをセット
	for (size_t col = 0; col < winSize() - 1; ++col)
	{
	    auto	idxI = make_vertical_iterator(rowI, col);
	    auto	idxG = make_vertical_iterator(rowG, col);
	    for (size_t n = winSize(); n > 0; --n)
	    {
		histograms.add(*idxI, *idxG);
		++idxI;
		++idxG;
	    }
	}
	
      // median点を探索し，その値を出力
	size_t	median = 0;
	auto	centG = guide->begin();
	std::advance(centG, winSize()/2);
	auto	centO = out->begin();
	std::advance(centO, winSize()/2);

	for (size_t col = winSize() - 1; col < rowI->size(); )
	{

	  // current点をヒストグラムに追加	    
	    auto	idxI = make_vertical_iterator(rowI, col);
	    auto	idxG = make_vertical_iterator(rowG, col);
	    for (size_t n = winSize(); n > 0; --n)
	    {
		histograms.add(*idxI, *idxG);		// ヒストグラムに追加
		++idxI;
		++idxG;
	    }

	  // median点を検出してその値を出力
	    median = histograms.updateMedian(median, super::weights(*centG));
	    *centO = _quantizerI[median];		// 出力

	  // head点をヒストグラムから除去
	    ++col;
	    idxI = make_vertical_iterator(rowI, col - winSize());
	    idxG = make_vertical_iterator(rowG, col - winSize());
	    for (size_t n = winSize(); n > 0; --n)
	    {
		histograms.remove(*idxI, *idxG);	// ヒストグラムから除去
		++idxI;
		++idxG;
	    }

	    ++centG;
	    ++centO;
	}

	++rowG;
	++guide;
	++out;
    }
}

}
#endif
