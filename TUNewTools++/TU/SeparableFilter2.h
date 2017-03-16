/*!
  \file		SeparableFilter2.h
  \brief	水平/垂直方向に分離可能な2次元フィルタを実装するための基底クラスの定義
*/
#ifndef __TU_SEPARABLEFILTER2_H
#define __TU_SEPARABLEFILTER2_H

#include "TU/Array++.h"
#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif

namespace TU
{
//! 水平/垂直方向に分離可能な2次元フィルタを実装するための基底クラス
/*!
  \param F	1次元フィルタの型
*/
template <class F>
class SeparableFilter2
{
  public:
    using	filter_type = F;
#if defined(USE_TBB)
  private:
    template <class IN, class OUT>
    class ConvolveRows
    {
      public:
	ConvolveRows(F const& filter, IN in, OUT out, size_t col)
	    :_filter(filter), _in(in), _out(out), _col(col)		{}

	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    auto	in = _in;
		    auto	ie = _in;
		    std::advance(in, r.begin());
		    std::advance(ie, r.end());
		    auto	col = _col + r.begin();
		    for (; in != ie; ++in, ++col)
			_filter.convolve(in->begin(), in->end(),
					 make_vertical_iterator(_out, col));
		}

      private:
	F      const	_filter;  // cache等の内部状態を持ち得るので参照は不可
	IN     const	_in;
	OUT    const	_out;
	size_t const	_col;
    };
#endif
  public:
    SeparableFilter2() :_filterH(), _filterV(), _grainSize(1)	{}
    template <class ARGH, class ARGV>
    SeparableFilter2(const ARGH& argH, const ARGV& argV)
	:_filterH(argH), _filterV(argV), _grainSize(1)		{}
    
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out)	const	;
    size_t	grainSize()			const	{ return _grainSize; }
    void	setGrainSize(size_t gs)			{ _grainSize = gs; }
    F const&	filterH()			const	{ return _filterH; }
    F const&	filterV()			const	{ return _filterV; }

  protected:
    F&		filterH()				{ return _filterH; }
    F&		filterV()				{ return _filterV; }

  private:
    template <class IN, class OUT>
    void	convolveRows(F const& filter, IN ib, IN ie, OUT out,
			     size_t, std::true_type)		const	;
    template <class IN, class OUT>
    void	convolveRows(F const& filter, IN ib, IN ie, OUT out,
			     size_t col, std::false_type)	const	;
    
  private:
    F		_filterH;
    F		_filterV;
    size_t	_grainSize;
};
    
//! 与えられた2次元配列とこのフィルタの畳み込みを行う
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
*/
template <class F> template <class IN, class OUT> void
SeparableFilter2<F>::convolve(IN ib, IN ie, OUT out) const
{
    using value_type	= iterator_value<subiterator<OUT> >;
    using buf_type	= Array2<value_type>;
    using is_numeric	= typename std::is_arithmetic<value_type>::type;

    if (ib == ie)
	return;
    
    buf_type	buf(_filterH.outLength(std::distance(ib->begin(), ib->end())),
		    std::distance(ib, ie));

    convolveRows(_filterH, ib, ie, buf.begin(), 0, is_numeric());
    convolveRows(_filterV, buf.cbegin(), buf.cend(), out, 0, is_numeric());
}

template <class F> template <class IN, class OUT> inline void
SeparableFilter2<F>::convolveRows(F const& filter, IN ib, IN ie,
				  OUT out, size_t, std::true_type) const
{
    size_t	col = 0;
  //#if defined(SSE2)
#if 0
    using col_iterator	= subiterator<OUT>;
    using value_type	= iterator_value<col_iterator>;

    const auto	vsize = simd::vec<value_type>::size;
    auto	in    = ib;
    col = (std::distance(ib, ie) / vsize) * vsize;
    std::advance(ib, col);
    convolveRows(filter,
		 simd::make_row_vec_iterator<value_type>(in),
		 simd::make_row_vec_iterator<value_type>(ib),
		 make_row_iterator<simd::store_iterator<col_iterator> >(out),
		 0, std::false_type());
#endif
    convolveRows(filter, ib, ie, out, col, std::false_type());
}

template <class F> template <class IN, class OUT> inline void
SeparableFilter2<F>::convolveRows(F const& filter, IN ib, IN ie,
				  OUT out, size_t col, std::false_type) const
{
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, std::distance(ib, ie),
						 _grainSize),
		      ConvolveRows<IN, OUT>(filter, ib, out, col));
#else
    for (; ib != ie; ++ib, ++col)
	filter.convolve(ib->begin(), ib->end(),
			make_vertical_iterator(out, col));
#endif
}

}
#endif	// !__TU_SEPARABLEFILTER2_H
