/*
 *  平成14-24年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2002-2012.
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
    typedef F	filter_type;
#if defined(USE_TBB)
  private:
    template <class IN, class OUT>
    class ConvolveRows
    {
      public:
	ConvolveRows(F const& filter,
		     IN const& in, OUT const& out, size_t col)
	    :_filter(filter), _in(in), _out(out), _col(col)		{}

	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    IN		in = _in, ie = _in;
		    std::advance(in, r.begin());
		    std::advance(ie, r.end());
		    size_t	col = _col + r.begin();
		    for (; in != ie; ++in, ++col)
			_filter.convolve(in->begin(), in->end(),
					 make_vertical_iterator(_out, col));
		}

      private:
	F      const	_filter;  // cache等の内部状態を持ち得るので参照は不可
	IN     const&	_in;
	OUT    const&	_out;
	size_t const	_col;
    };
#endif
  public:
    SeparableFilter2() :_filterH(), _filterV(), _grainSize(1)	{}

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
    typedef typename std::iterator_traits<subiterator<OUT> >::value_type
								value_type;
    typedef Array2<Array<value_type> >				buf_type;
    typedef typename std::is_arithmetic<value_type>::type	is_numeric;

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
#if defined(SSE2)
    typedef subiterator<OUT>					col_iterator;
    typedef typename std::iterator_traits<col_iterator>::value_type
								value_type;

    const size_t	vsize = mm::vec<value_type>::size;
    IN			in    = ib;
    col = (std::distance(ib, ie) / vsize) * vsize;
    std::advance(ib, col);
    convolveRows(filter,
		 mm::make_row_vec_iterator<value_type>(in),
		 mm::make_row_vec_iterator<value_type>(ib),
		 make_row_iterator<mm::store_iterator<col_iterator> >(out),
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
