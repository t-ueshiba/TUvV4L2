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
  \file		Filter2.h
  \brief	各種2次元フィルタを実装するための基底クラスの定義
*/
#ifndef __FILTER2_H
#define __FILTER2_H

#include "TU/iterator.h"
#include "TU/Array++.h"
#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif

namespace TU
{
template <class F>
class Filter2
{
  public:
    typedef F	filter_type;
#if defined(USE_TBB)
  private:
    template <class IN, class OUT> class ConvolveRows
    {
      public:
	ConvolveRows(F const& filter, size_t shift,
		     IN const& in, OUT const& out)
	    :_filter(filter), _shift(shift), _in(in), _out(out)		{}

	void	operator ()(const tbb::blocked_range<u_int>& r) const
		{
		    IN	row = _in;
		    std::advance(row, r.begin());
		    for (u_int i = r.begin(); i != r.end(); ++i, ++row)
			_filter.convolve(row->begin(), row->end(),
					 make_vertical_iterator(_out,
								_shift + i));
		}

      private:
	F      const	_filter;  // cache等の内部状態を持ち得るので参照は不可
	size_t const	_shift;
	IN     const&	_in;
	OUT    const&	_out;
    };
#endif
  public:
    Filter2()	:_filterH(), _filterV(), _shift(0) , _grainSize(1)	{}

    template <class IN, class OUT,
  	      class BVAL=typename std::iterator_traits<
			     typename std::iterator_traits<OUT>
			     ::value_type::iterator>::value_type>
    void	convolve(IN ib, IN ie, OUT out)	const	;
    size_t	shift()				const	{ return _shift; }
    void	setShift(size_t s)			{ _shift = s; }
    size_t	grainSize()			const	{ return _grainSize; }
    void	setGrainSize(size_t gs)			{ _grainSize = gs; }
    F const&	filterH()			const	{ return _filterH; }
    F const&	filterV()			const	{ return _filterV; }

  protected:
    F&		filterH()				{ return _filterH; }
    F&		filterV()				{ return _filterV; }

  private:
    F		_filterH;
    F		_filterV;
    size_t	_shift;		// 出力データの水平方向書き込み位置のずらし量
    size_t	_grainSize;
};
    
//! 与えられた2次元配列とこのフィルタの畳み込みを行う
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
*/
template <class F> template <class IN, class OUT, class BVAL> void
Filter2<F>::convolve(IN ib, IN ie, OUT out) const
{
    typedef Array2<Array<BVAL> >		buf_type;
    typedef typename buf_type::iterator		buf_iterator;
    typedef typename buf_type::const_iterator	const_buf_iterator;
    
    buf_type	buf((ib != ie ?
		     _filterH.outLength(std::distance(ib->begin(), ib->end())) :
		     0),
		    std::distance(ib, ie));
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<u_int>(0, buf.ncol(), _grainSize),
		      ConvolveRows<IN, buf_iterator>(
			  _filterH, 0, ib, buf.begin()));
    tbb::parallel_for(tbb::blocked_range<u_int>(0, buf.nrow(), _grainSize),
		      ConvolveRows<const_buf_iterator, OUT>(
			  _filterV, _shift, buf.begin(), out));
#else
    std::size_t	n = 0;
    for (; ib != ie; ++ib)
	_filterH.convolve(ib->begin(), ib->end(),
			  make_vertical_iterator(buf.begin(), n++));
    n = _shift;
    for (const_buf_iterator brow = buf.begin(); brow != buf.end(); ++brow)
	_filterV.convolve(brow->begin(), brow->end(),
			  make_vertical_iterator(out, n++));
#endif
}
    
}
#endif	// !__FILTER2_H
