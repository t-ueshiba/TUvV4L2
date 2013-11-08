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
#ifndef __TUSeparableFilter2_h
#define __TUSeparableFilter2_h

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
    template <class IN, class OUT> class ConvolveRows
    {
      public:
	ConvolveRows(F const& filter, IN const& in, OUT const& out)
	    :_filter(filter), _in(in), _out(out)			{}

	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    typedef typename subiterator<OUT>::value_type
								value_type;
		    typedef typename boost::is_arithmetic<value_type>
					  ::type		is_numeric;
		    
		    IN	ib = _in, ie = _in;
		    std::advance(ib, r.begin());
		    std::advance(ie, r.end());
		    SeparableFilter2<F>::convolveRows(_filter, ib, ie, _out,
						      r.begin(), is_numeric());
		}

      private:
	F      const	_filter;  // cache等の内部状態を持ち得るので参照は不可
	IN     const&	_in;
	OUT    const&	_out;
    };
#endif
#if defined(SSE)
    template <class OUT>
    struct row2vcol
    {
      public:
	typedef typename subiterator<OUT>::value_type		value_type;
	typedef typename std::iterator_traits<OUT>::reference	argument_type;
	typedef mm::detail::store_proxy<value_type>		result_type;
    
      public:
	row2vcol(size_t idx)	:_idx(idx)			{}
    
	result_type	operator ()(argument_type row) const
			{
			    return result_type(row.begin() + _idx);
			}
    
      private:
	size_t const	_idx;	//!< 列を指定するindex
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
    static void	convolveRows(F const& filter, IN ib, IN ie,
			     OUT out, size_t col, boost::true_type)	;
    template <class IN, class OUT>
    static void	convolveRows(F const& filter, IN ib, IN ie,
			     OUT out, size_t col, boost::false_type)	;
    
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
    typedef typename subiterator<OUT>::value_type	value_type;
    typedef typename boost::is_arithmetic<value_type>::type
							is_numeric;
    typedef Array2<Array<value_type> >			buf_type;
    typedef typename buf_type::iterator			buf_iterator;
    typedef typename buf_type::const_iterator		const_buf_iterator;

    if (ib == ie)
	return;
    
    buf_type	buf(_filterH.outLength(std::distance(ib->begin(), ib->end())),
		    std::distance(ib, ie));
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, buf.ncol(), _grainSize),
		      ConvolveRows<IN, buf_iterator>(
			  _filterH, ib, buf.begin()));
    tbb::parallel_for(tbb::blocked_range<size_t>(0, buf.nrow(), _grainSize),
		      ConvolveRows<const_buf_iterator, OUT>(
			  _filterV, buf.begin(), out));
#else
    convolveRows(_filterH, ib, ie, buf.begin(), 0, is_numeric());
    convolveRows(_filterV, buf.begin(), buf.end(), out, 0, is_numeric());
#endif
}

template <class F> template <class IN, class OUT> void
SeparableFilter2<F>::convolveRows(F const& filter, IN ib, IN ie, OUT out,
				  size_t col, boost::true_type)
{
#if defined(SSE2)
    typedef typename subiterator<OUT>::value_type		value_type;
    typedef mm::vec<value_type>					vec_type;
    typedef mm::row_vec_iterator<value_type, IN>		in_iterator;
    typedef boost::transform_iterator<
	row2vcol<OUT>, OUT, boost::use_default, vec_type>	out_iterator;

    const size_t	vsize = vec_type::size;
    
    IN	in = ib;
    std::advance(ib, (std::distance(ib, ie) / vsize) * vsize);
    for (in_iterator vec_in(in), vec_ib(ib); vec_in != vec_ib;
	 ++vec_in, col += vsize)
	filter.convolve(vec_in->begin(), vec_in->end(),
			out_iterator(out, row2vcol<OUT>(col)));
#endif
    convolveRows(filter, ib, ie, out, col, boost::false_type());
}

template <class F> template <class IN, class OUT> void
SeparableFilter2<F>::convolveRows(F const& filter, IN ib, IN ie, OUT out,
				  size_t col, boost::false_type)
{
    for (; ib != ie; ++ib, ++col)
	filter.convolve(ib->begin(), ib->end(),
			make_vertical_iterator(out, col));
}

}
#endif	// !__TUSeparableFilter2_h
