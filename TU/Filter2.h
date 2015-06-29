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
  \brief	水平/垂直方向に分離不可能な2次元フィルタを表すクラスの定義
*/
#ifndef __TU_FILTER2_H
#define __TU_FILTER2_H

#include "TU/iterator.h"
#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif

namespace TU
{
/************************************************************************
*  class Filter2<F>							*
************************************************************************/
//! 水平/垂直方向に分離不可能な2次元フィルタを表すクラス
/*!
  \param F	2次元フィルタの型
*/
template <class F>
class Filter2
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
		     IN const& ib, IN const& ie, OUT const& out)
	    :_filter(filter), _ib(ib), _ie(ie), _out(out)		{}

	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    IN	ib = _ib, ie = _ib;
		    std::advance(ib, r.begin());
		    std::advance(ie, r.end() + _filter.overlap());
		    OUT	out = _out;
		    std::advance(out, r.begin());
		    _filter.convolveRows(ib, std::min(ie, _ie), out);
		}

      private:
	F	const&	_filter;
	IN	const&	_ib;
	IN	const&	_ie;
	OUT	const&	_out;
    };
#endif
  public:
    template <class ...ARG>
    Filter2(const ARG& ...arg)	:_filter(arg...), _grainSize(1)	{}
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out)	const	;
    size_t	grainSize()			const	{ return _grainSize; }
    void	setGrainSize(size_t gs)			{ _grainSize = gs; }
	
  private:
    F const&	_filter;	// 2次元フィルタ
    size_t	_grainSize;
};
    
//! 与えられた2次元配列にこのフィルタを適用する
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
*/
template <class F> template <class IN, class OUT> void
Filter2<F>::convolve(IN ib, IN ie, OUT out) const
{
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, std::distance(ib, ie),
						 _grainSize),
		      ConvolveRows<IN, OUT>(_filter, ib, ie, out));
#else
    _filter.convolveRows(ib, ie, out);
#endif
}

}
#endif	// !__TU_FILTER2_H
