/*!
  \file		Filter2.h
  \author	Toshio UESHIBA
  \brief	水平/垂直方向に分離不可能な2次元フィルタを表すクラスの定義
*/
#ifndef TU_FILTER2_H
#define TU_FILTER2_H

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
    using filter_type	= F;
#if defined(USE_TBB)
  private:
    template <class IN_, class OUT_>
    class ConvolveRows
    {
      public:
	ConvolveRows(F const& filter, IN_ ib, IN_ ie, OUT_ out, bool shift)
	    :_filter(filter), _ib(ib), _ie(ie), _out(out), _shift(shift)  {}

	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    auto	ib = _ib;
		    auto	ie = _ib;
		    std::advance(ib, r.begin());
		    std::advance(ie, r.end() + _filter.overlap());
		    auto	out = _out;
		    std::advance(out, r.begin());
		    _filter.convolveRows(ib, std::min(ie, _ie), out, _shift);
		}

      private:
	const F&	_filter;
	const IN_	_ib;
	const IN_	_ie;
	const OUT_	_out;
	const bool	_shift;
    };
#endif
  public:
    template <class ...ARG_>
    Filter2(const ARG_& ...arg)	:_filter(arg...), _grainSize(1)	{}
    template <class IN_, class OUT_>
    void	convolve(IN_ ib, IN_ ie, OUT_ out,
			 bool shift=false)	const	;
    size_t	grainSize()			const	{ return _grainSize; }
    void	setGrainSize(size_t gs)			{ _grainSize = gs; }
	
  private:
    const F&	_filter;	// 2次元フィルタ
    size_t	_grainSize;
};
    
//! 与えられた2次元配列にこのフィルタを適用する
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \param shift	trueならば，入力データと対応するよう，出力位置を水平/垂直
		方向にそれぞれ offsetH(), offsetV() だけシフトする
*/
template <class F> template <class IN_, class OUT_> void
Filter2<F>::convolve(IN_ ib, IN_ ie, OUT_ out, bool shift) const
{
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, std::distance(ib, ie),
						 _grainSize),
		      ConvolveRows<IN_, OUT_>(_filter, ib, ie, out, shift));
#else
    _filter.convolveRows(ib, ie, out, shift);
#endif
}

}
#endif	// !TU_FILTER2_H
