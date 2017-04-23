/*!
  \file		BoxFilter.h
  \brief	box filterに関するクラスの定義と実装
*/
#ifndef	__TU_BOXFILTER_H
#define	__TU_BOXFILTER_H

#include "TU/Filter2.h"
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class box_filter_iterator<ITER, T>					*
************************************************************************/
//! コンテナ中の指定された要素に対してbox filterを適用した結果を返す反復子
/*!
  \param ITER	コンテナ中の要素を指す定数反復子の型
*/
template <class ITER, class T=iterator_substance<ITER> >
class box_filter_iterator
    : public boost::iterator_adaptor<box_filter_iterator<ITER, T>,
				     ITER,			// base
				     T,				// value_type
				     boost::single_pass_traversal_tag>
{
  private:
    using super = boost::iterator_adaptor<box_filter_iterator,
					  ITER,
					  T,
					  boost::single_pass_traversal_tag>;
		    
  public:
    using	typename super::value_type;
    using	typename super::reference;

    friend class	boost::iterator_core_access;

  public:
		box_filter_iterator()
		    :super(), _head(super::base()), _val(), _valid(true)
		{
		}
    
		box_filter_iterator(const ITER& iter, size_t w=0)
		    :super(iter), _head(iter), _val(), _valid(true)
		{
		    if (w > 0)
		    {
			_val = *super::base();
				
			while (--w > 0)
			    _val += *++super::base_reference();
		    }
		}

    void	initialize(const ITER& iter, size_t w=0)
		{
		    super::base_reference() = iter;
		    _head = iter;
		    _valid = true;

		    if (w > 0)
		    {
			_val = *super::base();
				
			while (--w > 0)
			    _val += *++super::base_reference();
		    }
		}
    
  private:
    reference	dereference() const
		{
		    if (!_valid)
		    {
			_val += (*super::base() - *_head);
			++_head;
			_valid = true;
		    }
		    return _val;
		}
    
    void	increment()
		{
		  // dereference() せずに increment() する可能性があ
		  // るなら次のコードを有効化する．ただし，性能は低下．
#ifdef TU_BOX_FILTER_ITERATOR_CONSERVATIVE
		    if (!_valid)
		    {
			_val += (*super::base() - *_head);
			++_head;
		    }
		    else
#endif
			_valid = false;
		    ++super::base_reference();
		}

  private:
    mutable ITER	_head;
    mutable value_type	_val;	// [_head, base()) or [_head, base()] の総和
    mutable bool	_valid;	// _val が [_head, base()] の総和ならtrue
};

//! box filter反復子を生成する
/*!
  \param iter	コンテナ中の要素を指す定数反復子
  \return	box filter反復子
*/
template <class T=void, class ITER>
box_filter_iterator<ITER, std::conditional_t<std::is_void<T>::value,
					     iterator_substance<ITER>, T> >
make_box_filter_iterator(ITER iter, size_t w=0)
{
    return {iter, w};
}

/************************************************************************
*  class BoxFilter							*
************************************************************************/
//! 1次元入力データ列にbox filterを適用するクラス
class BoxFilter
{
  public:
  //! box filterを生成する．
  /*!
    \param w	box filterのウィンドウ幅
   */	
		BoxFilter(size_t w=3) :_winSize(w)	{}
    
  //! box filterのウィンドウ幅を設定する．
  /*!
    \param w	box filterのウィンドウ幅
    \return	このbox filter
   */
    BoxFilter&	setWinSize(size_t w)		{_winSize = w; return *this;}

  //! box filterのウィンドウ幅を返す．
  /*!
    \return	box filterのウィンドウ幅
   */
    size_t	winSize()		const	{return _winSize;}

    template <class T=void, class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out)	const	;

  //! 与えられた長さの入力データ列に対する出力データ列の長さを返す
  /*!
    \param inLen	入力データ列の長さ
    \return		出力データ列の長さ
   */
    size_t	outLength(size_t inLen)	const	{return inLen + 1 - _winSize;}
	
  private:
    size_t	_winSize;		//!< box filterのウィンドウ幅
};
    
/*!
  \param ib	1次元入力データ列の先頭を示す反復子
  \param ie	1次元入力データ列の末尾の次を示す反復子
  \param out	box filterを適用した出力データ列の先頭を示す反復子
  \return	出力データ列の末尾の次を示す反復子
*/
template <class T, class IN, class OUT> void
BoxFilter::convolve(IN ib, IN ie, OUT out) const
{
    std::copy(make_box_filter_iterator<T>(ib, _winSize),
	      make_box_filter_iterator<T>(ie), out);
}

/************************************************************************
*  class BoxFilter2							*
************************************************************************/
//! 2次元入力データ列にbox filterを適用するクラス
class BoxFilter2 : public Filter2<BoxFilter2>
{
  private:
    using super	= Filter2<BoxFilter2>;
    
  public:
    using	super::grainSize;
    using	super::setGrainSize;
    
  //! box filterを生成する．
  /*!
    \param wrow	box filterのウィンドウの行幅(高さ)
    \param wcol	box filterのウィンドウの列幅(幅)
   */	
		BoxFilter2(size_t wrow=3, size_t wcol=3)
		    :super(*this), _rowWinSize(wrow), _colFilter(wcol)
		{
		    if (grainSize() < 2*_rowWinSize)
			setGrainSize(2*_rowWinSize);
		}
    
  //! box filterのウィンドウの行幅(高さ)を設定する．
  /*!
    \param wrow	box filterのウィンドウの行幅
    \return	このbox filter
   */
    BoxFilter2&	setRowWinSize(size_t wrow)
		{
		    _rowWinSize = wrow;
		    if (grainSize() < 2*_rowWinSize)
			setGrainSize(2*_rowWinSize);
		    return *this;
		}

  //! box filterのウィンドウの列幅(幅)を設定する．
  /*!
    \param wcol	box filterのウィンドウの列幅
    \return	このbox filter
   */
    BoxFilter2&	setColWinSize(size_t wcol)
		{
		    _colFilter.setWinSize(wcol);
		    return *this;
		}

  //! box filterのウィンドウ行幅(高さ)を返す．
  /*!
    \return	box filterのウィンドウの行幅
   */
    size_t	rowWinSize()	const	{ return _rowWinSize; }

  //! box filterのウィンドウ列幅(幅)を返す．
  /*!
    \return	box filterのウィンドウの列幅
   */
    size_t	colWinSize()	const	{ return _colFilter.winSize(); }

  //! 与えられた行幅(高さ)を持つ入力データ列に対する出力データ列の行幅を返す．
  /*!
    \param inRowLength	入力データ列の行幅
    \return		出力データ列の行幅
   */
    size_t	outRowLength(size_t inRowLength) const
		{
		    return inRowLength + 1 - rowWinSize();
		}
    
  //! 与えられた列幅(幅)を持つ入力データ列に対する出力データ列の列幅を返す．
  /*!
    \param inColLength	入力データ列の列幅
    \return		出力データ列の列幅
   */
    size_t	outColLength(size_t inColLength) const
		{
		    return inColLength + 1 - colWinSize();
		}

    size_t	overlap()	const	{ return rowWinSize() - 1; }

    template <class IN, class OUT>
    void	convolveRows(IN ib, IN ie, OUT out)	const	;
    
  private:
    size_t	_rowWinSize;
    BoxFilter	_colFilter;
};

//! 与えられた2次元配列とこのフィルタの畳み込みを行う
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
*/
template <class IN, class OUT> void
BoxFilter2::convolveRows(IN ib, IN ie, OUT out) const
{
    if (std::distance(ib, ie) < rowWinSize())
	throw std::runtime_error("BoxFilter2::convolveRows(): not enough rows!");
    
    for (box_filter_iterator<IN> row(ib, _rowWinSize), rowe(ie);
	 row != rowe; ++row, ++out)
	_colFilter.convolve(std::begin(*row), std::end(*row), std::begin(*out));
}

}
#endif	// !__TU_BOXFILTER_H
