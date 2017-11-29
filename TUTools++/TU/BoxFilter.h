/*!
  \file		BoxFilter.h
  \author	Toshio UESHIBA
  \brief	box filterに関するクラスの定義と実装
*/
#ifndef	TU_BOXFILTER_H
#define	TU_BOXFILTER_H

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
template <class ITER, class T=void>
class box_filter_iterator
    : public boost::iterator_adaptor<box_filter_iterator<ITER, T>,
				     ITER,
				     replace_element<iterator_substance<ITER>,
						     T>,
				     boost::single_pass_traversal_tag>
{
  private:
    using super = boost::iterator_adaptor<
			box_filter_iterator,
			ITER,
			replace_element<iterator_substance<ITER>, T>,
			boost::single_pass_traversal_tag>;
		    
  public:
    using	typename super::value_type;
    using	typename super::reference;

    friend	class boost::iterator_core_access;

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
		      //(_val += *super::base()) -= *_head;
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
		      //(_val += *super::base()) -= *_head;
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
template <class T=void, class ITER> box_filter_iterator<ITER, T>
make_box_filter_iterator(const ITER& iter, size_t w=0)
{
    return {iter, w};
}

/************************************************************************
*  class BoxFilter<T>							*
************************************************************************/
//! 1次元入力データ列にbox filterを適用するクラス
template <class T>
class BoxFilter
{
  public:
    using element_type	= T;
    
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

    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out, bool shift=false) const	;

  //! 与えられた長さの入力データ列に対する出力データ列の長さを返す
  /*!
    \param inSize	入力データ列の長さ
    \return		出力データ列の長さ
   */
    size_t	outSize(size_t inSize)	const	{return inSize + 1 - _winSize;}

  //! 入力データ列と対応づけるために出力データ列をシフトすべき量を返す
  /*!
    \return		出力データ列のシフト量
  */
    size_t	offset()		const	{return _winSize/2;}
	
  private:
    size_t	_winSize;		//!< box filterのウィンドウ幅
};

//! 与えられた1次元配列とこのフィルタの畳み込みを行う
/*!
  \param ib	1次元入力データ列の先頭を示す反復子
  \param ie	1次元入力データ列の末尾の次を示す反復子
  \param out	box filterを適用した出力データ列の先頭を示す反復子
  \param shift	true ならば，入力データと対応するよう，出力位置を
		offset() だけシフトする
  \return	出力データ列の末尾の次を示す反復子
*/
template <class T> template <class IN, class OUT> void
BoxFilter<T>::convolve(IN ib, IN ie, OUT out, bool shift) const
{
    if (shift)
	std::advance(out, offset());
    
    std::copy(make_box_filter_iterator<T>(ib, _winSize),
	      make_box_filter_iterator<T>(ie), out);
}

/************************************************************************
*  class BoxFilter2<T>							*
************************************************************************/
//! 2次元入力データ列にbox filterを適用するクラス
template <class T>
class BoxFilter2 : public Filter2<BoxFilter2<T> >
{
  private:
    using super	= Filter2<BoxFilter2<T> >;
    
  public:
    using element_type	= T;
    using super::grainSize;
    using super::setGrainSize;
    
  //! box filterを生成する．
  /*!
    \param wrow	box filterのウィンドウの行幅(高さ)
    \param wcol	box filterのウィンドウの列幅(幅)
   */	
		BoxFilter2(size_t wrow=3, size_t wcol=3)
		    :super(*this), _winSizeV(wrow), _colFilter(wcol)
		{
		    if (grainSize() < 2*_winSizeV)
			setGrainSize(2*_winSizeV);
		}
    
  //! box filterのウィンドウの行幅(高さ)を設定する．
  /*!
    \param wrow	box filterのウィンドウの行幅
    \return	このbox filter
   */
    BoxFilter2&	setWinSizeV(size_t wrow)
		{
		    _winSizeV = wrow;
		    if (grainSize() < 2*_winSizeV)
			setGrainSize(2*_winSizeV);
		    return *this;
		}

  //! box filterのウィンドウの列幅(幅)を設定する．
  /*!
    \param wcol	box filterのウィンドウの列幅
    \return	このbox filter
   */
    BoxFilter2&	setWinSizeH(size_t wcol)
		{
		    _colFilter.setWinSize(wcol);
		    return *this;
		}

  //! box filterのウィンドウ行幅(高さ)を返す．
  /*!
    \return	box filterのウィンドウの行幅
   */
    size_t	winSizeV()		const	{return _winSizeV;}

  //! box filterのウィンドウ列幅(幅)を返す．
  /*!
    \return	box filterのウィンドウの列幅
   */
    size_t	winSizeH()		const	{return _colFilter.winSize();}

  //! 与えられた行幅(高さ)を持つ入力データ列に対する出力データ列の行幅を返す．
  /*!
    \param inNrow	入力データ列の行幅
    \return		出力データ列の行幅
   */
    size_t	outSizeV(size_t nrow)	const	{return nrow + 1 - _winSizeV;}
    
  //! 与えられた列幅(幅)を持つ入力データ列に対する出力データ列の列幅を返す．
  /*!
    \param inNcol	入力データ列の列幅
    \return		出力データ列の列幅
   */
    size_t	outSizeH(size_t ncol)	const	{return _colFilter.outSize(ncol);}

  //! 入力データ列と対応づけるために出力データ列を行方向にシフトすべき量を返す
  /*!
    \return		出力データ列の行方向シフト量
  */
    size_t	offsetV()		const	{return _winSizeV/2;}
	
  //! 入力データ列と対応づけるために出力データ列を列方向にシフトすべき量を返す
  /*!
    \return		出力データ列の列方向シフト量
  */
    size_t	offsetH()		const	{return _colFilter.offset();}
	
    size_t	overlap()		const	{return _winSizeV - 1;}

    template <class IN, class OUT>
    void	convolveRows(IN ib, IN ie, OUT out, bool shift)	const	;
    
  private:
    size_t		_winSizeV;
    BoxFilter<T>	_colFilter;
};

//! 与えられた2次元配列とこのフィルタの畳み込みを行う
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \param shift	trueならば，入力データと対応するよう，出力位置を水平/垂直
		方向にそれぞれ offsetH(), offsetV() だけシフトする
*/
template <class T> template <class IN, class OUT> void
BoxFilter2<T>::convolveRows(IN ib, IN ie, OUT out, bool shift) const
{
    if (std::distance(ib, ie) < winSizeV())
	throw std::runtime_error("BoxFilter2::convolveRows(): not enough rows!");

    if (shift)
	std::advance(out, offsetV());
    
    for (box_filter_iterator<IN, T> row(ib, _winSizeV), rowe(ie);
	 row != rowe; ++row, ++out)
	_colFilter.convolve(std::cbegin(*row), std::cend(*row),
			    TU::begin(*out), shift);
}

}
#endif	// !TU_BOXFILTER_H
