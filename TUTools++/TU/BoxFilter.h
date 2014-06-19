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
  \file		BoxFilter.h
  \brief	box filterに関するクラスの定義と実装
*/
#ifndef	__TUBoxFilter_h
#define	__TUBoxFilter_h

#include <algorithm>
#include "TU/SeparableFilter2.h"

namespace TU
{
/************************************************************************
*  class box_filter_iterator<ITER, VAL>					*
************************************************************************/
//! コンテナ中の指定された要素に対してbox filterを適用した結果を返す反復子
/*!
  \param ITER	コンテナ中の要素を指す定数反復子の型
*/
template <class ITER, class VAL=typename std::iterator_traits<ITER>::value_type>
class box_filter_iterator
    : public boost::iterator_adaptor<box_filter_iterator<ITER, VAL>,
				     ITER,			// base
				     VAL,			// value_type
				     boost::single_pass_traversal_tag>
{
  private:
    typedef boost::iterator_adaptor<box_filter_iterator,
				    ITER,
				    VAL,
				    boost::single_pass_traversal_tag>	super;
		    
  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
		box_filter_iterator()
		    :super(), _head(super::base()), _val(), _valid(true)
		{
		}
    
		box_filter_iterator(ITER const& iter, size_t w=0)
		    :super(iter), _head(iter), _val(), _valid(true)
		{
		    if (w > 0)
		    {
			_val = *super::base();
				
			while (--w > 0)
			    _val += *++super::base_reference();
		    }
		}

    void	initialize(ITER const& iter, size_t w=0)
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
    typedef boost::mpl::bool_<detail::is_container<value_type>::value>
								value_is_expr;

    template <class _VITER, class _ITER>
    static void	update(_VITER val, _ITER curr, _ITER head, boost::mpl::false_)
		{
		    *val += (*curr - *head);
		}
    template <class _VITER, class _ITER>
    static void	update(_VITER val, _ITER curr, _ITER head, boost::mpl::true_)
		{
		    typedef typename std::iterator_traits<_ITER>::value_type
					::const_iterator	const_iterator;
		    typedef typename subiterator<_VITER>::type	iterator;
		    typedef boost::mpl::bool_<
			detail::is_container<
			    typename subiterator<_VITER>::value_type>::value>
								value_is_expr;
		    
		    const_iterator	c = curr->cbegin(), h = head->cbegin();
		    for (iterator v = val->begin(), ve = val->end();
			 v != ve; ++v, ++c, ++h)
			update(v, c, h, value_is_expr());
		}

    reference	dereference() const
		{
		    if (!_valid)
		    {
			update(&_val, super::base(), _head, value_is_expr());
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
			update(&_val, super::base(), _head, value_is_expr());
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
template <class ITER> box_filter_iterator<ITER>
make_box_filter_iterator(ITER iter, size_t w=0)
{
    return box_filter_iterator<ITER>(iter, w);
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

    template <class IN, class OUT>
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
template <class IN, class OUT> void
BoxFilter::convolve(IN ib, IN ie, OUT out) const
{
    std::copy(make_box_filter_iterator(ib, _winSize),
	      make_box_filter_iterator(ie), out);
}

/************************************************************************
*  class BoxFilter2							*
************************************************************************/
//! 2次元入力データ列にbox filterを適用するクラス
class BoxFilter2 : public SeparableFilter2<BoxFilter>
{
  public:
  //! box filterを生成する．
  /*!
    \param wrow	box filterのウィンドウの行幅(高さ)
    \param wcol	box filterのウィンドウの列幅(幅)
   */	
		BoxFilter2(size_t wrow=3, size_t wcol=3)
		{
		    setRowWinSize(wrow).setColWinSize(wcol);
		}
    
  //! box filterのウィンドウの行幅(高さ)を設定する．
  /*!
    \param wrow	box filterのウィンドウの行幅
    \return	このbox filter
   */
    BoxFilter2&	setRowWinSize(size_t wrow)
		{
		    filterV().setWinSize(wrow);
		    return *this;
		}

  //! box filterのウィンドウの列幅(幅)を設定する．
  /*!
    \param wcol	box filterのウィンドウの列幅
    \return	このbox filter
   */
    BoxFilter2&	setColWinSize(size_t wcol)
		{
		    filterH().setWinSize(wcol);
		    return *this;
		}

  //! box filterのウィンドウ行幅(高さ)を返す．
  /*!
    \return	box filterのウィンドウの行幅
   */
    size_t	rowWinSize()	const	{return filterV().winSize();}

  //! box filterのウィンドウ列幅(幅)を返す．
  /*!
    \return	box filterのウィンドウの列幅
   */
    size_t	colWinSize()	const	{return filterH().winSize();}

  //! 与えられた行幅(高さ)を持つ入力データ列に対する出力データ列の行幅を返す．
  /*!
    \param inRowLength	入力データ列の行幅
    \return		出力データ列の行幅
   */
    size_t	outRowLength(size_t inRowLength) const
		{
		    return filterV().outLength(inRowLength);
		}
    
  //! 与えられた列幅(幅)を持つ入力データ列に対する出力データ列の列幅を返す．
  /*!
    \param inColLength	入力データ列の列幅
    \return		出力データ列の列幅
   */
    size_t	outColLength(size_t inColLength) const
		{
		    return filterH().outLength(inColLength);
		}
};

}
#endif	/* !__TUBoxFilter_h	*/
