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

#include "TU/SeparableFilter2.h"
#include <algorithm>

namespace TU
{
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
