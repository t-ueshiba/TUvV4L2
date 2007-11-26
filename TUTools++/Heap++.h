/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，使用，第三者へ開示する
 *  等の著作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: Heap++.h,v 1.5 2007-11-26 07:55:48 ueshiba Exp $
 */
#ifndef __TUHeapPP_h
#define __TUHeapPP_h

#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class Heap<T, Compare>						*
************************************************************************/
//! 複数の要素をソートするヒープを表すクラス
/*!
  テンプレートパラメータTは要素の型を指定する．テンプレートパラメータCompare
  は2つのT型要素が昇順に並んでいる時にtrueを返す関数オブジェクト型である．
  たとえば，2つのint型を比較する場合は
  \verbatim
  struct ordered
  {
    bool	operator ()(const int& item0, const int& item1) const
		{
		    return item0 < item1;
		}
  };
  \endverbatim
  と定義する．
*/
template <class T, class Compare>
class Heap
{
  public:
    Heap(u_int d, Compare compare)	;
    Heap(Array<T>& a, Compare compare)	;

  //! 現在の要素数を返す
  /*!
    \return	要素数
  */
    u_int	nelements()	const	{return _n;}

  //! ヒープ先頭の要素を返す
  /*!
    \return	先頭の要素
  */
    T		head()		const	{return (_n != 0 ? _array[0] : 0);}

    void	add(T item)		;
    T		detach()		;
    
  private:
    void	upheap(int current)	;
    void	downheap(int current)	;
    
    Array<T>		_array;
    u_int		_n;		// # of items in the heap.
    const Compare	_compare;	// function for comparing two items.
};

/************************************************************************
*  Global functions							*
************************************************************************/
template <class T, class Compare> void
sort(Array<T>& a, Compare compare)	;
 
}

#endif	// !__TUHeapPP_h
