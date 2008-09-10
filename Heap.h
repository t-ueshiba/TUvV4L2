/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2002-2007.
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
 *  $Id: Heap.h,v 1.2 2008-09-10 05:10:38 ueshiba Exp $
 */
#ifndef __TUHeap_h
#define __TUHeap_h

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

//! 空のヒープを作る
/*!
  \param d		ヒープの最大要素数
  \param compare	比較関数オブジェクト
*/
template <class T, class Compare>
Heap<T, Compare>::Heap(u_int d, Compare compare)
    :_array(d), _n(0), _compare(compare)
{
}

//! 1次元配列からヒープを作る
/*!
  \param a		1次元配列
  \param compare	比較関数オブジェクト
*/
template <class T, class Compare>
Heap<T, Compare>::Heap(Array<T>& a, Compare compare)
    :_array((T*)a, a.dim()), _n(a.dim()), _compare(compare)
{
    for (int i = _n / 2; --i >= 0; )
	downheap(i);
}

//! ヒープに要素を追加する
/*!
  \param item	追加する要素
*/
template <class T, class Compare> void
Heap<T, Compare>::add(T item)
{
    if (_n >= _array.dim())
	throw std::length_error("TU::Heap<T, Compare>::add: Heap is full!!");

    _array[_n] = item;
    upheap(_n++);
}

//! ヒープ先頭の要素を取り除く
/*!
  \return	取り除かれた要素
*/
template <class T, class Compare> T
Heap<T, Compare>::detach()
{
    if (_n == 0)
	throw
	    std::out_of_range("TU::Heap<T, Compare>::detach: Heap is empty!!");
    
    T	val = _array[0];			// Store the root value.
    _array[0] = _array[--_n];			// Move the last to the root.
    downheap(0);
    return val;
}

template <class T, class Compare> void
Heap<T, Compare>::upheap(int current)
{
    T	val = _array[current];
    while (current > 0)				// While having a parent...
    {
	int	parent = (current - 1) / 2;	// Index of the parent node.
	if (_compare(val, _array[parent]))
	    break;
	
	_array[current] = _array[parent];	// Pull-down the parent.
	current = parent;			// Go upward.
    }
    _array[current] = val;
}

template <class T, class Compare> void
Heap<T, Compare>::downheap(int current)
{
    T	val = _array[current];
    for (int child; (child = 2 * current + 1) < _n; )
    {
	if (child + 1 < _n && _compare(_array[child], _array[child + 1]))
	    ++child;				// Choose larger child.

	if (_compare(_array[child], val))
	    break;

	_array[current] = _array[child];	// Pull-up the child.
	current = child;			// Go downward.
    }
    _array[current] = val;
}

/************************************************************************
*  Global functions							*
************************************************************************/
//! 1次元配列のヒープソートを行う
/*!
  \param a		ソートする配列
  \param compare	比較関数オブジェクト
*/
template <class T, class Compare> void
sort(Array<T>& a, Compare compare)
{
    Heap<T, Compare>	heap(a, compare);
    for (int i = a.dim(); i > 0; )
	a[--i] = heap.detach();
}
 
}

#endif	// !__TUHeap_h
