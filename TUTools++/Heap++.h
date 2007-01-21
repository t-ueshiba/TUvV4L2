/*
 *  $Id: Heap++.h,v 1.3 2007-01-21 23:36:36 ueshiba Exp $
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
