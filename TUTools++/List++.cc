/*
 *  $Id: List++.cc,v 1.2 2002-07-25 02:38:05 ueshiba Exp $
 */
#include "TU/List++.h"

namespace TU
{
/************************************************************************
*  class List<T>							*
************************************************************************/
/*
 *  'i' の位置に 'x' を挿入し、その挿入された位置を返す。
 */
template <class T> typename List<T>::Iterator
List<T>::insert(Iterator i, T& x)
{
    if (i == end())				// 末尾に挿入？
	_back = &x;

    if (i == begin())				// リストの先頭？
    {
	x._next = _front;			// 先頭に挿入
	_front = &x;
    }
    else
	i._prev->insertNext(&x);		// 「手前の次」に挿入

    return i;
}

/*
 *  'i' の位置にある要素を削除し、削除された要素への参照を返す。
 */
template <class T> typename List<T>::reference
List<T>::erase(Iterator i)
{
    T&	x = *i;
    if (&x == _back)				// リストの末尾？
	_back = i._prev;			// 末尾の要素を削除
    if (&x == _front)				// リストの先頭？
	_front = _front->_next;			// 先頭の要素を削除
    else
	i._prev->eraseNext();			// 「手前の次」を削除
    
    return x;
}

/*
 *  'x' と同じオブジェクト（高々１つしかないはず）を削除する。
 */
template <class T> void
List<T>::remove(const T& x)
{
    for (Iterator i = begin(); i != end(); )
    {
	Iterator next = i;
	++next;
	if (i.operator ->() == &x)
	{
	    erase(i);
	    return;
	}
	i = next;
    }
}
 
}
