/*
 *  平成19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  同所が著作権を所有する秘密情報です．著作者による許可なしにこのプロ
 *  グラムを第三者へ開示，複製，改変，使用する等の著作権を侵害する行為
 *  を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *  Copyright 2007
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Author: Toshio UESHIBA
 *
 *  Confidentail and all rights reserved.
 *  This program is confidential. Any changing, copying or giving
 *  information about the source code of any part of this software
 *  and/or documents without permission by the authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damages in the use of this program.
 *  
 *  $Id: List++.cc,v 1.3 2007-11-26 07:28:09 ueshiba Exp $
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
