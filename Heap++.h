/*
 *  $Id: Heap++.h,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#ifndef __TUHeapPP_h
#define __TUHeapPP_h

#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class Heap<T, Compare>						*
************************************************************************/
template <class T, class Compare>
class Heap
{
  public:
    Heap(u_int d, Compare compare)	;
    Heap(Array<T>& a, Compare compare)	;

    u_int	nelements()	const	{return _n;}
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
