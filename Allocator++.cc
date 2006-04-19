/*
 *  $Id: Allocator++.cc,v 1.4 2006-04-19 02:34:37 ueshiba Exp $
 */
#include "TU/Allocator++.h"


#define UintNBits	(8*sizeof(u_int))

namespace TU
{
/************************************************************************
*  class Allocator<T>							*
************************************************************************/
template <class T> void*
Allocator<T>::alloc()
{
    if (_freeList.empty())
    {
	_pageList.push_front(*new Page(_nelmsPerPage));
	_freeList.addPage(_pageList.front());
    }
    return &_freeList.pop_front();
}

template <class T> void
Allocator<T>::free(void* p)
{
    if (p != 0)
    {
	_freeList.push_front(*(Element*)p);
	for (typename List<Enumerator>::Iterator iter = _enumeratorList.begin();
	     iter != _enumeratorList.end(); ++iter)
	    iter->unmark(p);
    }
}

template <class T> void
Allocator<T>::setPageSize(u_int n)
{
    if (n == 0)
	_nelmsPerPage = UintNBits;
    else
	_nelmsPerPage = UintNBits * ((n - 1) / UintNBits + 1);
}

/************************************************************************
*  class Allocator<T>::PageList						*
************************************************************************/
template <class T>
Allocator<T>::PageList::~PageList()
{
    while (!empty())
	delete &pop_front();
}

/************************************************************************
*  class Allocator<T>::FreeList						*
************************************************************************/
template <class T> void
Allocator<T>::FreeList::addPage(Page& page)
{
    for (int i = 0; i < page.nelements(); ++i)
	push_front(*(Element*)page.nth(i));
}

/************************************************************************
*  class Allocator<T>::Enumerator					*
************************************************************************/
template <class T>
Allocator<T>::Enumerator::Enumerator(const Allocator<T>& allocator)
    :_allocator(allocator), _pageIter(_allocator._pageList.begin()),
     _page(0), _index(0), _bit(0), _mark(0)
{
  // Register myself to the allocator.
    ((Allocator<T>&)_allocator)._enumeratorList.push_front(*this);
    
  // Allocate mark bits.
    for (; _pageIter != _allocator._pageList.end(); ++_pageIter)
	++_page;
    _mark.resize(_page);			// # of pages
    _page = 0;
    for (_pageIter  = _allocator._pageList.begin();
	 _pageIter != _allocator._pageList.end(); ++_pageIter)
	_mark[_page++].resize((_pageIter->nelements() - 1) / UintNBits + 1);

  // Set all bits.
    for (_page = 0; _page < _mark.dim(); ++_page)
	for (_index = 0; _index < _mark[_page].dim(); ++_index)
	    _mark[_page][_index] = ~0;

  // Unmark bits corresponding to the elements in _freeList.
    for (typename FreeList::ConstIterator iter = _allocator._freeList.begin();
	 iter != _allocator._freeList.end(); ++iter)
	unmark(iter.operator ->());

    head();
}

template <class T>
Allocator<T>::Enumerator::~Enumerator()
{
    ((Allocator<T>&)_allocator)._enumeratorList.remove(*this);
}

template <class T> void
Allocator<T>::Enumerator::head()
{
    _pageIter = _allocator._pageList.begin();
    _page = 0;
    _index = 0;
    _bit = 0x01;
    ++(*this);
}

template <class T> typename Allocator<T>::Enumerator&
Allocator<T>::Enumerator::operator ++()
{
    do	// for all pages...
    {
	do	// for all indices...
	{
	    if (_mark[_page][_index] != 0)
	    {
		do	// for all bits...
		{
		    if (_mark[_page][_index] & _bit)
		    {
			u_int	r = 0;
			for (u_int bit = _bit; (bit >>= 1) != 0; )
			    ++r;
			_p = (T*)_pageIter->nth(UintNBits * _index + r);
			_bit <<= 1;
			return *this;
		    }
		} while ((_bit <<= 1) != 0x0);
	    }

	    _bit = 0x1;
	} while (++_index < _mark[_page].dim());

	++_pageIter;
	_index = 0;
    } while (++_page < _mark.dim());

    _p = 0;
    return *this;
}

template <class T> void
Allocator<T>::Enumerator::unmark(const void* p)
{
    typename PageList::ConstIterator	iter = _allocator._pageList.begin();
    u_int				page = 0;
    for (; p < iter->nth(0) || p >= iter->nth(iter->nelements()); ++iter)
	++page;

    u_int	offset = ((char*)p - (char*)iter->nth(0)) / sizeof(T),
		index  = offset / UintNBits,
		bit    = 0x1 << (offset % UintNBits);
    _mark[page][index] &= ~bit;
}
 
}
