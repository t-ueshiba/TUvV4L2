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
 *  $Id: Allocator.h,v 1.2 2008-09-10 05:10:29 ueshiba Exp $
 */
#ifndef __TUAllocator_h
#define __TUAllocator_h

#include "TU/Array++.h"
#include "TU/List.h"

namespace TU
{
/************************************************************************
*  class Allocator<T>							*
************************************************************************/
template <class T>
class Allocator
{
  public:
    enum	{UintNBits = 8*sizeof(u_int)};
    
    class Page : public List<Page>::Node
    {
      public:
	Page(u_int nelements)
	    :_nelements(nelements),
	     _p(new char[sizeof(T)*_nelements])	{}
	~Page()					{delete [] _p;}

	u_int		nelements()	const	{return _nelements;}
	void*		nth(int i)	const	{return _p + sizeof(T)*i;}

      private:
	const u_int	_nelements;	// # of elements in this page.
	char* const	_p;		// memory pool.
    };

    class PageList : public List<Page>
    {
      public:
      	~PageList()					;

	using		List<Page>::begin;
	using		List<Page>::end;
	using		List<Page>::empty;
	using		List<Page>::front;
	using		List<Page>::pop_front;
    };

    class Element : public List<Element>::Node
    {
    };

    class FreeList : public List<Element>
    {
      public:
	void		addPage(Page& page)		;

	using		List<Element>::begin;
	using		List<Element>::end;
	using		List<Element>::empty;
	using		List<Element>::front;
	using		List<Element>::pop_front;
    };

    class Enumerator : public List<Enumerator>::Node
    {
      public:
	Enumerator(const Allocator<T>& allocator)	;
	~Enumerator()					;

	void		head()				;
			operator T*()			{return _p;}
	T*		operator ->()			{return _p;}
	Enumerator&	operator ++()			;
	
      private:
	friend class	Allocator;
	
	void		unmark(const void* p)		;
	
	const Allocator<T>&			_allocator;
	typename List<Page>::ConstIterator	_pageIter;
	u_int					_page, _index, _bit;
	T*					_p;
	Array<Array<u_int> >	_mark;	// marked if not in freeList.
    };

  public:
    Allocator(u_int nelmsPerPage=1024)		{setPageSize(nelmsPerPage);}

    void*		alloc()			;
    void		free(void* p)		;
    void		setPageSize(u_int n)	;

  private:
    friend class	Enumerator;

    PageList		_pageList;
    FreeList		_freeList;
    u_int		_nelmsPerPage;
    List<Enumerator>	_enumeratorList;
};
 
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

/*
 *  class Allocator<T>::PageList
 */
template <class T>
Allocator<T>::PageList::~PageList()
{
    while (!empty())
	delete &pop_front();
}

/*
 *  class Allocator<T>::FreeList
 */
template <class T> void
Allocator<T>::FreeList::addPage(Page& page)
{
    for (int i = 0; i < page.nelements(); ++i)
	push_front(*(Element*)page.nth(i));
}

/*
 *  class Allocator<T>::Enumerator
 */
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

#endif	// !__TUAllocator_h
