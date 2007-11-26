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
 *  $Id: Allocator++.h,v 1.6 2007-11-26 07:55:48 ueshiba Exp $
 */
#ifndef __TUAllocatorPP_h
#define __TUAllocatorPP_h

#include "TU/List++.h"
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class Allocator<T>							*
************************************************************************/
template <class T>
class Allocator
{
  public:
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
 
}

#endif	// !__TUAllocatorPP_h
