/*
 *  $Id: List++.h,v 1.2 2002-07-25 02:38:05 ueshiba Exp $
 */
#ifndef __TUListPP_h
#define __TUListPP_h
#include <iterator>

namespace TU
{
/************************************************************************
*  class List<T>							*
************************************************************************/
template <class T>
class List
{
  public:
    typedef T		value_type;
    typedef ptrdiff_t	difference_type;
    typedef T&		reference;
    typedef const T&	const_reference;
    typedef T*		pointer;
    typedef const T*	const_pointer;

    class		Iterator;
    class		ConstIterator;
    
    class Node
    {
      public:
     	Node(const Node&)	:_next(0)	{}
     	Node&		operator =(const Node&)	{return *this;}

      protected:
	Node()		:_next(0)		{}
	~Node()					{_next = 0;}
	
	T*		_next;

      private:
	void		insertNext(T* p)	{p->_next = _next; _next = p;}
	void		eraseNext()		{_next = _next->_next;}

	friend class	List<T>;	// Allow access to private members.
	friend class	Iterator;	// Allow access to _next.
	friend class	ConstIterator;	// ibid.
    };
    
    class Iterator
#if (defined(__GNUC__) && (__GNUC__ < 3))
	: public forward_iterator<value_type, difference_type>
#else
	: public std::iterator<std::forward_iterator_tag, 
			       value_type, difference_type, pointer, reference>
#endif
    {
      public:
        typedef typename List<T>::reference	reference;
	typedef typename List<T>::pointer	pointer;

      public:
	Iterator()	:_list(0), _prev(0)	{}
	
	int		operator ==(const Iterator& i) const
						{return _prev == i._prev;}
	int		operator !=(const Iterator& i) const
						{return !(*this == i);}
	pointer		operator ->()	const	{return (_prev == 0 ?
							 _list->_front :
							 _prev->_next);}
	reference	operator * ()	const	{return *(operator ->());}
	Iterator&	operator ++()		{_prev = operator ->();
						 return *this;}
	Iterator	operator ++(int)	{Iterator tmp = *this;
						 ++*this;
						 return tmp;}
	
      protected:
	Iterator(List<T>* list, T* prev)
	  :_list(list), _prev(prev)		{}

      private:
	List<T>*	_list;
	T*		_prev;

	friend class	List<T>;	// Allow access to _prev.
    };

    class ConstIterator
#if (defined(__GNUC__) && (__GNUC__ < 3))
	: public forward_iterator<value_type, difference_type>
#else
	: public std::iterator<std::forward_iterator_tag,
			       value_type, difference_type, pointer, reference>
#endif
    {
      public:
	typedef typename List<T>::const_reference	const_reference;
	typedef typename List<T>::const_pointer		const_pointer;

      public:
	ConstIterator()	:_current(0)					{}
	ConstIterator(const Iterator& i) :_current(i.operator ->())	{}
	
	int		operator ==(const ConstIterator& i) const
						{return _current==i._current;}
	int		operator !=(const ConstIterator& i) const
						{return !(*this == i);}
	const_pointer	operator ->()	const	{return _current;}
	const_reference	operator * ()	const	{return *(operator ->());}
	ConstIterator&	operator ++()		{_current = _current->_next; 
						 return *this;}
	ConstIterator	operator ++(int)	{ConstIterator tmp = *this;
						 ++*this;
						 return tmp;}
	
      protected:
	ConstIterator(const T* current)
	  :_current(current)			{}
	
      private:
	const T*	_current;

	friend class	List<T>;	// Allow access to the protected ctor.
    };

  public:
    List()	:_front(0), _back(0)		{}
    ~List()					{}

    Iterator		begin()			{return Iterator(this, 0);}
    Iterator		end()			{return Iterator(this, _back);}
    ConstIterator	begin()		const	{return _front;}
    ConstIterator	end()		const	{return 0;}
    int			empty()		const	{return _front == 0;}
    reference		front()			{return *begin();}
    const_reference	front()		const	{return *begin();}
    void		push_front(T& x)	{insert(begin(), x);}
    reference		pop_front()		{return erase(begin());}
    Iterator		insert(Iterator i, T& x);
    reference		erase(Iterator i)	;
    void		remove(const T& x)	;
    
  private:
    List(const List&)				;
    List&		operator =(const List&)	;
    
    T			*_front, *_back;

    friend class	Iterator;	// Allow access to _front.
};
 
}

#endif	// !__TUListPP_h
