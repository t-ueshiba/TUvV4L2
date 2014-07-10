/*
 *  $Id$
 */
#ifndef __TU_COLLECTIONPP_H
#define __TU_COLLECTIONPP_H

#include "TU/Object++.h"

namespace TU
{
/************************************************************************
*  class IDs								*
************************************************************************/
const unsigned	id_List	    = 3;
const unsigned	id_DList    = 4;
const unsigned	id_TreeNode = 5;
const unsigned	id_TreeBase = 6;

/************************************************************************
*  class ObjList							*
************************************************************************/
template <class T>	class ObjList : public Object
{
  public:
    ObjList()	:_p(0), _next(this)	{}

    void	detach()		{_next = _next->_next;}
    void	add(T*)			;
    void	detach(T* p)		{ObjList<T>* node = findnode(p);
					 if (node) node->detach();}
    int		find(T* p)	const	{return findnode(p) != 0;}
    T*		head()		const	{return _next->_p;}

    DECLARE_COPY_AND_RESTORE(ObjList<T>)
    
  private:
    ObjList(T*, ObjList<T>*)		;

    ObjList*	findnode(T*)	const	;

    T*		_p;
    ObjList*	_next;

    DECLARE_DESC
    DECLARE_CONSTRUCTORS(ObjList<T>)

  public:
    class		Iterator;
    friend class	Iterator;
    
    class Iterator
    {
      public:
	Iterator(ObjList<T>* list)	:_dummy(list), _prev(_dummy)	{}
	
	void		head()			{_prev = _dummy;}
	void		add(T* p)		{_prev->add(p);}
	void		detach()		{if (*this) _prev->detach();}
	T&		operator * ()	const	{return *(operator ->());}
	T*		operator ->()	const	{return _prev->_next->_p;}
			operator T*()	const	{return
						     (_prev->_next != _dummy ?
						      _prev->_next->_p : 0);}
	Iterator&	operator ++()		{_prev = _prev->_next;
						 return *this;}
	Iterator	operator ++(int)	{Iterator iter(*this);
						 _prev = _prev->_next;
						 return iter;}

      private:
	ObjList<T>* const	_dummy;
	ObjList<T>*		_prev;
    };
};

template <class T> inline void
ObjList<T>::add(T* p)
{
    _next = new ObjList<T>(p, _next);
}

template <class T> inline
ObjList<T>::ObjList(T* p, ObjList<T>* next)
    :_p(p), _next(next)
{
}

template <class T> const Object::Desc
ObjList<T>::_desc(id_List, 0, ObjList<T>::newObject,
		  &ObjList<T>::_p, &ObjList<T>::_next, MbrpEnd);

/************************************************************************
*  class ObjDList							*
************************************************************************/
template <class T>	class ObjDList : public Object
{
  public:
    ObjDList()	:_p(0), _prev(this),  _next(this)	{}
    
    void	addAtHead(T* p)		{_next->add(p);}
    void	addAtTail(T* p)		{add(p);}
    void	detachFromHead()	{_next->detach();}
    void	detachFromTail()	{_prev->detach();}
    void	detach(T* p)		{ObjDList<T>* node = findnode(p);
					 if (node) node->detach();}
    int		find(T* p)	const	{return findnode(p) != 0;}
    T*		head()		const	{return _next->_p;}
    T*		tail()		const	{return _prev->_p;}

    DECLARE_COPY_AND_RESTORE(ObjDList<T>)
    
  private:
    ObjDList(T*, ObjDList*, ObjDList*)	;
    
    void	add(T*)			;
    void	detach()		{_prev->_next = _next;
					 _next->_prev = _prev;}
    ObjDList*	findnode(T*)	const	;

    T*		_p;
    ObjDList	*_prev, *_next;

    DECLARE_DESC
    DECLARE_CONSTRUCTORS(ObjDList<T>)

  public:
    class		Iterator;
    friend class	Iterator;
    
    class Iterator
    {
      public:
	enum Direction	{Forward, Backward};
    
	Iterator(ObjDList<T>* list, Direction dir=Forward)
	    :_dummy(list), _node(_dummy)	{if (dir == Backward) tail();
						 else		      head();}

	void		head()			{_node = _dummy->_next;}
	void		tail()			{_node = _dummy->_prev;}
	void		add(T* p)		{_node->add(p);}	
	void		detach()		{if (*this) _node->detach();}
	T&		operator * ()	const	{return *(operator ->());}
	T*		operator ->()	const	{return _node->_p;}
			operator T*()	const	{return (_node != _dummy ?
							 _node->_p : 0);}
	Iterator&	operator ++()		{_node = _node->_next;
						 return *this;}
	Iterator&	operator --()		{_node = _node->_prev;
						 return *this;}
	Iterator	operator ++(int)	{Iterator iter(*this);
						 _node = _node->_next;
						 return iter;}
	Iterator	operator --(int)	{Iterator iter(*this);
						 _node = _node->_prev;
						 return iter;}
    
      private:
	ObjDList<T>* const	_dummy;
	ObjDList<T>*		_node;
    };
};

template <class T> inline void
ObjDList<T>::add(T* p)
{
    ObjDList<T>* node = new ObjDList<T>(p, _prev, this);
    node->_prev->_next = node->_next->_prev = node;
}

template <class T> inline
ObjDList<T>::ObjDList(T* p, ObjDList<T>* prev, ObjDList<T>* next)
    :_p(p), _prev(prev), _next(next)
{
}

template <class T> const Object::Desc
ObjDList<T>::_desc(id_DList, 0, ObjDList<T>::newObject,
		   &ObjDList<T>::_p, &ObjDList<T>::_next, &ObjDList<T>::_prev,
		   MbrpEnd);

/************************************************************************
*  class ObjTree							*
************************************************************************/
class ObjTreeBase : public Object
{
  public:
    typedef	int	(*Compare)(const Object*, const Object*);

  private:
    class Node : public Object
    {
      protected:
	void		saveGuts(std::ostream&)		const	;
	void		restoreGuts(std::istream&)		;
    
      private:
	Node(const Object* p=0) :_left(0), _right(0), _d(1), _p(p)	{}
    
	Node*		add(const Object*, Compare, int&)	;
	const Node*	find(const Object*, Compare)	const	;
	Node*		detach(const Object*, Compare, int&)	;
	int		depth()				const	{return this ?
								     _d : 0;}
	Node*		detach_max()				;
	Node*		balance()				;
	Node*		rotate_right()				;
	Node*		rotate_left()				;
	Node*		update_depth()				;

#ifdef TUCollectionPP_DEBUG
	void		check_depth()		const		;
	void		print(std::ostream&)	const		;
	int		real_depth()		const		;
#endif
	
	const Object*	_p;
	Node		*_left, *_right;		// children nodes
	int		_d;				// depth of this tree

	DECLARE_DESC
	DECLARE_CONSTRUCTORS(Node)

	friend class	ObjTreeBase;
    };

  protected:
    ObjTreeBase(Compare comp=0)	:_root(0), _comp(comp)	{}
    
    int		add(const Object* p)		{int err;
						 _root =
						 _root->add(p, _comp, err);
						 return !err;}
    int		find(const Object* p)const	{return
						 (_root->find(p, _comp) != 0);}
    int		detach(const Object* p)		{int err;
						 _root =
						 _root->detach(p, _comp, err);
						 return !err;}
#ifdef TUCollectionPP_DEBUG
  public:
    void	check_depth()		const	{_root->check_depth();}
    void	print(std::ostream& out)const	{_root->print(out);}
#endif

  private:
    Node*	_root;
    Compare	_comp;

    DECLARE_DESC
    DECLARE_CONSTRUCTORS(ObjTreeBase)
};

template <class T>	class ObjTree : public ObjTreeBase
{
  public:
    typedef	int	(*Compare)(const T*, const T*);

    DECLARE_COPY_AND_RESTORE(ObjTree<T>)
    
  public:
    ObjTree(Compare comp=0) :ObjTreeBase((ObjTreeBase::Compare)comp)	{}

    int		add(T* p)		{return ObjTreeBase::add(p);}
    int		find(T* p)	const	{return ObjTreeBase::find(p);}
    int		detach(T* p)		{return ObjTreeBase::detach(p);}
};
 
}
#endif	// !__TU_COLLECTIONPP_H
