/*
 *  $Id$
 */
#ifndef TU_OBJECTPP_H
#define TU_OBJECTPP_H

#include <sys/types.h>
#include <iostream>
#include <map>

namespace TU
{
/************************************************************************
*  class PtrBase:	 abstract pointer class for the object to be	*
*			 protected from GC				*
*  class Ptr<T>:	 pointer class of "T" derived from "Object"	*
************************************************************************/
class				Object;
typedef Object* Object::*	Mbrp;
Mbrp const			MbrpEnd = 0;
    
class PtrBase
{
  public:
    PtrBase(const PtrBase& q):_p(q._p), _nxt(_root)	{_root = this;}
    PtrBase&	operator =(const PtrBase& q)		{_p=q._p;return *this;}
    Object*&	operator ->*(Mbrp q)		const	{return _p->*q;}
//		operator bool()			const	{return _p != 0;}
//  bool	operator !()			const	{return _p == 0;}
    
  protected:
    PtrBase(Object* obj)	 :_p(obj),  _nxt(_root)	{_root = this;}
    ~PtrBase()						{
							  if (_root == this)
							    _root = _nxt;
							  else
							    _root->_nxt = _nxt;
							}
    void		operator delete(void*)		{}

    Object*		_p;

  private:
    void*		operator new(size_t)	; // prohibit heap allocation

    static void		mark()			;
			
    PtrBase*		_nxt;
    static PtrBase*	_root;

    friend class	Object;			// allow access to mark()
};

template <class T>	class Ptr : public PtrBase
{
  public:
		Ptr(T* obj=0)	:PtrBase(obj)		{}

		operator T*()			const	{return (T*)_p;}
    Ptr&	operator = (T* obj)			{_p=obj; return *this;}
    T*		operator ->()			const	{return (T*)_p;}
    bool	operator ==(T* obj)		const	{return _p == obj;}
    bool	operator !=(T* obj)		const	{return _p != obj;}
};

/************************************************************************
*  Class Object:	base class of all object			*
************************************************************************/
class ObjectHeader
{
  protected:
    ObjectHeader()	   :_gc(0), _sv(0), _cp(0), _fr(0)		{}
    ObjectHeader(u_int nb) :_gc(0), _sv(0), _cp(0), _fr(0), _nb(nb)	{}
    ObjectHeader(const ObjectHeader&)
			   :_gc(0), _sv(0), _cp(0), _fr(0)		{}
    ObjectHeader&	operator =(const ObjectHeader&)	{return *this;}
    virtual		~ObjectHeader()			{}

    unsigned	_gc	: 1;	// Object is alive. Don't sweep it !
    unsigned	_sv	: 1;	// Already saved in stream
    unsigned	_cp	: 1;	// Already deeply copied
    unsigned	_fr	: 1;	// In free list of PAGE::CELL
    unsigned	_nb	: 28;	// Object size in # of Page::Blocks
};

class Object : private ObjectHeader
{
  protected:
    class Desc
    {
      private:
	typedef std::map<u_short, Desc*>	Map;
	typedef	Object*				(*Pftype)();

      public:
	Desc(u_short, u_short, Pftype, ...)	;
	~Desc()					;
	u_short		id()		const	{return _id;}
	const Mbrp*	mbrp()		const	{return _p;}
	static Object*	newObject(u_short id)	{return (*_map)[id]->_pf();}

      private:    
	u_int		nMbrp()		const	;
	bool		setMbrp()		;
    
	static u_int	_ndescs;		// # of descs
	static Map*	_map;			// id -> desc looking-up
    
	const u_short	_id;			// class ID of mine
	const u_short	_bid;			// class ID of base
	const Pftype	_pf;			// constructor
	Mbrp*		_p;			// pointer members
	bool		_init;
    };

  public:
    void*		operator new(size_t)	;
    void		operator delete(void*)	{}

    bool		null()		const	{return (this == 0);}
    bool		consp()		const	{return !null() && iscons();}
    std::ostream&	save(std::ostream&)	const	;

  protected:
    virtual bool	iscons()		const	{return false;}
    virtual void	saveGuts(std::ostream&)	const	{}
    virtual void	restoreGuts(std::istream&)	{}
    Object*		copyObject(u_int)	const	;
    static Object*	restoreObject(std::istream&)	;
    
  private:
    void		mark()		const	;
    virtual const Desc&	desc()		const	= 0;
    virtual Object*	clone()		const	= 0;

    friend void		PtrBase::mark();	// allow access to mark()
    friend class	SaveMap;		// allow access to header
    friend class	CopyMap;		// allow access to header
};

#define DECLARE_COPY_AND_RESTORE(TYPE)					   \
    Ptr<TYPE >		copy()	const	{				   \
					    Object* obj = copyObject(0);   \
					    return Ptr<TYPE >((TYPE*)obj); \
					}				   \
    static Ptr<TYPE >	restore(std::istream& in)			   \
					{				   \
					    Object* obj=restoreObject(in); \
					    return Ptr<TYPE >((TYPE*)obj); \
					}

#define DECLARE_DESC							   \
    static const Desc	_desc;						   \
    const Desc&		desc()		const	{return _desc;}

#define DECLARE_CONSTRUCTORS(TYPE)					   \
    Object*		clone()		const	{return new TYPE(*this);}  \
    static Object*	newObject()		{return new TYPE;}

/************************************************************************
*  class Cons:	cons cell						*
************************************************************************/
template <class T>
class Cons : public Object
{
  public:
    static Ptr<Cons>	cons0(T* ca)		{return new Cons(ca, 0);}
    Ptr<Cons>		cons(T* ca)		{return new Cons(ca, this);}
    T*			car()		const	{return (!null() ? _ca : 0);}
    Cons*		cdr()		const	{return (!null() ? _cd : 0);}
    const Cons*		nthcdr(int)	const	;
    Cons*		nthcdr(int)		;
    T*			nth(int n)	const	{return nthcdr(n)->car();}
    const Cons*		last()		const	;
    Cons*		last()			;
    int			length()	const	;
    Ptr<Cons>		reverse()	const	;
    Ptr<Cons>		append(Cons*)	const	;
    const Cons*		member(const T*)const	;
    Cons*		member(const T*)	;
    Ptr<Cons>		remove(const T*)const	;
    Cons*		rplaca(T* ca)		{_ca = ca; return this;}
    Cons*		rplacd(Cons* cd)	{_cd = cd; return this;}
    Ptr<Cons>		nreverse()		;
    Cons*		nconc(Cons*)		;
    Cons*		detach(const T*)	;

    DECLARE_COPY_AND_RESTORE(Cons<T>)

  protected:
    Cons(T* ca=0, Cons* cd=0)	:_ca(ca), _cd(cd)	{}

    virtual bool	iscons()			const	{return true;}

  private:
    T*		_ca;
    Cons*	_cd;

    DECLARE_DESC
    DECLARE_CONSTRUCTORS(Cons<T>)
};

/************************************************************************
*  some implementations							*
************************************************************************/
std::ostream&	eoc(std::ostream&);			// End of context
 
}
#endif	// !TU_OBJECTPP_H
