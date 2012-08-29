/*
 *  $Id: Brep++.h,v 1.3 2012-08-29 21:16:44 ueshiba Exp $
 */
#ifndef __TUBrepPP_h
#define __TUBrepPP_h

#include "TU/Object++.h"
#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  ClassID for object types						*
************************************************************************/
const unsigned	id_Loop     = 256;
const unsigned	id_Face     = id_Loop	  + 1;
const unsigned	id_Ring	    = id_Face	  + 1;
const unsigned	id_Root	    = id_Ring	  + 1;
const unsigned	id_HalfEdge = id_Root	  + 1;
const unsigned	id_Geometry = id_HalfEdge + 1;
const unsigned	id_Point    = id_Geometry + 1;

namespace Brep
{
/************************************************************************
*  defined classes							*
************************************************************************/
class	Loop;
class	Face;
class	Ring;
class	Root;
class	HalfEdge;
class	Geometry;
class	PointB;
class	Neighbor;

typedef	 PointB* (*PFUNC)(const PointB*, int, ...);

/************************************************************************
*  class Loop, Face, Ring: loop						*
************************************************************************/
class Loop : public Object
{
  public:
    class ChildIterator
    {
      public:
	ChildIterator(const Loop* l)	:_l(l->_children)	{}

			operator Loop*()	{return _l;}
	Loop*		operator ->()		{return operator Loop*();}
	ChildIterator&	operator ++()		{_l = _l->_brother;
						 return *this;}

      private:
	Loop*		_l;
    };

  public:
    virtual bool	isFace()	const	= 0;
    Loop*		parent()	const	{return _parent;}
    virtual Root*	root()			{return parent()->root();}
    HalfEdge*		head()		const	{return _head;}
    virtual Face*	l2f()			= 0;
    Face*		p2f(int, int)		;
    Geometry*		p2g(int, int)		;
    Object*		prop()		const	{return _prop;}
    
  protected:
    Loop()		:_parent(0), _brother(0), _children(0),
			 _head(0), _prop(0)
#ifdef TUBrepPP_DEBUG
			, n(nLoops++)
#endif
								{}
    ~Loop()		{_parent=_brother=_children=0; _head=0; _prop=0;}
    
  private:
    Loop*		set_parent(Loop*)			;
    Loop*		put_children(Loop*)			;
    Loop*		get_children(Loop*)			;
    Ptr<Loop>		detach()				;
    Loop*		p2l(const Point2<int>*)			;
    Geometry*		p2g(const Point2<int>*)			;
    bool		involve(const Point2<int>*)	const	;
    
    Loop*		_parent;		// parent Ring or Face
    Loop*		_brother;		// brother Faces or Rings
    Loop*		_children;		// child Rings or Faces
    HalfEdge*		_head;			// representative HALF EDGE
    Object*		_prop;			// property

    DECLARE_DESC

    friend class	ChildIterator;
    friend class	Face;
    friend class	HalfEdge;

#ifdef TUBrepPP_DEBUG
  public:
    const u_int		n;
    static u_int	nLoops;
#endif
};

class Face : public Loop
{
  public:
    bool		isFace()			const	{return true;}
    Face*		l2f()					{return this;}
    HalfEdge*		mvr(int, int)				;
    Ptr<Cons<PointB> >	make_path(int, int, PFUNC, ...)	const	;
    HalfEdge*		make_edge(Cons<PointB>*)		;

  private:
    DECLARE_DESC
    DECLARE_CONSTRUCTORS(Face)
};

class Ring : public Loop
{
  public:
    bool		isFace()	const	{return false;}
    Face*		l2f()			{return parent()->l2f();}

  private:
    DECLARE_DESC
    DECLARE_CONSTRUCTORS(Ring)
};

class Root : public Face
{
  public:
    Root*	root()				{return this;}

    DECLARE_COPY_AND_RESTORE(Root)
    
  private:
    DECLARE_DESC
    DECLARE_CONSTRUCTORS(Root)
};

/************************************************************************
*  class Geometry:	half point					*
************************************************************************/
class Geometry : public Cons<PointB>
{
  public:
    HalfEdge*	mvse()				;
    PointB*	point()			const	{return car();}
    HalfEdge*	parent()		const	{return (HalfEdge*)last()
						     ->cdr();}
    Root*	root()			const	;
    Geometry*	prev()			const	;
    Geometry*	cdr()			const	{return (Geometry*)
						     Cons<PointB>::cdr();}
    Geometry*	next()			const	;
    Geometry*	conj()			const	;
    int		operator [](u_int i)	const	;
    int		dir()			const	;
    int		angle()			const	;
    int		juncnum()		const	;
    const Geometry*
		sameside(const Point2<int>*) const	;
};

/************************************************************************
*  class HalfEdge:	half edge					*
************************************************************************/
class HalfEdge : public Object
{
  private:
    enum	LoopDir				{FACE, RING};

  public:
    Loop*	parent()		const	{return _parent;}
    Root*	root()			const	{return parent()->root();}
    HalfEdge*	prev()			const	{return _prev;}
    HalfEdge*	next()			const	{return _next;}
    HalfEdge*	conj()			const	{return _conj;}
    Geometry*	geom()			const	{return _geo;}
    HalfEdge*	kvr()				;
    HalfEdge*	kvje()				;
    HalfEdge*	me(HalfEdge*, Cons<PointB>* = 0);
    HalfEdge*	ke()				;
    HalfEdge*	kill_edge()			;
    PointB*	point()			const	;
    int		operator [](u_int i)	const	;
    int		dir()			const	;
    int		angle()			const	;
    int		juncnum()		const	;
    Object*	prop()			const	{return _prop;}
    
  protected:
    HalfEdge()
	:_parent(0), _prev(this), _next(this), _conj(this), _geo(0), _prop(0)
#ifdef TUBrepPP_DEBUG
	, n(nHalfEdges++)
#endif
			{}
    ~HalfEdge()		{_parent=0; _prev=_next=_conj = 0; _geo=0; _prop=0;}

  private:
    HalfEdge*	connect()				;
    HalfEdge*	reshape()				;
    HalfEdge*	mesf(HalfEdge*, Cons<PointB>*)		;
    HalfEdge*	mef(HalfEdge*, Cons<PointB>*)		;
    HalfEdge*	mejr(HalfEdge*, Cons<PointB>*)		;
    HalfEdge*	mekr(HalfEdge*, Cons<PointB>*)		;
    HalfEdge*	kejf()					;
    HalfEdge*	kef()					;
    HalfEdge*	kesr()					;
    HalfEdge*	kemr()					;
    HalfEdge*	join(HalfEdge*, Cons<PointB>*)		;
    HalfEdge*	separate()				;
    HalfEdge*	set_prev(HalfEdge* h)	{_prev=h; h->_next=this; return this;}
    HalfEdge*	set_next(HalfEdge* h)	{h->set_prev(this); return this;}
    HalfEdge*	set_conj(HalfEdge* h)	{_conj=h; h->_conj=this; return this;}
    HalfEdge*	set_parent(Loop*)			;
    HalfEdge*	set_geom(Cons<PointB>* g)		;
    LoopDir	loop_dir()			const	;
    
    Loop*	_parent;		// parent Face or Ring
    HalfEdge	*_prev, *_next, *_conj;	// winged edge
    Geometry*	_geo;			// HALF POINT list
    Object*	_prop;			// property

    DECLARE_DESC
    DECLARE_CONSTRUCTORS(HalfEdge)

    friend HalfEdge*	Face::make_edge(Cons<PointB>*);
    friend HalfEdge*	Face::mvr(int, int);
    friend HalfEdge*	Geometry::mvse();

#ifdef TUBrepPP_DEBUG
  public:
    const u_int		n;
    static u_int	nHalfEdges;
#endif
};

/************************************************************************
*  Class PointB:	2D coordinate of a point			*
************************************************************************/
class PointB : public Object, public Point2<int>
{
  public:
    PointB(int u=0, int v=0)	:Point2<int>(u, v)		{}

    int		adj(const Point2<int>*)			const	;
    int		adj(const Neighbor&)			const	;
    int		dir(const Point2<int>*)			const	;
    int		angle(const Point2<int>*, Point2<int>*)	const	;

  protected:
    void	saveGuts(std::ostream&)			const	;
    void	restoreGuts(std::istream&)			;

  private:
    enum	{SECOND = 0x1, CROSS = 0x2};

    int		check(const Cons<PointB>*)		const	;
    
    DECLARE_DESC
    DECLARE_CONSTRUCTORS(PointB)

    friend Ptr<Cons<PointB> >
		Face::make_path(int, int, PFUNC ...)	const	;
};

/************************************************************************
*  Class Neighbor: neighbor						*
************************************************************************/
class Neighbor
{
  public:
    class Element
    {
      public:
	Geometry*	head()			const	{return _gs;}
	Geometry*	tail()			const	{return _ge;}
    
      private:
	void		set(Geometry* s, Geometry* e)	{_gs=s; _ge=e;}

	Geometry	*_gs, *_ge;
    
	friend class	Neighbor;
    };

  public:
    Neighbor(const PointB*, const Face*)		;
    Neighbor(const Geometry*)				;

    const Element&	operator [](int i)	const	{return _elm[i];}
    Element&		operator [](int i)		{return _elm[i];}
    int			nelm()			const	{return _n;}
    int			check()			const	;

  private:
    void		inFace(const Face*)		;
    void		onLoop(const Loop*)		;
    void		sort()				;

    const PointB* const	_p;			// central point
    const PointB* const	_pn;			// next point
    int			_n;
    Element		_elm[8];
};

/************************************************************************
*  Several implementations						*
************************************************************************/
inline int
PointB::adj(const Point2<int>* p) const
{
    return Point2<int>::adj(*p);
}

inline int
PointB::dir(const Point2<int>* p) const
{
    return Point2<int>::dir(*p);
}

inline int
PointB::angle(const Point2<int>* p, Point2<int>* q) const
{
    return Point2<int>::angle(*p, *q);
}

inline Root*
Geometry::root() const
{
    return parent()->root();
}

inline Geometry*
Geometry::next() const
{
    Geometry* g = cdr();
    return (g->consp() ? g : ((HalfEdge*)g)->next()->geom());
}

inline Geometry*
Geometry::conj() const
{
    HalfEdge* h = parent();
    return (Geometry*)(h->geom() == this ? h->conj()->next()->geom() :
			h->conj()->geom()->member(point()));
}

inline int
Geometry::operator [](u_int i) const
{
    return (*point())[i];
}

inline int
Geometry::dir() const
{
    return point()->dir(next()->point());
}

inline int
Geometry::angle() const
{
    return point()->angle(conj()->next()->point(), next()->point());
}

inline int
Geometry::juncnum() const
{
    HalfEdge* h = parent();
    return (h->geom() != this ? 2 : h->juncnum());
}

inline PointB*
HalfEdge::point() const
{
    return geom()->point();
}

inline int
HalfEdge::operator [](u_int i) const
{
    return (*point())[i];
}

inline int
HalfEdge::dir() const
{
    return geom()->dir();
}

inline int
HalfEdge::angle() const
{
    return point()->angle(prev()->conj()->geom()->next()->point(),
			  geom()->next()->point());
}

inline HalfEdge*
HalfEdge::set_geom(Cons<PointB>* g)
{
    _geo = (Geometry*)g->nconc((Cons<PointB>*)this);
    return this;
}

inline
Neighbor::Neighbor(const PointB* p, const Face* f)
    :_p(p), _pn(p), _n(0)
{
    inFace(f);
}

inline
Neighbor::Neighbor(const Geometry* g)
    :_p(g->point()), _pn(g->next()->point()), _n(0)
{
    inFace(g->parent()->parent()->l2f());
}
 
}
}
#endif	/* !__TUBrepPP_h */
