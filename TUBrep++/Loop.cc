/*
 *  $Id: Loop.cc,v 1.1.1.1 2002-07-25 02:14:15 ueshiba Exp $
 */
#include <stdarg.h>
#include <limits.h>
#include "TU/Brep/Brep++.h"

namespace TU
{
namespace Brep
{
/*-------------------- Public member functions --------------------*/

Ptr<Cons<PointB> >
Face::make_path(int u, int v, PFUNC next_candidate, ...) const
{
    Ptr<PointB>		p0 = new PointB(u, v);
    Neighbor		nbr0(p0, this);
    int			nflags = nbr0.check();
    if (nflags == -1)				// adjacent neighbors ?
	return 0;
    
    Ptr<Cons<PointB> >	path = Cons<PointB>::cons0(p0);
    va_list		args;
    va_start(args, next_candidate);

    for (int backward = 0; backward < 2 ; backward++)
    {
	for (Ptr<PointB> p = p0; ; )
	{
	    if (!(p = next_candidate(p, backward, args)))
		break;
	    
	    Neighbor	nbr(p, this);

	    if (p0 == path->car() && nbr0.nelm() &&
		(p->adj(nbr0) != -1 || nbr.check() == -1))
		break;
	    
	    int pflags = p->check(path);
	    if (pflags == -1)			// p: equal to some on "path" ?
		break;
	    else if (pflags & PointB::SECOND)	// p: adj. to 2nd of "path" ?
		path = path->rplaca(p);
	    else
		path = path->cons(p);

	    if (nbr.nelm() || pflags & PointB::CROSS)
		break;
	}
	path = path->nreverse();
    }
    
    va_end(args);
    if (path->cdr()->null() && nflags == -2)
	return 0;
    else
	return path;
}

HalfEdge*
Face::make_edge(Cons<PointB>* path)
{
    if (path->null())			// "path" contains no points?
	return 0;
    
    Ptr<PointB>	p = path->car();
    path = path->cdr();

    if(path->null())			// "path" contains only one point?
	return mvr((*p)[0], (*p)[1])->connect();

    Ptr<Cons<PointB> >	g = path->nreverse();
    Ptr<PointB>		q = g->car();
    g = g->cdr();
    Neighbor		nbr(p, this);
    int	i = q->adj(nbr);
    if (i != -1 && nbr[i].head() != nbr[i].tail())
    {
	PointB* tmp = p;
	p = q;
	q = tmp;
    }
    else
	g = g->nreverse();

    HalfEdge* h = mvr((*p)[0], (*p)[1])->me(mvr((*q)[0], (*q)[1]), g)
				       ->connect();
    return (h->point() == q ? h->connect() : h->conj()->prev()->connect());
}

HalfEdge*
Face::mvr(int u, int v)
{
    Ptr<Ring>		r = new Ring;
    Ptr<HalfEdge>	h = new HalfEdge;
    Ptr<PointB>		p = new PointB(u, v);
    h->set_parent(r->set_parent(this))->set_geom(Cons<PointB>::cons0(p));
    return h;
}

Face*
Loop::p2f(int u, int v)
{
    Point2<int>	p(u, v);
    Loop*	ll;
    for (Loop* l = this; !(ll = l->p2l(&p)); l = l->parent());
    return (ll->isFace() ? ll->l2f() : 0);
}

Geometry*
Loop::p2g(int u, int v)
{
    Point2<int>	p(u, v);
    Loop*	l;
    for (l = this; !l->involve(&p); l = l->parent());
    return l->p2g(&p);
}


/*-------------------- Private member functions --------------------*/

Ptr<Loop>
Loop::detach()
{
    if (_parent)
    {
	Loop	*child, *prev = 0;
	for (ChildIterator iter(_parent); child = iter; ++iter)
	    if (child == this)
	    {
		if (prev)
		    prev->_brother = child->_brother;
		else
		    _parent->_children = child->_brother;
		break;
	    }
	    else
		prev = child;
	_parent = 0;
    }
    return this;
}

Loop*
Loop::set_parent(Loop* l)
{
    _parent  = l;
    _brother = l->_children;
    return l->_children = this;
}

Loop*
Loop::put_children(Loop* l)
{
    for (ChildIterator iter(this); iter; ++iter)
	iter->detach()->set_parent(l);
    return this;
}

Loop*
Loop::get_children(Loop* l)
{
    for (ChildIterator iter(l); iter; ++iter)
	if (involve(iter->head()->geom()->point()))
	    iter->detach()->set_parent(this);
    return this;
}

Loop*
Loop::p2l(const Point2<int>* p)
{
    if (involve(p))
    {
	for (ChildIterator iter(this); iter; ++iter)
	{
	    Loop*	l = iter->p2l(p);
	    if (l)
		return l;
	}
	return this;
    }
    return 0;
}

bool
Loop::involve(register const Point2<int>* p) const
{
    if (head() == 0)
	return true;

    register const int		v = (*p)[1];
    register int		u_nearest = INT_MIN, u_sameside = INT_MIN;
    register Geometry*	g = head()->geom();
    do
    {
	register PointB*	q = g->point();
	if ((*q)[1] == v && u_nearest <= (*q)[0] && (*q)[0] <= (*p)[0])
	{
	    if ((*q)[0] == (*p)[0])
		return !isFace();

	    u_nearest = (*q)[0];
	    if (g->sameside(p) == g)
		u_sameside = (*q)[0];
	}
    } while ((g = g->next()) != head()->geom());
    
    if (u_nearest == INT_MIN)
	return false;
    return (isFace() ? u_nearest == u_sameside : u_nearest != u_sameside);
}

Geometry*
Loop::p2g(const Point2<int>* p)
{
    if (head() != 0)
    {
	register Geometry*	g = head()->geom();
	do
	{
	    if (*g->point() == *p)
		return g;
	} while ((g = g->next()) != head()->geom());
    }

    for (ChildIterator iter(this); iter; ++iter)
    {
	Geometry*	g = iter->p2g(p);
	if (g != 0)
	    return g;
    }
    return 0;
}
 
}
}
