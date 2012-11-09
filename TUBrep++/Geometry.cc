/*
 *  $Id$
 */
#include "TU/Brep/Brep++.h"

namespace TU
{
namespace Brep
{
/*--------------------- Public member functions -------------------------*/

HalfEdge*
Geometry::mvse()
{
    HalfEdge*	h = parent();
    if (h->geom() == this)
	return h;

    Ptr<HalfEdge>	hn = new HalfEdge, hc = new HalfEdge;

    Geometry*	g;
    for (g = h->geom(); g->cdr() != this; g = g->cdr());
    hn->set_geom(this);
    g->rplacd(0);
    h ->set_geom(h->geom());

    for (g = h->conj()->geom(); g->cdr()->car() != car(); g = g->cdr());
    hc->set_geom(g->cdr());
    g->rplacd(0);
    h ->conj()->set_geom(h->conj()->geom());
    
    hn->set_parent(h->parent())
      ->set_next(h->next())->set_prev(h)->set_conj(h->conj());
    hc->set_parent(h->conj()->parent())
      ->set_next(h->conj()->next())->set_prev(h->conj())->set_conj(h);

    return hn;
}

Geometry*
Geometry::prev() const
{
    HalfEdge*	h = parent();
    
    if (h->geom() == this)
	return (Geometry*)h->prev()->geom()->last();
    Geometry*	g;
    for (g = h->geom(); g->cdr() != this; g = g->cdr());
    return g;
}

const Geometry*
Geometry::sameside(const Point2<int>* p) const
{
    const Geometry	*g = this, *g_max = this;
    int			n, max = -4;

    do
	if ((n = g->point()->angle(p, g->next()->point())) > max)
	{
	    g_max = g;
	    max   = n;
	}
    while ((g = g->conj()) != this);
    return g_max;
}
 
}
}
