/*
 *  $Id$
 */
#include "TU/Brep/Brep++.h"
#include <stdexcept>

namespace TU
{
namespace Brep
{
/*--------------------- Pulic member functions -------------------------*/

HalfEdge*
HalfEdge::kill_edge()
{
    if (!kvr())
	return 0;

    if (prev() == this || conj()->prev() == conj())
	return ke()->reshape();
    else
    {
	HalfEdge* hn = next();
	HalfEdge* hcn = ke();
	hn->reshape();
	return hcn->reshape();
    }
}

HalfEdge*
HalfEdge::kvr()
{
    if (juncnum() != 0)
	return this;
    parent()->detach();
    delete parent();
    delete this;
    return 0;
}

HalfEdge*
HalfEdge::kvje()
{
    if (juncnum() == 2 && prev() != this)
    {
	HalfEdge*	hcn = conj()->next();
	conj()->set_geom(conj()->geom()->nconc(conj()->next()->geom()));
	prev()->set_geom(prev()->geom()->nconc(geom()));
	conj()->set_next(conj()->next()->next())->set_parent(conj()->parent());
	HalfEdge*	hp = prev()->set_next(next())->set_conj(conj())
				   ->set_parent(parent());
	delete hcn;
	delete this;
	return hp;
    }
    else
	return prev();
}

HalfEdge*
HalfEdge::me(HalfEdge* hn, Cons<PointB>* g)
{
    HalfEdge*	h;
    
    if (parent() == hn->parent())
	if (parent()->isFace())
	    h = mesf(hn, g);
	else
	    h = mef(hn, g);
    else if (parent()->parent() == hn->parent()->parent() &&
	     !parent()->isFace())
	h = mejr(hn, g);
    else if ((parent()->parent() == hn->parent() && !parent()->isFace()) ||
	     (hn->parent()->parent() == parent() && !hn->parent()->isFace()))
	h = mekr(hn, g);
    else
	throw std::invalid_argument("TU::Brep::HalfEdge::me\tInconsistent half edges!");

    return h;
}

HalfEdge*
HalfEdge::ke()
{
    HalfEdge	*hcn = conj()->next(), *hn = next(), *hc = conj(), *h;
    
    if (parent() == conj()->parent())
	if (parent()->isFace())
	    h = kemr();
	else
	    h = kesr();
    else if (parent()->isFace() != conj()->parent()->isFace())
	h = kef();
    else if (parent()->isFace())
	h = kejf();
    else
	throw std::invalid_argument("TU::Brep::HalfEdge::ke\tInconsistent half edge. Cannot kill it!");
    
    if (this != h)
	delete this;
    if (hc != hn && hc != h)
	delete hc;
    
    return h;
}

int
HalfEdge::juncnum() const
{
    if (conj() == this)
	return 0;
    int	  jn = 0;
    const HalfEdge* h = this;
    do
    {
	++jn;
    } while ((h = h->conj()->next()) != this);
    return jn;
}


/*--------------------- Private member functions -------------------------*/

HalfEdge*
HalfEdge::connect()
{
    Neighbor	nbr(geom());
    HalfEdge*	hcn = this;
    
    for (int i = 0; i < nbr.nelm(); i++)
    {
	Geometry	*gs, *ge;
	
	for (gs = nbr[i].head(), ge = nbr[i].tail(); gs != ge; )
	{
	    Geometry*	g;
	    
	    for (g = gs->next(); g != ge; g = g->next())
		if (g->juncnum() != 2)
		    break;
	    
	    if (g == gs->next())
		if (gs->juncnum() == 1)
		    (gs = g)->mvse()->prev()->ke()->kvr();
		else if (g->juncnum() == 1)
		    (gs = ge = g->next())->mvse()->prev()->ke()->kvr();
		else if (point()->dir(gs->point()) % 2)	// gs: 8-nbr
		    gs = g;
		else if (point()->dir(g->point()) % 2)	// g:  8-nbr
		    if (g == ge)
			ge = gs;
		    else if (g->next() == ge)
			gs = ge = g;
		    else
		    {
			hcn = hcn->me(gs->mvse());
			HalfEdge	*h, *he;
			for (h = g->mvse(), he = ge->mvse(); h->next() != he; )
			    h->next()->kvje();
			h->ke();
			gs = ge;
		    }
		else						// gs, g: 4-nbr
		{
		    g->mvse();
		    hcn = hcn->me(gs->mvse())->next()->ke()->kvje();
		    gs = g;
		}
	    else	// g: not adjacent to gs
	    {
		HalfEdge	*hs, *h;

		for (hs = gs->mvse(), h = g->mvse(); hs->next() != h; )
		    hs->next()->kvje();
		hcn = hcn->me(hs)->next()->ke()->kvje();
		gs = g;
	    }
	}
	
	hcn = hcn->me(gs->mvse())->next()->kvje();
    }

    return hcn->kvje();
}

HalfEdge*
HalfEdge::reshape()
{
    HalfEdge* h = this;
    do
	switch (h->angle())
	{
	  case -4:			// -180 deg
	  {
	    for (Geometry *g  = h->geom()->next()->next(),
			   *ge = h->geom()->prev();
		 g != ge; g = g->next())
		if (g->point()->adj(h->point()) == 1)
		{
		    (h = h->next())->prev()->kvr();
		    return h->reshape();
		}
	  }
	    break;
	  case -2:			// -90 deg
	    if (!(h->dir() % 2))
	    {
		(h = h->conj()->next()->geom()->next()->mvse())->prev()->ke();
		return h->connect();
	    }
	    break;
	  case 3:			// 45 deg
	  {
	    int	  d = h->dir() % 2;
	    if (d && h->conj()->next()->dir() != 3)
		(h = h->geom()->next()->mvse())->prev()->ke();
	    else if (!d && h->prev()->conj()->dir() != 3)
		(h = h->prev()->conj()->geom()->next()->mvse())->prev()->ke();
	    else
		break;
	  }
	    return h->connect();
	}
    while ((h = h->conj()->next()) != this);

    return h;
}

HalfEdge*
HalfEdge::mesf(HalfEdge* hn, Cons<PointB>* g)
{
    Loop*	f_old = parent();
    Ptr<Face>	f_new = new Face;
    HalfEdge*	h = join(hn, g)->set_parent(f_new);

    h->conj()->set_parent(f_old);
    f_new->set_parent(f_old->parent())->get_children(f_old);
    return h;
}

HalfEdge*
HalfEdge::mef(HalfEdge* hn, Cons<PointB>* g)
{
    Loop*	r = parent();
    Ptr<Face>	f = new Face;
    HalfEdge*	h = join(hn, g);
    
    if (h->loop_dir() == FACE)
	h->set_parent(f)->conj()->set_parent(r);
    else
	h->set_parent(r)->conj()->set_parent(f);
    f->set_parent(r)->get_children(r->parent());
    return h;
}

HalfEdge*
HalfEdge::mejr(HalfEdge* hn, Cons<PointB>* g)
{
    Loop*	r = parent();
    HalfEdge*	h = join(hn, g)->set_parent(hn->parent());
    
    r->put_children(h->parent())->detach();
    delete r;
    return h;
}

HalfEdge*
HalfEdge::mekr(HalfEdge* hn, Cons<PointB>* g)
{
    Loop	*f, *r;
    
    if (parent()->isFace())
    {
	f = parent();
	r = hn->parent();
    }
    else
    {
	f = hn->parent();
	r = parent();
    }
    HalfEdge* h = join(hn, g)->set_parent(f);
    r->put_children(f->parent())->detach();
    delete r;
    return h;
}

HalfEdge*
HalfEdge::kejf()
{
    Loop*	f = parent();
    HalfEdge*	h = separate()->set_parent(conj()->parent());
    
    f->put_children(h->parent())->detach();
    delete f;
    return h;
}

HalfEdge*
HalfEdge::kef()
{
    Loop	*f, *r;
    
    if (parent()->isFace())
    {
	f = parent();
	r = conj()->parent();
    }
    else
    {
	f = conj()->parent();
	r = parent();
    }
    HalfEdge* h = separate()->set_parent(r);
    f->put_children(r->parent())->detach();
    delete f;
    return h;
}

HalfEdge*
HalfEdge::kesr()
{
    Loop*	r_old = parent();
    Ptr<Ring>	r_new = new Ring;
    HalfEdge*	hn    = next();
    HalfEdge*	h     = separate()->set_parent(r_new);

    hn->set_parent(r_old);
    r_new->set_parent(r_old->parent())->get_children(r_old);
    return h;
}

HalfEdge*
HalfEdge::kemr()
{
    Loop*	f  = parent();
    Ptr<Ring>	r  = new Ring;
    HalfEdge*	hn = next();
    HalfEdge*	h  = separate();

    if (h->loop_dir() == FACE)
    {
	h ->set_parent(f);
	hn->set_parent(r);
    }
    else
    {
	h ->set_parent(r);
	hn->set_parent(f);
    }
    r->set_parent(f)->get_children(f->parent());
    return h;
}

HalfEdge*
HalfEdge::join(HalfEdge* hn, Cons<PointB>* g)
{
    Ptr<HalfEdge>	h, hc;
    
    if (this == hn)
    {
	h  = new HalfEdge;
	hc = (conj() == this ?
	      this : (new HalfEdge)->set_prev(prev())->set_next(hn));
	h->set_conj(hc);

	if (h->set_geom(g->cons(point()))->loop_dir() == RING)
	{
	    HalfEdge*	h_tmp = h;
	    h  = hc;
	    hc = h_tmp;
	}
    }
    else
    {
	h  = (conj()	==this ? this : (new HalfEdge)->set_prev(prev()));
	hc = (hn->conj()==hn   ? hn   : (new HalfEdge)->set_prev(hn->prev()));
	h ->set_next(hn);
	hc->set_next(this)->set_conj(h);
    }

    if (h == this)
	h->set_geom(geom()->rplacd(g));
    else
	h->set_geom(g->cons(point()));
    if (hc == hn)
	hc->set_geom(hn->geom()->rplacd(g->reverse()));
    else
	hc->set_geom(g->reverse()->cons(hn->point()));
    return h;
}

HalfEdge*
HalfEdge::separate()
{
    if (prev() == this)			// "this" forms a loop.
	if (conj()->prev() == conj())	// "hc" also forms a loop.
	    if (parent()->isFace())	// "this" is a Face.
	    {
		HalfEdge*	hc = conj()->set_conj(conj())
				    ->set_geom(conj()->geom()->rplacd(0));
		return hc;
	    }
	    else
	    {
		set_conj(this)->set_geom(geom()->rplacd(0));
		return this;
	    }
	else
	{
	    HalfEdge*	hcn = conj()->next()->set_prev(conj()->prev());
	    return hcn;
	}
    else if (conj()->prev() == conj())	// "hc" forms a loop.
    {
	HalfEdge*	hn = next()->set_prev(prev());
	return hn;
    }

    HalfEdge*	hcn = conj()->next();
    if (next() == conj())		// Origin of "hc" is an end point.
	conj()->set_conj(conj())->set_prev(conj())
	      ->set_geom(conj()->geom()->rplacd(0));
    else
	next()->set_prev(conj()->prev());
    if (hcn == this)			// Origin of "this" is an end point.
	set_conj(this)->set_prev(this)->set_geom(geom()->rplacd(0));
    else
	hcn->set_prev(prev());
    return hcn;
}

HalfEdge*
HalfEdge::set_parent(Loop* l)
{
    l->_head = this;
    HalfEdge* h = this;
    do
	h->_parent = l;
    while ((h = h->next()) != l->head());
    return this;
}

HalfEdge::LoopDir
HalfEdge::loop_dir() const
{
    Geometry	*g0 = geom(), *g1 = g0->next();
    int		ang = 0;

    do
    {
	Geometry* g2 = g1->next();
	ang += g1->point()->angle(g0->point(), g2->point());
	g0 = g1;
	g1 = g2;
    } while (g0 != geom());
    
    switch (ang)
    {
      case 8:
	return FACE;
      case 4:
      case -8:
	return RING;
    }
    throw std::invalid_argument("TU::Brep::HalfEdge::loop_dir\tLoop structure is broken!!");
    return FACE;
}

}
}
