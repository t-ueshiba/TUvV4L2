/*
 *  $Id$
 */
#include "TU/Brep/Brep++.h"

namespace TU
{
namespace Brep
{
int
Neighbor::check() const
{
    for (int i = 1; i < nelm(); ++i)
	if (_elm[i-1].tail()->point()->adj(_elm[i].head()->point()) == 1)
	    return -1;
    if (nelm() > 1 &&
	_elm[nelm()-1].tail()->point()->adj(_elm[0].head()->point()) == 1)
	return -1;
    if (nelm() == 1 && _elm[0].head() != _elm[0].tail())
	return -2;
    return nelm();
}

void
Neighbor::inFace(const Face* f)
{
    onLoop(f);
    for (Loop::ChildIterator iter(f); iter; ++iter)
	onLoop(iter);
    sort();
}

void
Neighbor::onLoop(const Loop* l)
{
    struct
    {
	Geometry	*gs, *ge;
	int		isreachable;
    } nbr[8];

    if (!l->head())
	return;
    
    int		i = -1, isneighbor = 0, isreachable = 0;
    Geometry* g = l->head()->geom();
    do
	switch (g->point()->adj(_p))
	{
	  case 0:			// "g" is not adjacent to "p".
	    isneighbor = isreachable = 0;
	    break;
	  case -1:			// "g" coincides with "p".
	    isreachable = 1;
	  case 1:			// "g" is adjacent to "p".
	    if (g->sameside(_p) == g)
	    {
		if (!isneighbor)
		{
		    nbr[++i].gs = g;
		    isneighbor = 1;
		}
		nbr[i].ge = g;
		nbr[i].isreachable = isreachable;
	    }
	    else
		isneighbor = isreachable = 0;
	    break;
	}
    while ((g = g->next()) != l->head()->geom());

    if (i >= 0 && nbr[i].ge->next() == nbr[0].gs)
	if (i > 0)
	{
	    nbr[0].gs = nbr[i].gs;
	    nbr[0].isreachable |= nbr[i--].isreachable;
	}
	else if (nbr[0].gs->point()->angle(_p, nbr[0].gs->next()->point()) < 0)
	{
	    Geometry* tmp = nbr[0].gs;
	    nbr[0].gs = nbr[0].ge;
	    nbr[0].ge = tmp;
	}

    for (int j = 0; j <= i; ++j)
	if (!nbr[j].isreachable)
	    _elm[_n++].set(nbr[j].gs, nbr[j].ge);
}

void
Neighbor::sort()
{
    const PointB* const q = (nelm() && _p == _pn ?
			       _elm[0].head()->point() : _pn);
    for (int i = 0; i < nelm(); ++i)
    {
	int	min = i;
	
	for (int j = i + 1; j < nelm(); ++j)
	    if (_p->angle(q, _elm[j]  .head()->point()) <
		_p->angle(q, _elm[min].head()->point()))
		min = j;
	if (min != i)
	{
	    Element tmp	 = _elm[i];
	    _elm[i]	 = _elm[min];
	    _elm[min]	 = tmp;
	}
    }
}
 
}
}
