/*
 *  $Id: PointB.cc,v 1.2 2002-07-25 02:37:36 ueshiba Exp $
 */
#include "TU/Brep/Brep++.h"

namespace TU
{
namespace Brep
{
/*--------------------- Public member functions -------------------------*/

int
PointB::adj(const Neighbor& nbr) const
{
    for (int i = 0; i < nbr.nelm(); i++)
	for (const Geometry* g = nbr[i].head(); ; g = g->next())
	{
	    switch (adj(g->point()))
	    {
	      case -1:
	      case 1:
		return i;
	    }
	    if (g == nbr[i].tail())
		break;
	}
    return -1;
}

/*--------------------- Private member functions -------------------------*/

int
PointB::check(const Cons<PointB>* g) const
{
    int		npoint = 0, flags = 0;
    
    for (; !g->null(); g = g->cdr(), npoint++)
	switch (adj(((Geometry*)g)->point()))
	{
	  case -1:		// identical point
	    return -1;
	  case 1:		// neighbor point
	    switch (npoint)
	    {
	      case 0:			// 1st point is in neighbor.
		break;
	      case 1:			// 2nd point is in neighbor.
		flags |= SECOND;
		break;
	      case 2:			// 3rd point is in neighbor.
		return (flags & SECOND ? -1: flags | CROSS);
	      default:
		return (flags | CROSS);
	    }
	    break;
	}
    return flags;
}

void
PointB::saveGuts(std::ostream& out) const
{
    out.write((const char*)&(*this)[0], 2*sizeof((*this)[0]));
}

void
PointB::restoreGuts(std::istream& in)
{
    in.read((char*)&(*this)[0], 2*sizeof((*this)[0]));
}
 
}
}
