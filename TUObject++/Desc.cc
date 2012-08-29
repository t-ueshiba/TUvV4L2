/*
 *  $Id: Desc.cc,v 1.5 2012-08-29 21:17:03 ueshiba Exp $
 */
#include "Object++_.h"
#include <stdarg.h>

namespace TU
{
/************************************************************************
*  Object::Desc: table for constructors of objects associated with	*
*		   classID						*
************************************************************************/
Object::Desc::Desc(u_short id, u_short bid, Pftype ctor, ...)
    :_id(id), _bid(bid), _pf(ctor), _p(0), _init(_bid == 0)
{
#ifdef TUObjectPP_DEBUG
    std::cerr << "Desc::Desc(): myId = " << _id << ", baseId = " << _bid
	      << std::endl;
#endif
    if (_ndescs++ == 0)
	_map = new Map;
    (*_map)[_id] = this;

    u_int       i = 0;
    va_list     args;
    va_start(args, ctor);
    for (Mbrp q; (q = va_arg(args, Mbrp)) != 0; )
	++i;
    va_end(args);
#ifdef TUObjectPP_DEBUG
    std::cerr << "  " << i << " members found..." << std::endl;
#endif
    _p = new Mbrp[i+1];
    va_start(args, ctor);
    for (i = 0; (_p[i] = va_arg(args, Mbrp)) != 0; )
	++i;
    va_end(args);

    for (Map::iterator i = _map->begin(); i != _map->end(); ++i)
    {
#ifdef TUObjectPP_DEBUG
        std::cerr << "  setting pointers to member functions of class "
		  << (*i).first << "..." << std::endl;
#endif
	(*i).second->setMbrp();
    }
}

Object::Desc::~Desc()
{
    delete [] _p;
    if (--_ndescs == 0)
	delete _map;
}

u_int
Object::Desc::nMbrp() const
{
    u_int       i = 0;
    while (_p[i] != 0)
	++i;
    return i;
}

bool
Object::Desc::setMbrp()
{
    if (!_init)		// Not initialized yet?
    {
	Desc*	base = (_map->find(_bid) == _map->end() ? 0 : (*_map)[_bid]);

	if (base != 0 && base->setMbrp() == true)
	{
	    Mbrp* q = new Mbrp[base->nMbrp() + nMbrp() + 1];
	    int i = 0;
	    while ((q[i] = base->_p[i]) != 0)
		++i;
	    for (int j = 0; (q[i] = _p[j]) != 0; ++j)
		++i;
	    delete [] _p;
	    _p = q;

	    _init = true;
	}
    }

    return _init;
}
 
}
