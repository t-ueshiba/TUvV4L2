/*
 *  $Id: print.cc,v 1.1 2002-07-25 04:36:08 ueshiba Exp $
 */
#include "TU/Collection++.h"
#include "Int.h"

namespace TU
{
void
ObjTreeBase::Node::print(std::ostream& out) const
{
    if (this == 0)
	return;
    _left->print(out);
    Int*	p = (Int*)_p;
    out << p;
    _right->print(out);
}
 
}
