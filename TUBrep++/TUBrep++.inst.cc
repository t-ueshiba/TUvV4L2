/*
 *  $Id: TUBrep++.inst.cc,v 1.3 2004-03-08 02:05:01 ueshiba Exp $
 */

#if defined(__GNUG__) || defined(__INTEL_COMPILER)

#include "TU/Brep/Brep++.h"
#include "TU/Object++.cc"

namespace TU
{
template class Cons<Brep::PointB>;
}

#endif
