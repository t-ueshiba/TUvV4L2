/*
 *  $Id: TUv++.inst.cc,v 1.4 2003-01-10 00:59:19 ueshiba Exp $
 */
#if defined(__GNUG__) || defined(__INTEL_COMPILER)

#include "TU/v/TUv++.h"
#include "TU/List++.cc"

namespace TU
{
template class List<v::Cmd>;
template class List<v::Window>;
template class List<v::Pane>;
}

#endif
