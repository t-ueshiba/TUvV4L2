/*
 *  $Id: TUv++.inst.cc,v 1.3 2003-01-10 00:33:11 ueshiba Exp $
 */
#if defined(__GNUG__) || defined(__INTEL_COMPILER)

#include "TU/v/TUv++.h"
#include "TU/List++.cc"

namespace TU
{
namespace v
{
template class List<Cmd>;
template class List<Window>;
template class List<Pane>;
}
}

#endif
