/*
 *  $Id: TUv++.inst.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifdef __GNUG__

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
