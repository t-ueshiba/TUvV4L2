/*
 *  $Id: TUXil++.inst.cc,v 1.2 2002-07-25 02:38:08 ueshiba Exp $
 */
#ifdef __GNUG__

#include "TU/v/XilDC.cc"

namespace TU
{
template class XilImage<s_char>;
template class XilImage<u_char>;
//template class XilImage<short>;
template class XilImage<BGR>;
template class XilImage<ABGR>;
}

#endif
