/*
 *  $Id: Image.cc,v 1.2 2002-07-25 02:38:05 ueshiba Exp $
 */
#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class Image<YUV411>							*
************************************************************************/
void
Image<YUV411>::resize(u_int h, u_int w)
{
    Array2<ImageLine<YUV411> >::resize(h, w/2);
}

void
Image<YUV411>::resize(YUV411* p, u_int h, u_int w)
{
    Array2<ImageLine<YUV411> >::resize(p, h, w/2);
}

}

#ifdef __GNUG__
#  include "TU/Array++.cc"

namespace TU
{
template void	Array2<ImageLine<YUV411> >::resize(u_int, u_int)	;
template void	Array2<ImageLine<YUV411> >::resize(YUV411*,
						   u_int, u_int)	;
}

#endif

