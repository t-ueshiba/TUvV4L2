/*
 *  $Id: Image.cc,v 1.3 2002-07-25 11:53:22 ueshiba Exp $
 */
#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class Image<YUV411>							*
************************************************************************/
template <> void
Image<YUV411>::resize(u_int h, u_int w)
{
    Array2<ImageLine<YUV411> >::resize(h, w/2);
}

template <> void
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

