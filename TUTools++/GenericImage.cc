/*
 *  $Id: GenericImage.cc,v 1.1 2008-03-12 00:12:34 ueshiba Exp $
 */
#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class GenericImage							*
************************************************************************/
std::istream&
GenericImage::restoreData(std::istream& in)
{
    for (int v = 0; v < height(); )
  	if (!(*this)[v++].restore(in))
	    break;
    return in;
}

std::ostream&
GenericImage::saveData(std::ostream& out) const
{
    for (int v = 0; v < height(); )
	if (!(*this)[v++].save(out))
	    break;
    return out;
}

u_int
GenericImage::_width() const
{
    return (ncol()*8) / type2depth(_type);
}

u_int
GenericImage::_height() const
{
    return nrow();
}

void
GenericImage::_resize(u_int h, u_int w, ImageBase::Type type)
{
    w = (type2depth(type)*w - 1)/8 + 1;
    Array2<Array<u_char> >::resize(h, w);
}

}
