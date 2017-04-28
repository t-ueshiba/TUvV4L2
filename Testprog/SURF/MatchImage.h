/*
 *  $Id: MatchImage.h,v 1.2 2010-11-22 06:29:02 ueshiba Exp $
 */
#ifndef __MATCHIMAGE_H
#define __MATCHIMAGE_H

#include "TU/Image++.h"
#include "TU/FeatureMatch.h"

namespace TU
{
/************************************************************************
*  class MatchImage							*
************************************************************************/
class MatchImage : public Image<RGB>
{
  public:
    MatchImage()	:Image<RGB>()					{}

    template <class IMG>
    Point2i	initializeH(IMG begin, IMG end)				;
    template <class IMG>
    Point2i	initializeV(IMG begin, IMG end)				;
    MatchImage&	copy(const Image<u_char>& image, size_t u0, size_t v0)	;
    template <class MATCH>
    MatchImage&	drawMatches(MATCH begin, MATCH end,
			    const Point2i& origin0,
			    const Point2i& origin1, bool green)		;

  private:
    MatchImage&	drawLine(const Point2i& p, const Point2i& q, bool green);
};
		    
template <class IMG> Point2i
MatchImage::initializeH(IMG begin, IMG end)
{
    using	element_type = Point2i::element_type;
    
    size_t	w = 0, h = 0;
    for (IMG image = begin; image != end; ++image)
    {
	w += image->width();
	if (image->height() > h)
	    h = image->height();
    }
    resize(h, w);

    size_t	u0 = width(), nimages = 0;
    for (IMG image = begin; image != end; ++image)
    {
	u0 -= image->width();
	copy(*image, u0, 0);
	++nimages;
    }

    if (nimages > 2)
	return Point2i({element_type(width() / nimages), 0});
    else if (nimages == 2)
	return Point2i({element_type((++begin)->width()), 0});
    else
	return Point2i({0, 0});
}

template <class IMG> Point2i
MatchImage::initializeV(IMG begin, IMG end)
{
    using element_type	= Point2i::element_type;
    
    size_t	w = 0, h = 0;
    for (IMG image = begin; image != end; ++image)
    {
	if (image->width() > w)
	    w = image->width();
	h += image->height();
    }
    resize(h, w);

    size_t	v0 = height(), nimages = 0;
    for (IMG image = begin; image != end; ++image)
    {
	v0 -= image->height();
	copy(*image, 0, v0);
	++nimages;
    }

    if (nimages > 2)
	return Point2i({element_type(0), element_type(height() / nimages)});
    else if (nimages == 2)
	return Point2i({element_type(0), element_type((++begin)->height())});
    else
	return Point2i({element_type(0), element_type(0)});
}

template <class MATCH> MatchImage&
MatchImage::drawMatches(MATCH begin, MATCH end,
			const Point2i& origin0, const Point2i& origin1,
			bool green)
{
    for (MATCH match = begin; match != end; ++match)
	drawLine(match->first + origin0, match->second + origin1, green);

    return *this;
}
    
}
#endif
