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

    template <class ITER>
    Point2i	initializeH(ITER begin, ITER end)			;
    template <class ITER>
    Point2i	initializeV(ITER begin, ITER end)			;
    MatchImage&	copy(const Image<u_char>& image, int u0, int v0)	;
    MatchImage&	drawMatches(const FeatureMatch::MatchSet& matchSet,
			    const Point2i& origin0,
			    const Point2i& origin1, bool green)		;

  private:
    MatchImage&	drawLine(const Point2i& p, const Point2i& q, bool green);
};
		    
template <class ITER> Point2i
MatchImage::initializeH(ITER begin, ITER end)
{
    size_t	w = 0, h = 0;
    for (ITER image = begin; image != end; ++image)
    {
	w += image->width();
	if (image->height() > h)
	    h = image->height();
    }
    resize(h, w);

    size_t	u0 = width(), nimages = 0;
    for (ITER image = begin; image != end; ++image)
    {
	u0 -= image->width();
	copy(*image, u0, 0);
	++nimages;
    }

    if (nimages > 2)
	return Point2i(width() / nimages, 0);
    else if (nimages == 2)
	return Point2i((++begin)->width(), 0);
    else
	return Point2i(0, 0);
}

template <class ITER> Point2i
MatchImage::initializeV(ITER begin, ITER end)
{
    size_t	w = 0, h = 0;
    for (ITER image = begin; image != end; ++image)
    {
	if (image->width() > w)
	    w = image->width();
	h += image->height();
    }
    resize(h, w);

    size_t	v0 = height(), nimages = 0;
    for (ITER image = begin; image != end; ++image)
    {
	v0 -= image->height();
	copy(*image, 0, v0);
	++nimages;
    }

    if (nimages > 2)
	return Point2i(0, height() / nimages);
    else if (nimages == 2)
	return Point2i(0, (++begin)->height());
    else
	return Point2i(0, 0);
}

}
#endif
