/*
 *  $Id: DC.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/DC.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class DC:		coorindate system for drawing			*
************************************************************************/
DC::~DC()
{
}

DC&
DC::setSize(u_int width, u_int height, u_int mul, u_int div)
{
    _width  = width;
    _height = height;
    _mul    = mul;
    _div    = div;
    return *this;
}

DC&
DC::setOffset(int u0, int v0)
{
    _offset = Point2<int>(u0, v0);
    return *this;
}

DC&
DC::setLayer(Layer layer)
{
    _layer = layer;
    return *this;
}

DC&
DC::repaint()	// This action is never invoked by Expose events.
{
    if (getLayer() == UNDERLAY)
	return repaintUnderlay();
    else
	return repaintOverlay();
}

DC&
DC::repaintAll() // This action is invoked by Expose events or application.
{
    Layer	layer = getLayer();	// Back-up original state.
    setLayer(UNDERLAY);
    repaintUnderlay();
    setLayer(OVERLAY);
    repaintOverlay();
    return setLayer(layer);		// Restore the original state.
}

/************************************************************************
*  Manipulators								*
************************************************************************/
DC&
x0_25(DC& vDC)
{
    return vDC.setZoom(1, 4);
}

DC&
x0_5(DC& vDC)
{
    return vDC.setZoom(1, 2);
}

DC&
x1(DC& vDC)
{
    return vDC.setZoom(1, 1);
}

DC&
x1_5(DC& vDC)
{
    return vDC.setZoom(3, 2);
}

DC&
x2(DC& vDC)
{
    return vDC.setZoom(2, 1);
}

DC&
x4(DC& vDC)
{
    return vDC.setZoom(4, 1);
}

DC&
underlay(DC& vDC)
{
    return vDC.setLayer(DC::UNDERLAY);
}

DC&
overlay(DC& vDC)
{
    return vDC.setLayer(DC::OVERLAY);
}

DC&
dot(DC& vDC)
{
    return vDC.setPointStyle(DC::DOT);
}

DC&
cross(DC& vDC)
{
    return vDC.setPointStyle(DC::CROSS);
}

DC&
circle(DC& vDC)
{
    return vDC.setPointStyle(DC::CIRCLE);
}

DC&
clear(DC& vDC)
{
    return vDC.clear();
}

DC&
repaint(DC& vDC)
{
    return vDC.repaint();
}

DC&
repaintAll(DC& vDC)
{
    return vDC.repaintAll();
}

DC&
sync(DC& vDC)
{
    return vDC.sync();
}

OManip1<DC, const BGR&>
foreground(const BGR& fg)
{
    return OManip1<DC, const BGR&>(&DC::setForeground, fg);
}

OManip1<DC, const BGR&>
background(const BGR& bg)
{
    return OManip1<DC, const BGR&>(&DC::setBackground, bg);
}

OManip1<DC, u_int>
foreground(u_int fg)
{
    return OManip1<DC, u_int>(&DC::setForeground, fg);
}

OManip1<DC, u_int>
background(u_int bg)
{
    return OManip1<DC, u_int>(&DC::setBackground, bg);
}

OManip1<DC, u_int>
thickness(u_int thick)
{
    return OManip1<DC, u_int>(&DC::setThickness, thick);
}

OManip1<DC, u_int>
saturation(u_int s)
{
    return OManip1<DC, u_int>(&DC::setSaturation, s);
}

OManip2<DC, int, int>
offset(int u, int v)
{
    return OManip2<DC, int, int>(&DC::setOffset, u, v);
}

}
}
