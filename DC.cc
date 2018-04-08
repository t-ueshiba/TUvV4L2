/*
 *  $Id$  
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
DC::setSize(size_t width, size_t height, float zoom)
{
    _width  = width;
    _height = height;
    _zoom   = zoom;
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
x0_125(DC& vDC)
{
    return vDC.setZoom(0.125);
}

DC&
x0_25(DC& vDC)
{
    return vDC.setZoom(0.25);
}

DC&
x0_5(DC& vDC)
{
    return vDC.setZoom(0.5);
}

DC&
x1(DC& vDC)
{
    return vDC.setZoom(1);
}

DC&
x1_5(DC& vDC)
{
    return vDC.setZoom(1.5);
}

DC&
x2(DC& vDC)
{
    return vDC.setZoom(2);
}

DC&
x4(DC& vDC)
{
    return vDC.setZoom(4);
}

DC&
x8(DC& vDC)
{
    return vDC.setZoom(8);
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

OManip1<DC, size_t>
foreground(size_t fg)
{
    return OManip1<DC, size_t>(&DC::setForeground, fg);
}

OManip1<DC, size_t>
background(size_t bg)
{
    return OManip1<DC, size_t>(&DC::setBackground, bg);
}

OManip1<DC, size_t>
thickness(size_t thick)
{
    return OManip1<DC, size_t>(&DC::setThickness, thick);
}

OManip1<DC, size_t>
saturation(size_t s)
{
    return OManip1<DC, size_t>(&DC::setSaturation, s);
}

OManip1<DC, float>
saturationF(float s)
{
    return OManip1<DC, float>(&DC::setSaturationF, s);
}

OManip2<DC, int, int>
offset(int u, int v)
{
    return OManip2<DC, int, int>(&DC::setOffset, u, v);
}

}
}
