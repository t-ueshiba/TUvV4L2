/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id: Colormap.cc,v 1.5 2008-09-10 05:12:03 ueshiba Exp $  
 */
#include "TU/v/Colormap.h"
#include <stdexcept>
#include <sstream>

namespace TU
{
namespace v
{
/************************************************************************
*  Static functions							*
************************************************************************/
inline u_int	max(u_int a, u_int b)	{return (a > b ? a : b);}
inline u_int	min(u_int a, u_int b)	{return (a < b ? a : b);}
inline double	slant(int b, int a)	{return (a > 0 && b > 0 ?
						 double(b) / double(a) : 0.0);}
static u_int
underlayResolution(u_int resolutionRequired, const XVisualInfo& vinfo)
{
    switch (vinfo.c_class)
    {
      case GrayScale:
      case PseudoColor:
      case DirectColor:
	return resolutionRequired;
    }
    return vinfo.colormap_size;
}

/************************************************************************
*  class Colormap							*
************************************************************************/
/*
 *  public member functions
 */
Colormap::Colormap(Display* display, const XVisualInfo& vinfo)
    :_display(display),
     _vinfo(vinfo),
     _colormap(_vinfo.visual != DefaultVisual(_display, _vinfo.screen) ?
	       XCreateColormap(_display, RootWindow(_display, _vinfo.screen),
			       _vinfo.visual, AllocNone) :
	       DefaultColormap(_display, _vinfo.screen)),
   // underlay stuffs
     _resolution(0),
     _saturation(256),
   // overlay stuffs
     _overlayPlanes(0),
   // colorcube stuffs
     _colorcubeNPixels(0),
     _gStride(0),
     _bStride(0),
     _rMul(0),
     _gMul(0),
     _bMul(0),
   // general stuffs
     _pixels(0, 0),
     _mode(RGBColor),
     _map(Graymap)
{
}

Colormap::Colormap(Display* display, const XVisualInfo& vinfo,
			 Mode mode, u_int resolution,
			 u_int underlayCmapDim, u_int overlayDepth,
			 u_int rDim, u_int gDim, u_int bDim)
    :_display(display),
     _vinfo(vinfo),
     _colormap(XCreateColormap(_display, RootWindow(_display, _vinfo.screen),
			       _vinfo.visual, AllocNone)),
   // underlay stuffs
     _resolution(underlayResolution(resolution, _vinfo)),
     _saturation(256),
   // overlay stuffs
     _overlayPlanes(0),
   // colorcube stuffs
     _colorcubeNPixels(rDim * gDim * bDim),
     _gStride(rDim),
     _bStride(rDim * gDim),
     _rMul((rDim-1) / 255.0),
     _gMul((gDim-1) / 255.0),
     _bMul((bDim-1) / 255.0),
   // general stuffs
     _pixels((_vinfo.c_class == GrayScale   ||
	      _vinfo.c_class == PseudoColor ||
	      _vinfo.c_class == DirectColor ?  1 << overlayDepth : 1),
	     (_vinfo.c_class == GrayScale   ||
	      _vinfo.c_class == DirectColor ?  _resolution + underlayCmapDim : 
	      _vinfo.c_class == PseudoColor ?
	      max(_resolution + underlayCmapDim,
		  (_colorcubeNPixels - 1) / (1 << overlayDepth) + 1) :
	      _resolution)),
     _mode(mode),
     _map(Graymap)
{
  // Set dither mask.
    _dithermask[0][0] = 0.0;
    _dithermask[0][1] = 0.5;
    _dithermask[0][2] = 0.125;
    _dithermask[0][3] = 0.625;
    _dithermask[1][0] = 0.75;
    _dithermask[1][1] = 0.25;
    _dithermask[1][2] = 0.875;
    _dithermask[1][3] = 0.375;
    _dithermask[2][0] = 0.1875;
    _dithermask[2][1] = 0.6875;
    _dithermask[2][2] = 0.0625;
    _dithermask[2][3] = 0.5625;
    _dithermask[3][0] = 0.9375;
    _dithermask[3][1] = 0.4375;
    _dithermask[3][2] = 0.8125;
    _dithermask[3][3] = 0.3125;

  // Allocate color cells from X colormap if needed.
    switch (_vinfo.c_class)
    {
      case GrayScale:
      case PseudoColor:
      case DirectColor:
      {
      // Allocate private color cells and planes.
	u_long	planes[8*sizeof(u_long)];
	if (!XAllocColorCells(_display, _colormap, False, planes, overlayDepth,
			      _pixels[0], _pixels.ncol()))
	{
	    throw std::runtime_error("TU::v::Colormap::Colormap(): failed to allocate private colors.");
	}

      // Set overlay pixel values by ORing planes obtained.
	for (u_int overlay = 1; overlay < _pixels.nrow(); ++overlay)
	{
	    u_long	mask = 0;
	    for (u_int i = 0; i < overlayDepth; ++i)
		if ((overlay >> i) & 0x01)
		    mask |= planes[i];
	    for (u_int underlay = 0; underlay < _pixels.ncol(); ++underlay)
		_pixels[overlay][underlay] = _pixels[0][underlay] | mask;
	}

      // Set overlay planes.
	for (u_int i = 0; i < overlayDepth; ++i)
	    _overlayPlanes |= planes[i];
      }
      break;
    }

  // Set gray values.
    setGraymapInternal();

  // Setup the underlay lookup table.
    setSaturation(_saturation);

  // Setup the overlay lookup table.
    double	delta = slant(_pixels.nrow() - 1, 255);
    int		i;
    for (i = 0; i < 256; ++i)
	_overlayTable[i] = _pixels[u_int(0.5 + i*delta)][0] & _overlayPlanes;
}

Colormap::~Colormap()
{
    XFreeColors(_display, _colormap, _pixels[0], _pixels.ncol(),
		_overlayPlanes);
    XFreeColormap(_display, _colormap);
}

void
Colormap::setGraymap()
{
    if (_map != Graymap)
    {
	_map = Graymap;
	setSaturation(_saturation);
	if (_vinfo.c_class == PseudoColor)
	    setGraymapInternal();
    }
}

void
Colormap::setSignedmap()
{
    if (_map != Signedmap)
    {
	_map = Signedmap;
	setSaturation(_saturation);
	if (_vinfo.c_class == PseudoColor)
	    setSignedmapInternal();
    }
}

void
Colormap::setColorcube()
{
    if (_map != Colorcube)
    {
	_map = Colorcube;
	setSaturation(_saturation);
	if (_vinfo.c_class == PseudoColor)
	    setColorcubeInternal();
    }
}

void
Colormap::setSaturation(u_int saturation)
{
    _saturation = saturation;
    
    u_int	resolution1 = (_resolution > 0 ? _resolution - 1 : 0);
    if (_map == Signedmap)
    {
	if (_vinfo.c_class == PseudoColor)
	{
	    double	delta = slant((_resolution - 1)/2, _saturation - 1);
	    int		val = 0;
	    for (; val < min(_saturation, 128); ++val)
		_underlayTable[val] = _pixels[0][u_int(0.5 + val*delta)];
	    for (; val < 128; ++val)
		_underlayTable[val] = _pixels[0][resolution1/2];
	    delta = slant(_resolution/2 - 1, _saturation - 1);
	    for (; val < 256 - _saturation; ++val)
		_underlayTable[val] = _pixels[0][_resolution - _resolution/2];
	    for (; val < 256; ++val)
		_underlayTable[val] = _pixels[0][_resolution - 1 -
						 u_int(0.5 + (255-val)*delta)];
	}
	else
	{
	    double	delta = slant(_resolution - 1, _saturation - 1);
	    int		val = 0;
	    for (; val < min(_saturation, 128); ++val)
		_underlayTable[val] = _pixels[0][u_int(0.5 + val*delta)];
	    delta = slant(_resolution - 1, _saturation);
	    for (; val < 256 - _saturation; ++val)
		_underlayTable[val] = _pixels[0][resolution1];
	    for (; val < 256; ++val)
		_underlayTable[val] = _pixels[0][u_int(0.5 + (256-val)*delta)];
	}
    }
    else if (_map != Colorcube)
    {
	double	delta = slant(_resolution - 1, _saturation - 1);
	int	val = 0;
	for (; val < min(_saturation, 256); ++val)
	    _underlayTable[val] = _pixels[0][u_int(0.5 + val*delta)];
	for (; val < 256; ++val)
	    _underlayTable[val] = _pixels[0][resolution1];
    }
}

/*
 *  underlay stuffs
 */
u_long
Colormap::getUnderlayPixel(u_int index) const
{
    if (_mode != IndexedColor)
    {
	throw std::runtime_error("TU::v::Colormap::getUnderlayPixel: Not in IndexedColor mode!!");
    }
    if (_resolution + index >= _pixels.ncol())
    {
        std::ostringstream	msg;
	msg << "TU::v::Colormap::getUnderlayPixel: Index (" << index
	    << ") is not within the range [" << 0 << ", "
	    << _pixels.ncol() - _resolution << ")!" << std::endl;
	throw std::out_of_range(msg.str());
    }
    return _pixels[0][_resolution + index];
}

BGR
Colormap::getUnderlayValue(u_int index) const
{
    XColor	color;
    color.pixel = getUnderlayPixel(index);
    XQueryColor(_display, _colormap, &color);
    BGR		bgr;
    bgr.r = color.red   >> 8;
    bgr.g = color.green >> 8;
    bgr.b = color.blue  >> 8;
    return bgr;
}

void
Colormap::setUnderlayValue(u_int index, BGR bgr)
{
    if (_mode != IndexedColor)
    {
	throw std::runtime_error("TU::v::Colormap::setUnderlayValue: Not in IndexedColor mode!!");
    }
    if (_resolution + index >= _pixels.ncol())
    {
	std::ostringstream	msg;
	msg << "TU::v::Colormap::setUnderlayValue: Index (" << index
	    << ") is not within the range [" << 0 << ", "
	    << _pixels.ncol() - _resolution
	    << ")!" << std::endl;
	throw std::out_of_range(msg.str());
    }

    XColor	color;
    color.red   = bgr.r << 8;
    color.green = bgr.g << 8;
    color.blue  = bgr.b << 8;
    color.pixel = _pixels[0][_resolution + index];
    color.flags = DoRed | DoGreen | DoBlue;
    XStoreColor(_display, _colormap, &color);
}

Array<u_long>
Colormap::getUnderlayPixels() const
{
    if (_mode == IndexedColor)
    {
	Array<u_long>	pixels(_pixels.ncol() - _resolution);
	for (int i = 0; i < pixels.dim(); ++i)
	    pixels[i] = _pixels[0][_resolution + i];	// user-defined pixels.
	return pixels;
    }
    else
    {
	Array<u_long>	pixels(_vinfo.c_class == PseudoColor ?
			       _colorcubeNPixels : _resolution);
	for (int i = 0; i < pixels.dim(); ++i)
	    pixels[i] = _pixels[0][i];			// ordinary pixels.
	return pixels;
    }
}

/*
 *  overlay stuffs
 */
u_long
Colormap::getOverlayPixel(u_int index) const
{
    if (_mode != IndexedColor)
    {
	throw std::runtime_error("TU::v::Colormap::getOverlayPixel: Not in IndexedColor mode!!");
    }
    if (index >= _pixels.nrow())
    {
	std::ostringstream	msg;
	msg << "TU::v::Colormap::getOverlayPixel: Index (" << index
	    << ") is not within the range [" << 0 << ", " << _pixels.nrow()
	    << ")!" << std::endl;
	throw std::out_of_range(msg.str());
    }
    return _pixels[index][0];
}

BGR
Colormap::getOverlayValue(u_int index) const
{
    XColor	color;
    color.pixel = getOverlayPixel(index);
    XQueryColor(_display, _colormap, &color);
    BGR	bgr;
    bgr.r = color.red   >> 8;
    bgr.g = color.green >> 8;
    bgr.b = color.blue  >> 8;
    return bgr;
}

void
Colormap::setOverlayValue(u_int index, BGR bgr)
{
    if (_mode != IndexedColor)
    {
	throw std::runtime_error("TU::v::Colormap::setOverlayValue: Not in IndexedColor mode!!");
    }
    if (index < 1 || index >= _pixels.nrow())
    {
	std::ostringstream	msg;
	msg << "TU::v::Colormap::setOverlayValue: Index (" << index
	    << ") is not within the range [" << 1 << ", " << _pixels.nrow()
	    << ")!" << std::endl;
	throw std::out_of_range(msg.str());
    }

    XColor	color;
    color.red	= bgr.r << 8;
    color.green	= bgr.g << 8;
    color.blue	= bgr.b << 8;
    color.flags = DoRed | DoGreen | DoBlue;
    for (int i = 0; i < _pixels.ncol(); ++i)
    {
	color.pixel = _pixels[index][i];
	XStoreColor(_display, _colormap, &color);
    }
}

Array<u_long>
Colormap::getOverlayPixels() const
{
    Array<u_long>	pixels(_pixels.nrow());
    for (int i = 0; i < pixels.dim(); ++i)
	pixels[i] = _pixels[i][0];	// transparent and user-defined pixels.
    return pixels;
}

/*
 *  Private member functions
 */
void
Colormap::setGraymapInternal()
{
  // Set underlay gray map.
    double	delta = slant(255, _resolution - 1);
    int		i;
    for (i = 0; i < _resolution; ++i)
    {
	XColor	color;
	color.red = color.green = color.blue = u_char(0.5 + i*delta) << 8;
	color.flags = DoRed | DoGreen | DoBlue;

	switch (_vinfo.c_class)
	{
	  case GrayScale:
	  case PseudoColor:
	  case DirectColor:
	    color.pixel = _pixels[0][i];
	    XStoreColor(_display, _colormap, &color);
	    break;
	
	  default:
	    if (!XAllocColor(_display, _colormap, &color))
	    {
		throw std::runtime_error("TU::v::Colormap::setGraymapInternal: Failed to allocate shared color!!");
	    }
	    _pixels[0][i] = color.pixel;
	    break;
	}
    }

  // Set overlay gray map.    
    delta = slant(255, _pixels.nrow() - 1);
    for (i = 1; i < _pixels.nrow(); ++i)	// Keep transparency pixel:i=0.
    {
	XColor	color;
	color.red = color.green = color.blue = u_char(0.5 + i*delta) << 8;
	color.flags = DoRed | DoGreen | DoBlue;
	for (int j = 0; j < _pixels.ncol(); ++j)
	{
	    color.pixel = _pixels[i][j];
	    XStoreColor(_display, _colormap, &color);
	}
    }
}

void
Colormap::setSignedmapInternal()
{
    if (_vinfo.c_class == PseudoColor)
    {
      // Set underlay signed map.
	double	delta  = slant(255, (_resolution - 1)/2),
		delta2 = slant(255, _resolution/2 - 1);
	for (int i = 0; i < _resolution; ++i)
	{
	    XColor	color;
	    color.red = color.green = color.blue = 0;
	    if (i < (_resolution + 1)/2)
		color.red = u_char(0.5 + i*delta) << 8;
	    else
		color.green = u_char(0.5 + (_resolution - 1 - i)*delta2) << 8;
	    color.flags = DoRed | DoGreen | DoBlue;
	    color.pixel = _pixels[0][i];
	    XStoreColor(_display, _colormap, &color);
	}
    }
    else
	setGraymapInternal();
}

void
Colormap::setColorcubeInternal()
{
    double	rStep = (rDim() > 1 ? 255.0 / (rDim()-1) : 0.0),
		gStep = (gDim() > 1 ? 255.0 / (gDim()-1) : 0.0),
		bStep = (bDim() > 1 ? 255.0 / (bDim()-1) : 0.0);
    for (u_int b = 0; b < bDim(); ++b)
    {
	XColor	color;
	color.flags = DoRed | DoGreen | DoBlue;
	
	color.blue = u_char(0.5 + b * bStep) << 8;
	for (u_int g = 0; g < gDim(); ++g)
	{
	    color.green = u_char(0.5 + g * gStep) << 8;
	    for (u_int r = 0; r < rDim(); ++r)
	    {
		color.red = u_char(0.5 + r * rStep) << 8;
		color.pixel = _pixels[0][r + g*_gStride + b*_bStride];
		XStoreColor(_display, _colormap, &color);
	    }
	}
    }
}

}
}
