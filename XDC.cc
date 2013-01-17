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
 *  $Id$  
 */
#include "TU/v/XDC.h"
#include <stdexcept>
#include <X11/XWDFile.h>
#include <arpa/inet.h>

namespace TU
{
namespace v
{
/************************************************************************
*  static functions							*
************************************************************************/
inline u_short
lowbit(u_short x)
{
    return x & (~x + 1);
}

template <class S> inline int
uu(int u)
{
    return u;
}

template <> inline int
uu<YUV411>(int u)
{
    return u / 2;
}

/************************************************************************
*  class XDC								*
************************************************************************/
/*
 *  Private template member functions
 */
template <class S> void
XDC::createXImage(const Image<S>& image)
{
    const int	buffWidth  = log2devR(image.width()),
		buffHeight = log2devR(image.height());
    if (_ximage == 0 ||
	_ximage->width != buffWidth || _ximage->height != buffHeight)
	allocateXImage(buffWidth, buffHeight);

    switch (_ximage->bits_per_pixel)
    {
      case 8:
	fillBuff<S, u_char>(image);
	break;

      case 16:
	if (_ximage->byte_order == MSBFirst)
	    fillBuff<S, XDC::BPixel16>(image);
	else
	    fillBuff<S, XDC::LPixel16>(image);
	break;

      case 24:
	if (_ximage->byte_order == MSBFirst)
	    fillBuff<S, XDC::BPixel24>(image);
	else
	    fillBuff<S, XDC::LPixel24>(image);
	break;

      case 32:
	if (_ximage->byte_order == MSBFirst)
	    fillBuff<S, XDC::BPixel32>(image);
	else
	    fillBuff<S, XDC::LPixel32>(image);
	break;

      default:
	throw std::domain_error("TU::v::XDC::createXImage: unsupported value of bits_per_pixel!");
	break;
    }
}

template <class S, class T> void
XDC::fillBuff(const Image<S>& image)
{
    char* const	buff = _ximage->data;
    const int	buffWidth = _ximage->width, buffHeight = _ximage->height,
		bytesPerLine = _ximage->bytes_per_line;
    
    if (image.width() == buffWidth && image.height() == buffHeight)
    {
	for (u_int v = 0; v < image.height(); ++v)
	{
	    const S*	p = image[v].data();
	    T*		q = (T*)(buff + v * bytesPerLine);
	    for (register int u = 0; u < image.width(); ++u)
		*q++ = _colormap.getUnderlayPixel(p[uu<S>(u)], u, v);
	}
    }
    else
    {
	if (image.width() < buffWidth)	// expansion
	{
	    for (int vs = -1, vdp = 0, vd = 0; vd < buffHeight; ++vd)
		if ((vs+1)*buffHeight <= vd*image.height())
		{
		    const S*	p = image[++vs].data();
		    T*		q = (T*)(buff + vd * bytesPerLine);
		    for (int us = -1, ud = 0; ud < buffWidth; )
		    {
			if ((us+1)*buffWidth <= ud*image.width())
			    ++us;
			q[ud++] = _colormap.getUnderlayPixel(p[uu<S>(us)],
							     us, vs);
		    }
		    vdp = vd;	// keep vd for the following copy of scanline
		}
		else
		    memcpy(buff + vd *bytesPerLine,
			   buff + vdp*bytesPerLine, bytesPerLine);
	}
	else					// shrink
	{
	    for (int vs = 0, vd = 0; vd < buffHeight; ++vs)
		if (vd*image.height() <= vs*buffHeight)
		{
		    const S*	p = image[vs].data();
		    T*		q = (T*)(buff + (vd++) * bytesPerLine);
		    for (int us = 0, ud = 0; ud < buffWidth; ++us)
			if (ud*image.width() <= us*buffWidth)
			    q[ud++] = _colormap.getUnderlayPixel(p[uu<S>(us)],
								 us, vs);
		}
	}
    }
}

/*
 *  Public member functions
 */
DC&
XDC::setLayer(Layer layer)
{
    DC::setLayer(layer);
    XSetPlaneMask(_colormap.display(), _gc, (getLayer() == DC::UNDERLAY ?
					     _colormap.getUnderlayPlanes() :
					     _colormap.getOverlayPlanes()));
    return *this;
}

DC&
XDC::setThickness(u_int thickness)
{
    XGCValues       values;
    XGetGCValues(_colormap.display(), _gc, GCLineWidth, &values);
    values.line_width = thickness;
    XChangeGC(_colormap.display(), _gc, GCLineWidth, &values);
    
    return *this;
}

DC&
XDC::setForeground(const BGR& fg)
{
    XSetForeground(_colormap.display(), _gc,
		   (getLayer() == DC::UNDERLAY ?
		    _colormap.getUnderlayPixel(fg, 0, 0) :
		    _colormap.getOverlayPixel(fg)));
    return *this;
}

DC&
XDC::setBackground(const BGR& bg)
{
    XSetBackground(_colormap.display(), _gc,
		   (getLayer() == DC::UNDERLAY ?
		    _colormap.getUnderlayPixel(bg, 0, 0) :
		    _colormap.getOverlayPixel(bg)));
    return *this;
}

DC&
XDC::setForeground(u_int fg)
{
    XSetForeground(_colormap.display(), _gc, (getLayer() == DC::UNDERLAY ?
					      _colormap.getUnderlayPixel(fg) :
					      _colormap.getOverlayPixel(fg)));
    return *this;
}

DC&
XDC::setBackground(u_int bg)
{
    XSetBackground(_colormap.display(), _gc, (getLayer() == DC::UNDERLAY ?
					      _colormap.getUnderlayPixel(bg) :
					      _colormap.getOverlayPixel(bg)));
    return *this;
}

DC&
XDC::setSaturation(u_int saturation)
{
    _colormap.setSaturation(saturation);
    return *this;
}

DC&
XDC::setSaturationF(float saturation)
{
    _colormap.setSaturationF(saturation);
    return *this;
}

DC&
XDC::clear()
{
    if (getLayer() == DC::UNDERLAY)
    {
	XGCValues       values;
	XGetGCValues(_colormap.display(), _gc, GCForeground, &values);
	XSetForeground(_colormap.display(), _gc,
		       _colormap.getUnderlayPixel(u_char(0), 0, 0));
	XFillRectangle(_colormap.display(), drawable(), _gc,
		       0, 0, realWidth(), realHeight());
	XChangeGC(_colormap.display(), _gc, GCForeground, &values);
    }
    else
    {
	XGCValues       values;
	XGetGCValues(_colormap.display(), _gc, GCFunction, &values);
	XSetFunction(_colormap.display(), _gc, GXclear);
	XFillRectangle(_colormap.display(), drawable(), _gc,
		       0, 0, realWidth(), realHeight());
	XChangeGC(_colormap.display(), _gc, GCFunction, &values);
    }
  // XIL, XGL, OpenGL など、Xlib を介さずに direct に graphic hardware 
  // に描画する（direct 描画）API を使う場合は、Xlib の実行と direct 
  // 描画の実行の順序性が崩れることがある.  したがって、window の 
  // clear が direct 描画の後になってしまうことを防ぐため、ここで 
  // XSync() を呼ぶ。
    XSync(_colormap.display(), False);
    return *this;
}

DC&
XDC::sync()
{
    XSync(_colormap.display(), False);
    return *this;
}

DC&
XDC::operator <<(const Point2<int>& p)
{
    switch (getPointStyle())
    {
      default:
      case DOT:
      {
	u_int	w = (mul() > div() ? mul() / div() : 1);
	XFillRectangle(_colormap.display(), drawable(), _gc,
		       log2devR(p[0] + offset()[0]),
		       log2devR(p[1] + offset()[1]), w, w);
      }
	break;
	
      case CROSS:
	XDrawLine(_colormap.display(), drawable(), _gc,
		  log2devU(p[0] - PRADIUS), log2devV(p[1] - PRADIUS),
		  log2devU(p[0] + PRADIUS), log2devV(p[1] + PRADIUS));
	XDrawLine(_colormap.display(), drawable(), _gc,
		  log2devU(p[0] - PRADIUS), log2devV(p[1] + PRADIUS),
		  log2devU(p[0] + PRADIUS), log2devV(p[1] - PRADIUS));
        break;
      
      case CIRCLE:
	XDrawArc(_colormap.display(), drawable(), _gc, 
		 log2devU(p[0] - PRADIUS),  log2devV(p[1] - PRADIUS),
		 log2devR(2*(int)PRADIUS + 1), log2devR(2*(int)PRADIUS + 1),
		 0, 360 * 64);
	break;
    }
    
    return *this;
}

DC&
XDC::operator <<(const LineP2f& l)
{
    return operator <<(LineP2d(l));
}

DC&
XDC::operator <<(const LineP2d& l)
{
    if (fabs(l[0]) == 0.0 && fabs(l[1]) == 0.0)
	return *this;
    
    Point2<int>	p, q;

    if (fabs(l[0]) > fabs(l[1]))
    {
	p[0] = int(0.5 + (offset()[1] * l[1] - l[2]) / l[0]);
	p[1] = -offset()[1];
	q[0] = int(0.5 + ((offset()[1] - (int)height()) * l[1] - l[2]) / l[0]);
	q[1] = -offset()[1] + (int)height();
    }
    else
    {
	p[0] = -offset()[0];
	p[1] = int(0.5 + (offset()[0] * l[0] - l[2]) / l[1]);
	q[0] = -offset()[0] + (int)width();
	q[1] = int(0.5 + ((offset()[0] - (int)width()) * l[0] - l[2]) / l[1]);
    }

    XDrawLine(_colormap.display(), drawable(), _gc,
	      log2devU(p[0]), log2devV(p[1]), log2devU(q[0]), log2devV(q[1]));

    return *this;
}

DC&
XDC::operator <<(const Image<u_char>& image)
{
    setGraymap();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<s_char>& image)
{
    setSignedmap();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<short>& image)
{
    setSignedmap();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<float>& image)
{
    setSignedmap();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<BGR>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<ABGR>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<BGRA>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<RGB>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<RGBA>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<ARGB>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<YUV444>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<YUV422>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<YUYV422>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::operator <<(const Image<YUV411>& image)
{
    setColorcube();
    createXImage(image);
    putXImage();

    return *this;
}

DC&
XDC::drawLine(const Point2<int>& p, const Point2<int>& q)
{
    XDrawLine(_colormap.display(), drawable(), _gc,
	      log2devU(p[0]), log2devV(p[1]), log2devU(q[0]), log2devU(q[1]));

    return *this;
}

DC&
XDC::draw(const char* s, int u, int v)
{
    XDrawString(_colormap.display(), drawable(), _gc, log2devU(u), log2devV(v),
		s, strlen(s));

    return *this;
}

void
XDC::dump(std::ostream& out) const
{
    XImage*	ximage = XGetImage(_colormap.display(), drawable(),
				   0, 0, deviceWidth(), deviceHeight(),
				   ~0L, ZPixmap);
    if (ximage == 0)
	throw std::runtime_error("TU::v::XDC::dump: Unable to get image!");

  // Write out header information.
    XWDFileHeader	header;
    header.header_size	    = (CARD32)htonl(sizeof(header));
    header.file_version	    = (CARD32)htonl(XWD_FILE_VERSION);
    header.pixmap_format    = (CARD32)htonl(ZPixmap);
    header.pixmap_depth	    = (CARD32)htonl(ximage->depth);
    header.pixmap_width	    = (CARD32)htonl(ximage->width);
    header.pixmap_height    = (CARD32)htonl(ximage->height);
    header.xoffset	    = (CARD32)htonl(ximage->xoffset);
    header.byte_order	    = (CARD32)htonl(ximage->byte_order);
    header.bitmap_unit	    = (CARD32)htonl(ximage->bitmap_unit);
    header.bitmap_bit_order = (CARD32)htonl(ximage->bitmap_bit_order);
    header.bitmap_pad	    = (CARD32)htonl(ximage->bitmap_pad);
    header.bits_per_pixel   = (CARD32)htonl(ximage->bits_per_pixel);
    header.bytes_per_line   = (CARD32)htonl(ximage->bytes_per_line);
    header.visual_class	    = (CARD32)htonl(_colormap.vinfo().c_class);
    header.red_mask	    = (CARD32)htonl(_colormap.vinfo().red_mask);
    header.green_mask	    = (CARD32)htonl(_colormap.vinfo().green_mask);
    header.blue_mask	    = (CARD32)htonl(_colormap.vinfo().blue_mask);
    header.bits_per_rgb	    = (CARD32)htonl(_colormap.vinfo().bits_per_rgb);
    header.colormap_entries = (CARD32)htonl(_colormap.vinfo().colormap_size);
    header.ncolors	    = (CARD32)htonl(_colormap.npixels());
    header.window_width	    = (CARD32)htonl(ximage->width);
    header.window_height    = (CARD32)htonl(ximage->height);
    header.window_x	    = htonl(0);
    header.window_y	    = htonl(0);
    header.window_bdrwidth  = htonl(0);
    if (!out.write((char *)&header, sizeof(header)))
	throw std::runtime_error("TU::v::XDC::dump: failed to write header!");

  // Create a XColor array for pixel values of 0, 1, 2, .....
    Array<XColor>	xcolors(_colormap.npixels());
    switch (_colormap.vinfo().c_class)
    {
      case TrueColor:
      case DirectColor:
      {
	u_long	red = 0, green = 0, blue = 0,
		red1 = lowbit(_colormap.vinfo().red_mask),
		green1 = lowbit(_colormap.vinfo().green_mask),
		blue1 = lowbit(_colormap.vinfo().blue_mask);
	for (int i = 0; i < xcolors.dim(); ++i)
	{
	    xcolors[i].pixel = red | green | blue;
	    xcolors[i].pad = 0;
	    if ((red += red1) > _colormap.vinfo().red_mask)
		red = 0;
	    if ((green += green1) > _colormap.vinfo().green_mask)
		green = 0;
	    if ((blue += blue1) > _colormap.vinfo().blue_mask)
		blue = 0;
	}
      }
	break;
      
      default:
      {
	for (int i = 0; i < xcolors.dim(); ++i)
	{
	    xcolors[i].pixel = i;
	    xcolors[i].pad = 0;
	}
      }
      break;
    }
    XQueryColors(_colormap.display(), _colormap,
		 xcolors.data(), xcolors.dim());

  // Convert XColor entries to XWDColor format and save them.
    for (int i = 0; i < xcolors.dim(); ++i)
    {
	XWDColor	xwdcolor;
	xwdcolor.pixel = htonl(xcolors[i].pixel);
	xwdcolor.red   = htons(xcolors[i].red);
	xwdcolor.green = htons(xcolors[i].green);
	xwdcolor.blue  = htons(xcolors[i].blue);
	xwdcolor.flags = xcolors[i].flags;
	if (!out.write((char*)&xwdcolor, sizeof(xwdcolor)))
	    throw std::runtime_error("TU::v::XDC::dump: Failed to save a colormap endty!");
    }

  // Write out the image data.
    if (!out.write(ximage->data, ximage->bytes_per_line * ximage->height))
	throw std::runtime_error("TU::v::XDC::dump: faild to save image data!");

    XDestroyImage(ximage);
}

/*
 *  Protected member functions
 */
XDC::XDC(u_int width, u_int height, u_int mul, u_int div,
	 Colormap& colormap, GC gc)
    :DC(width, height, mul, div),
     _colormap(colormap), _gc(gc), _buff(0), _ximage(0)
{
    setLayer(getLayer());	// Set plane mask of GC correctly.
    if (_colormap.getMode() == Colormap::RGBColor)
	_colormap.setColorcube();
}

XDC::~XDC()
{
    if (_ximage != 0)
    {
	_ximage->data = 0;
	XDestroyImage(_ximage);
    }
}

XDC&
XDC::setGraymap()
{
    _colormap.setGraymap();
    setLayer(getLayer());
    return *this;
}

XDC&
XDC::setSignedmap()
{
    _colormap.setSignedmap();
    setLayer(getLayer());
    return *this;
}

XDC&
XDC::setColorcube()
{
    _colormap.setColorcube();
    if (_colormap.vinfo().c_class == PseudoColor)
	XSetPlaneMask(_colormap.display(), _gc, ~0);
    return *this;
}

void
XDC::allocateXImage(int buffWidth, int buffHeight)
{
    if (_ximage != 0)
    {
	_ximage->data = 0;
	XDestroyImage(_ximage);
    }

    _ximage = XCreateImage(_colormap.display(), _colormap.vinfo().visual,
			   _colormap.vinfo().depth, ZPixmap, 0,
			   0, buffWidth, buffHeight, 32, 0);
    if (_ximage == 0)
	throw std::runtime_error("TU::v::XDC::allocateXImage(): XCreateImage failed!!");
    _buff.resize(_ximage->bytes_per_line * _ximage->height);
    _ximage->data = _buff.data();
}

void
XDC::putXImage() const
{
    XPutImage(_colormap.display(), drawable(), _gc, _ximage,
	      0, 0, log2devR(offset()[0]), log2devR(offset()[1]),
	      _ximage->width, _ximage->height);
}

u_int
XDC::getThickness() const
{
    XGCValues       values;
    XGetGCValues(_colormap.display(), _gc, GCLineWidth, &values);
    return values.line_width;
}

u_int
XDC::realWidth() const
{
    return deviceWidth();
}

u_int
XDC::realHeight() const
{
    return deviceHeight();
}

}
}
