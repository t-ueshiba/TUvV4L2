/*
 *  $Id: Colormap.h,v 1.2 2002-07-25 02:38:10 ueshiba Exp $
 */
#ifndef __TUvColormap_h
#define __TUvColormap_h

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include "TU/Image++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class Colormap							*
************************************************************************/
class Colormap
{
  public:
    enum	{InheritFromParent = -1};
    enum Mode	{IndexedColor, RGBColor};
    
  public:
    Colormap(Display* display, const XVisualInfo& vinfo)		;
    Colormap(Display* display, const XVisualInfo& vinfo,
	     Mode mode, u_int resolution,
	     u_int underlayCmapDim, u_int overlayDepth,
	     u_int rDim, u_int gDim, u_int bDim)			;
    ~Colormap()								;

  // X stuffs
    Display*		display()				const	;
    const XVisualInfo&	vinfo()					const	;
			operator ::Colormap()			const	;

  // mode stuffs
    Mode		getMode()				const	;
    void		setGraymap()					;
    void		setSignedmap()					;
    void		setColorcube()					;
    u_int		getSaturation()				const	;
    void		setSaturation(u_int saturation)			;
    
  // underlay stuffs
    template <class T>
    u_long		getUnderlayPixel(T val,
					 u_int u, u_int v)	const	;
    u_long		getUnderlayPixel(const YUV422& yuv,
					 u_int u, u_int v)	const	;
    u_long		getUnderlayPixel(const YUV411& yuv,
					 u_int u, u_int v)	const	;
    u_long		getUnderlayPixel(u_int index)		const	;
    BGR			getUnderlayValue(u_int index)		const	;
    void		setUnderlayValue(u_int index, BGR bgr)		;
    Array<u_long>	getUnderlayPixels()			const	;
    u_long		getUnderlayPlanes()			const	;

  // overlay stuffs
    template <class T>
    u_long		getOverlayPixel(T bgr)			const	;
    u_long		getOverlayPixel(u_int index)		const	;
    BGR			getOverlayValue(u_int index)		const	;
    void		setOverlayValue(u_int index, BGR bgr)		;
    Array<u_long>	getOverlayPixels()			const	;
    u_long		getOverlayPlanes()			const	;

  // colorcube stuffs
    u_int		rDim()					const	;
    u_int		gDim()					const	;
    u_int		bDim()					const	;
    u_int		rStride()				const	;
    u_int		gStride()				const	;
    u_int		bStride()				const	;

    u_int		npixels()				const	;
    
  private:
    Colormap(const Colormap&)						;
    Colormap&		operator =(const Colormap&)			;

    void		setGraymapInternal()				;
    void		setSignedmapInternal()				;
    void		setColorcubeInternal()				;
    
  // X stuffs
    Display* const	_display;		// X server
    const XVisualInfo	_vinfo;
    const ::Colormap	_colormap;

  // underlay stuffs
    const u_int		_resolution;
    u_int		_saturation;
    u_long		_underlayTable[256];

  // overlay stuffs
    u_long		_overlayPlanes;		// mask for overlay planes
    u_long		_overlayTable[256];

  // colorcube stuffs
    enum		{DITHERMASK_SIZE = 4};

    const u_int		_colorcubeNPixels;
    const u_int		_gStride, _bStride;
    const double	_rMul, _gMul, _bMul;
    double		_dithermask[DITHERMASK_SIZE][DITHERMASK_SIZE];

  // general stuffs
    enum Map		{Graymap, Signedmap, Colorcube};

    Array2<Array<u_long> >	_pixels;
    const Mode			_mode;
    Map				_map;
};

inline Display*
Colormap::display() const
{
    return _display;
}

inline const XVisualInfo&
Colormap::vinfo() const
{
    return _vinfo;
}

inline
Colormap::operator ::Colormap() const
{
    return _colormap;
}

inline Colormap::Mode
Colormap::getMode() const
{
    return _mode;
}

inline u_int
Colormap::getSaturation() const
{
    return _saturation;
}

template <> inline u_long
Colormap::getUnderlayPixel<u_char>(u_char val, u_int, u_int) const
{
    return _underlayTable[val];
}

template <> inline u_long
Colormap::getUnderlayPixel<s_char>(s_char val, u_int, u_int) const
{
    switch (_vinfo.c_class)
    {
      case TrueColor:
      case DirectColor:
	return (val < 0 ? _underlayTable[val+256] & _vinfo.green_mask :
			  _underlayTable[val]	  & _vinfo.red_mask);
    }
    return (val < 0 ? _underlayTable[val+256] : _underlayTable[val]);
}

template <> inline u_long
Colormap::getUnderlayPixel<short>(short val, u_int, u_int) const
{
    if (val > 127)
	val = 127;
    else if (val < -128)
	val = -128;
    
    switch (_vinfo.c_class)
    {
      case TrueColor:
      case DirectColor:
	return (val < 0 ? _underlayTable[val+256] & _vinfo.green_mask :
			  _underlayTable[val]	  & _vinfo.red_mask);
    }
    return (val < 0 ? _underlayTable[val+256] : _underlayTable[val]);
}

template <class T> inline u_long
Colormap::getUnderlayPixel(T val, u_int u, u_int v) const
{
    switch (_vinfo.c_class)
    {
      case PseudoColor:
      {
	double		r = val.r * _rMul, g = val.g * _gMul,
			b = val.b * _bMul;
	const u_int	rIndex = u_int(r), gIndex = u_int(g),
			bIndex = u_int(b);
	r -= rIndex;
	g -= gIndex;
	b -= bIndex;
	const double mask = _dithermask[v%DITHERMASK_SIZE][u%DITHERMASK_SIZE];
	return _pixels[0][(r > mask ? rIndex + 1 : rIndex) +
			  (g > mask ? gIndex + 1 : gIndex) * _gStride +
			  (b > mask ? bIndex + 1 : bIndex) * _bStride];
      }
      case TrueColor:
      case DirectColor:
	return (_underlayTable[val.r] & _vinfo.red_mask)   |
	       (_underlayTable[val.g] & _vinfo.green_mask) |
	       (_underlayTable[val.b] & _vinfo.blue_mask);
    }
    return getUnderlayPixel((u_char)val, u, v);
}

template <> inline u_long
Colormap::getUnderlayPixel<YUV444>(YUV444 yuv, u_int u, u_int v) const
{
    return getUnderlayPixel(fromYUV<BGR>(yuv.y, yuv.u, yuv.v), u, v);
}

inline u_long
Colormap::getUnderlayPixel(const YUV422& yuv, u_int u, u_int v) const
{
    if (u & 0x1)
    {
	u_char	uu = (&yuv)[-1].x;
	return getUnderlayPixel(fromYUV<BGR>(yuv.y, uu, yuv.x), u, v);
    }
    else
    {
	u_char	vv = (&yuv)[1].x;
	return getUnderlayPixel(fromYUV<BGR>(yuv.y, yuv.x, vv), u, v);
    }
}

inline u_long
Colormap::getUnderlayPixel(const YUV411& yuv, u_int u, u_int v) const
{
    switch (u % 4)
    {
      case 0:
      {
	u_char	vv = (&yuv)[1].x;
	return getUnderlayPixel(fromYUV<BGR>(yuv.y0, yuv.x, vv), u, v);
      }
      case 1:
      {
	u_char	vv = (&yuv)[1].x;
	return getUnderlayPixel(fromYUV<BGR>(yuv.y1, yuv.x, vv),
				u, v);
      }
      case 2:
      {
	u_char	uu = (&yuv)[-1].x;
	return getUnderlayPixel(fromYUV<BGR>(yuv.y0, uu, yuv.x), u, v);
      }
    }
    u_char	uu = (&yuv)[-1].x;
    return getUnderlayPixel(fromYUV<BGR>(yuv.y1, uu, yuv.x), u, v);
}

inline u_long
Colormap::getUnderlayPlanes() const
{
    return ~_overlayPlanes;
}

template <class T> inline u_long
Colormap::getOverlayPixel(T bgr) const
{
    return (_overlayTable[bgr.r] & _vinfo.red_mask)   |
	   (_overlayTable[bgr.g] & _vinfo.green_mask) |
	   (_overlayTable[bgr.b] & _vinfo.blue_mask);
}

inline u_long
Colormap::getOverlayPlanes() const
{
    return _overlayPlanes;
}

inline u_int
Colormap::rDim() const
{
    return _gStride;
}

inline u_int
Colormap::gDim() const
{
    return (_gStride != 0 ? _bStride / _gStride : 0);
}

inline u_int
Colormap::bDim() const
{
    return (_bStride != 0 ? _colorcubeNPixels / _bStride : 0);
}

inline u_int
Colormap::rStride() const
{
    return 1;
}

inline u_int
Colormap::gStride() const
{
    return _gStride;
}

inline u_int
Colormap::bStride() const
{
    return _bStride;
}

inline u_int
Colormap::npixels() const
{
    return _pixels.nrow() * _pixels.ncol();
}

}
}
#endif	/* !__TUvColormap_h */
