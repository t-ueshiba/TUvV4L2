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
#ifndef __TU_V_COLORMAP_H
#define __TU_V_COLORMAP_H

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include "TU/Image++.h"
#include <stdexcept>
#include <algorithm>
#include <limits.h>

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
	     Mode mode, size_t resolution,
	     size_t underlayCmapDim, size_t overlayDepth,
	     size_t rDim, size_t gDim, size_t bDim)			;
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
    size_t		getSaturation()				const	;
    void		setSaturation(size_t saturation)		;
    float		getSaturationF()			const	;
    void		setSaturationF(float saturation)		;
    
  // underlay stuffs
    template <class T>
    u_long		getUnderlayPixel(T val,
					 size_t u, size_t v)	const	;
    u_long		getUnderlayPixel(const YUV422& yuv,
					 size_t u, size_t v)	const	;
    u_long		getUnderlayPixel(const YUYV422& yuv,
					 size_t u, size_t v)	const	;
    u_long		getUnderlayPixel(const YUV411& yuv,
					 size_t u, size_t v)	const	;
    u_long		getUnderlayPixel(size_t index)		const	;
    BGR			getUnderlayValue(size_t index)		const	;
    void		setUnderlayValue(size_t index, BGR bgr)		;
    Array<u_long>	getUnderlayPixels()			const	;
    u_long		getUnderlayPlanes()			const	;

  // overlay stuffs
    template <class T>
    u_long		getOverlayPixel(T bgr)			const	;
    u_long		getOverlayPixel(size_t index)		const	;
    BGR			getOverlayValue(size_t index)		const	;
    void		setOverlayValue(size_t index, BGR bgr)		;
    Array<u_long>	getOverlayPixels()			const	;
    u_long		getOverlayPlanes()			const	;

  // colorcube stuffs
    size_t		rDim()					const	;
    size_t		gDim()					const	;
    size_t		bDim()					const	;
    size_t		rStride()				const	;
    size_t		gStride()				const	;
    size_t		bStride()				const	;

    size_t		npixels()				const	;
    
  private:
    Colormap(const Colormap&)						;
    Colormap&		operator =(const Colormap&)			;

    void		setGraymapInternal()				;
    void		setSignedmapInternal()				;
    void		setColorcubeInternal()				;
    
  // X stuffs
    Display* const	_display;		// X server
    XVisualInfo		_vinfo;
    const ::Colormap	_colormap;

  // underlay stuffs
    enum		{UnderlayTableSize = 65536};
    
    const size_t		_resolution;
    size_t		_saturation;
    float		_gain;
    u_long		_underlayTable[UnderlayTableSize];

  // overlay stuffs
    u_long		_overlayPlanes;		// mask for overlay planes
    u_long		_overlayTable[256];

  // colorcube stuffs
    enum		{DITHERMASK_SIZE = 4};

    const size_t	_colorcubeNPixels;
    const size_t	_gStride, _bStride;
    const float		_rMul, _gMul, _bMul;
    float		_dithermask[DITHERMASK_SIZE][DITHERMASK_SIZE];

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

inline size_t
Colormap::getSaturation() const
{
    return _saturation;
}

inline float
Colormap::getSaturationF() const
{
    return (_vinfo.c_class == PseudoColor ? _resolution / (2.0*_gain)
					  : _resolution / _gain);
}

inline void
Colormap::setSaturationF(float saturation)
{
    if (saturation <= 0.0)
	throw std::out_of_range("TU::v::Colormap::setSaturationF: saturation value must be positive!!");
    _gain = (_vinfo.c_class == PseudoColor ?
	     _gain = _resolution / (2.0*saturation) :
	     _gain = _resolution / saturation);
}

template <> inline u_long
Colormap::getUnderlayPixel<u_char>(u_char val, size_t, size_t) const
{
    return _underlayTable[val];
}

template <> inline u_long
Colormap::getUnderlayPixel<s_char>(s_char val, size_t, size_t) const
{
    return (val < 0 ?
	    _underlayTable[val+UnderlayTableSize] & _vinfo.red_mask :
	    _underlayTable[val]			  & _vinfo.green_mask);
}

template <> inline u_long
Colormap::getUnderlayPixel<short>(short val, size_t, size_t) const
{
    return (val < 0 ?
	    _underlayTable[val+UnderlayTableSize] & _vinfo.red_mask :
	    _underlayTable[val]			  & _vinfo.green_mask);
}

template <> inline u_long
Colormap::getUnderlayPixel<float>(float val, size_t, size_t) const
{
    float	fidx = _gain * val;
    int		idx = (fidx > float(INT_MAX) ? INT_MAX :
		       fidx < float(INT_MIN) ? INT_MIN + 1 : int(fidx + 0.5));
    bool	positive = true;
    if (idx < 0)
    {
	idx = -idx;
	positive = false;
    }
    
    if (_vinfo.c_class == PseudoColor)
    {
	idx = std::min(idx, int(_resolution - 1) / 2);
	return (positive ? _pixels[0][idx] : _pixels[0][_resolution - idx]);
    }
    else
    {
	idx = std::min(idx, int(_resolution - 1));
	return (positive ? _pixels[0][idx] & _vinfo.green_mask
			 : _pixels[0][idx] & _vinfo.red_mask);
    }
}

template <class T> inline u_long
Colormap::getUnderlayPixel(T val, size_t u, size_t v) const
{
    switch (_vinfo.c_class)
    {
      case PseudoColor:
      {
	float		r = val.r * _rMul, g = val.g * _gMul,
			b = val.b * _bMul;
	const size_t	rIndex = size_t(r), gIndex = size_t(g),
			bIndex = size_t(b);
	r -= rIndex;
	g -= gIndex;
	b -= bIndex;
	const float mask = _dithermask[v%DITHERMASK_SIZE][u%DITHERMASK_SIZE];
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
Colormap::getUnderlayPixel<YUV444>(YUV444 yuv, size_t u, size_t v) const
{
    return getUnderlayPixel(BGR(yuv), u, v);
}

inline u_long
Colormap::getUnderlayPixel(const YUV422& yuv, size_t u, size_t v) const
{
    if (u & 0x1)
	return getUnderlayPixel(BGR(YUV444(yuv.y, (&yuv)[-1].x, yuv.x)), u, v);
    else
	return getUnderlayPixel(BGR(YUV444(yuv.y, yuv.x, (&yuv)[1].x)), u, v);
}

inline u_long
Colormap::getUnderlayPixel(const YUYV422& yuv, size_t u, size_t v) const
{
    if (u & 0x1)
	return getUnderlayPixel(BGR(YUV444(yuv.y, (&yuv)[-1].x, yuv.x)), u, v);
    else
	return getUnderlayPixel(BGR(YUV444(yuv.y, yuv.x, (&yuv)[1].x)), u, v);
}

inline u_long
Colormap::getUnderlayPixel(const YUV411& yuv, size_t u, size_t v) const
{
    switch (u % 4)
    {
      case 0:
	return getUnderlayPixel(BGR(YUV444(yuv.y0, yuv.x, (&yuv)[1].x)), u, v);
      case 1:
	return getUnderlayPixel(BGR(YUV444(yuv.y1, yuv.x, (&yuv)[1].x)), u, v);
      case 2:
	return getUnderlayPixel(BGR(YUV444(yuv.y0, (&yuv)[-1].x, yuv.x)), u, v);
    }
    return getUnderlayPixel(BGR(YUV444(yuv.y1, (&yuv)[-1].x, yuv.x)), u, v);
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

inline size_t
Colormap::rDim() const
{
    return _gStride;
}

inline size_t
Colormap::gDim() const
{
    return (_gStride != 0 ? _bStride / _gStride : 0);
}

inline size_t
Colormap::bDim() const
{
    return (_bStride != 0 ? _colorcubeNPixels / _bStride : 0);
}

inline size_t
Colormap::rStride() const
{
    return 1;
}

inline size_t
Colormap::gStride() const
{
    return _gStride;
}

inline size_t
Colormap::bStride() const
{
    return _bStride;
}

inline size_t
Colormap::npixels() const
{
    return _pixels.nrow() * _pixels.ncol();
}

}
}
#endif	// !__TU_V_COLORMAP_H
