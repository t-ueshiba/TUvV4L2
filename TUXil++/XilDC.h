/*
 *  $Id: XilDC.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifndef __TUvXilDC_h
#define __TUvXilDC_h

#include <xil/xil.h>
#include "TU/v/CanvasPaneDC.h"

namespace TU
{
/************************************************************************
*  class XilObject							*
************************************************************************/
class XilObject
{
  protected:
    XilObject()					;
    ~XilObject()					;
    XilObject(const XilObject&)			{++_nobjects;}
    XilObject&	operator =(const XilObject&)	{return *this;}
    
    static XilSystemState	xilstate()		{return _xilstate;}
    
  private:
    static XilSystemState	_xilstate;
    static u_int		_nobjects;
};

/************************************************************************
*  class XilImage<T>							*
************************************************************************/
template <class T>
class XilImage : public Image<T>, public XilObject
{
  public:
    XilImage(u_int w=1, u_int h=1)
        :Image<T>(w, h), XilObject(),
         _xilimage(xil_create(xilstate(), width(), height(), xil_nbands(),
			      xil_type()))		{set_storage();}
    XilImage(T* p, u_int w, u_int h)
	:Image<T>(p, w, h), XilObject(),
	 _xilimage(xil_create(xilstate(), width(), height(), xil_nbands(),
			      xil_type()))		{set_storage();}
    XilImage(const XilImage& image)
	:Image<T>(image), XilObject(image),
	 _xilimage(xil_create(xilstate(), width(), height(), xil_nbands(),
			      xil_type()))		{set_storage();}
    XilImage(const Image<T>& image)
	:Image<T>(image), XilObject(),
	 _xilimage(xil_create(xilstate(), width(), height(), xil_nbands(),
			      xil_type()))		{set_storage();}
    ~XilImage()					;

    XilImage&		operator =(const XilImage&)	;
    XilImage&		operator =(const Image<T>&)	;
    
			operator ::XilImage()	const	{return _xilimage;}
    void		resize(u_int, u_int)		;

  protected:
    void		resize(T*, u_int, u_int)	;
    
  private:
    static XilDataType	xil_type()			;
    static u_int	xil_nbands()			;
    void		set_storage()			;

    ::XilImage		_xilimage;
};

template <class T> inline
XilImage<T>::~XilImage()
{
    xil_destroy(_xilimage);
}

template <class T> inline XilImage<T>&
XilImage<T>::operator =(const XilImage<T>& image)
{
    Image<T>::operator =(image);
    return *this;
}

template <class T> inline XilImage<T>&
XilImage<T>::operator =(const Image<T>& image)
{
    Image<T>::operator =(image);
    return *this;
}

template <class T> inline void
XilImage<T>::resize(u_int h, u_int w)
{
    if (w != width() || h != height())
    {
	xil_destroy(_xilimage);
	_xilimage = xil_create(xilstate(), w, h, xil_nbands(), xil_type());
    }
    Image<T>::resize(h, w);
    set_storage();
}

template <class T> inline void
XilImage<T>::resize(T* p, u_int h, u_int w)
{
    if (w != width() || h != height())
    {
	xil_destroy(_xilimage);
	_xilimage = xil_create(xilstate(), w, h, xil_nbands(), xil_type());
    }
    Image<T>::resize(p, h, w);
    set_storage();
}

template <class T> inline void
XilImage<T>::set_storage()
{
    XilMemoryStorage	storage;
    const u_int		s = (height() > 1 ? &(*this)[1][0] - &(*this)[0][0] :
			     height() > 0 ? width() : 0);
    
    storage.byte.data		 = (u_char*)(T*)*this;
    storage.byte.scanline_stride = sizeof(T) * s;
    storage.byte.pixel_stride	 = xil_nbands();
    xil_export(_xilimage);
    xil_set_memory_storage(_xilimage, &storage);
    xil_set_storage_movement(_xilimage, XIL_KEEP_STATIONARY);
    xil_import(_xilimage, TRUE);
}

template <class T> inline XilDataType
XilImage<T>::xil_type()
{
    return XIL_BYTE;
}

template <class T> inline u_int
XilImage<T>::xil_nbands()
{
    return 1;
}

/*
 *  Specialization
 */
template <> inline u_int
XilImage<BGR>::xil_nbands()
{
    return 3;
}

template <> inline u_int
XilImage<ABGR>::xil_nbands()
{
    return 4;
}

template <> inline u_int
XilImage<RGB>::xil_nbands()
{
    return 3;
}

template <> inline u_int
XilImage<RGBA>::xil_nbands()
{
    return 4;
}

template <> inline XilDataType
XilImage<short>::xil_type()
{
    return XIL_SHORT;
}

namespace v
{
/************************************************************************
*  class XilDC								*
************************************************************************/
class XilDC : virtual public CanvasPaneDC, public XilObject
{
  public:
    XilDC(CanvasPane& parentCanvasPane,
	     u_int width=0, u_int height=0)				;
    virtual		~XilDC()					;
    
    virtual DC&		setSize(u_int width, u_int height,
				u_int mul,   u_int div)			;
    virtual DC&		setOffset(int u0, int v0)			;
#ifdef TUXilPP_UsePlaneMasks
    virtual DC&		setLayer(Layer layer)				;
#endif
    virtual DC&		setSaturation(u_int saturation)			;
    
    virtual DC&		operator <<(const Point2<int>& p)		;
    virtual DC&		operator <<(const LineP2<double>& l)		;
    virtual DC&		operator <<(const Image<u_char>& image)		;
    virtual DC&		operator <<(const Image<s_char>& image)		;
    virtual DC&		operator <<(const Image<short>&  image)		;
    virtual DC&		operator <<(const Image<BGR>&  image)		;
    virtual DC&		operator <<(const Image<ABGR>& image)		;
    XilDC&		operator <<(::XilImage)				;
    XilDC&		operator <<(const XilImage<u_char>& image)	;
    XilDC&		operator <<(const XilImage<s_char>& image)	;
    XilDC&		operator <<(const XilImage<BGR>&  image)	;
    XilDC&		operator <<(const XilImage<ABGR>& image)	;
    
    XilDC&		useGraymap()					;
    XilDC&		useSignedmap()					;
    
  protected:
    virtual void	initializeGraphics()				;

    virtual XDC&	setGraymap()					;
    virtual XDC&	setSignedmap()					;

  private:
    ::XilImage		_xilDisplay;	// display window.
    ::XilImage		_xilTmpImage;	// temporary image for scaling.
#ifdef TUXilPP_UsePlaneMasks
    ::XilImage		_xilTmpImage2;	// temporary image for masking.
#endif
    XilLookup		_grayLookup;	// lookup table for gray images.
    XilLookup		_lookups[3];	// for combined lookup table.
    XilLookup		_colorLookup;	// lookup table for color images.
    XilDitherMask const	_ditherMask;	// dither mask for 8-bit _xilDisplay.
#ifdef TUXilPP_UsePlaneMasks
    u_int		_planeMasks[3];	// plane masks for each channel.
#endif
    u_int		_useSignedmap;	// use signed map for 1-band images.
};

inline XilDC&
XilDC::operator <<(const XilImage<u_char>& image)
{
    return *this << (::XilImage)image;
}

inline XilDC&
XilDC::operator <<(const XilImage<s_char>& image)
{
    return *this << (::XilImage)image;
}

inline XilDC&
XilDC::operator <<(const XilImage<BGR>& image)
{
    return *this << (::XilImage)image;
}

inline XilDC&
XilDC::operator <<(const XilImage<ABGR>& image)
{
    return *this << (::XilImage)image;
}

inline XilDC&
XilDC::useGraymap()
{
    _useSignedmap = 0;
    return *this;
}

inline XilDC&
XilDC::useSignedmap()
{
    _useSignedmap = 1;
    return *this;
}
 
}
}
#endif	/* ! __TUvXilDC_h */
