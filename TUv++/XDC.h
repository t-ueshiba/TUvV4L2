/*
 *  $Id: XDC.h,v 1.4 2007-02-06 23:31:02 ueshiba Exp $
 */
#ifndef __TUvXDC_h
#define __TUvXDC_h

#include "TU/v/DC.h"
#include "TU/v/Colormap.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class XDC								*
************************************************************************/
class XDC : public DC
{
  public:
    struct BPixel16
    {
	BPixel16&	operator =(u_long pixel)	{_p[0] = pixel >> 8;
							 _p[1] = pixel;
							 return *this;}
	
	u_char		_p[2];
    };
    struct LPixel16
    {
	LPixel16&	operator =(u_long pixel)	{_p[0] = pixel;
							 _p[1] = pixel >> 8;
							 return *this;}
	
	u_char		_p[2];
    };
    struct BPixel24
    {
	BPixel24&	operator =(u_long pixel)	{_p[0] = pixel >> 16;
							 _p[1] = pixel >>  8;
							 _p[2] = pixel;
							 return *this;}
	
	u_char		_p[3];
    };
    struct LPixel24
    {
	LPixel24&	operator =(u_long pixel)	{_p[0] = pixel;
							 _p[1] = pixel >>  8;
							 _p[2] = pixel >> 16;
							 return *this;}
	
	u_char		_p[3];
    };
    struct BPixel32
    {
	BPixel32&	operator =(u_long pixel)	{_p[1] = pixel >> 16;
							 _p[2] = pixel >>  8;
							 _p[3] = pixel;
							 return *this;}
	
	u_char		_p[4];
    };
    struct LPixel32
    {
	LPixel32&	operator =(u_long pixel)	{_p[0] = pixel;
							 _p[1] = pixel >>  8;
							 _p[2] = pixel >> 16;
							 return *this;}
	
	u_char		_p[4];
    };
    
  public:
    virtual DC&		setLayer(Layer layer)				;
    virtual DC&		setThickness(u_int thickness)			;
    virtual DC&		setForeground(const BGR& fg)			;
    virtual DC&		setBackground(const BGR& bg)			;
    virtual DC&		setForeground(u_int fg)				;
    virtual DC&		setBackground(u_int bg)				;
    virtual DC&		setSaturation(u_int saturation)			;
    
    virtual DC&		clear()						;
    virtual DC&		sync()						;

    using		DC::operator <<;
    virtual DC&		operator <<(const Point2<int>& p)		;
    virtual DC&		operator <<(const LineP2<double>& p)		;
    virtual DC&		operator <<(const Image<u_char>& image)		;
    virtual DC&		operator <<(const Image<s_char>& image)		;
    virtual DC&		operator <<(const Image<short>& image)		;
    virtual DC&		operator <<(const Image<BGR>&  image)		;
    virtual DC&		operator <<(const Image<ABGR>& image)		;
    virtual DC&		operator <<(const Image<RGB>&  image)		;
    virtual DC&		operator <<(const Image<RGBA>& image)		;
    virtual DC&		operator <<(const Image<YUV444>& image)		;
    virtual DC&		operator <<(const Image<YUV422>& image)		;
    virtual DC&		operator <<(const Image<YUV411>& image)		;
    virtual DC&		draw(const char* s, int u, int v)		;

    void		dump(std::ostream& out)			const	;

  protected:
    XDC(u_int width, u_int height, Colormap& colormap, GC gc)		;
    virtual		~XDC()						;

    virtual Drawable	drawable()				const	= 0;
    const Colormap&	colormap()				const	;
    GC			gc()					const	;
    
    virtual XDC&	setGraymap()					;
    virtual XDC&	setSignedmap()					;
    virtual XDC&	setColorcube()					;
    virtual void	allocateXImage(int buffWidth, int buffHeight)	;
    virtual void	putXImage()				const	;
    
    virtual u_int	getThickness()				const	;

  private:
    virtual u_int	realWidth()				const	;
    virtual u_int	realHeight()				const	;
    template <class S>
    void		createXImage(const Image<S>& image)		;
    template <class S, class T>
    void		fillBuff(const Image<S>& image)			;
    
    Colormap&		_colormap;
    GC			_gc;
    Array<char>		_buff;

  protected:
    XImage*		_ximage;
};

inline const Colormap&	XDC::colormap()		const	{return _colormap;}
inline GC		XDC::gc()		const	{return _gc;}

}
}
#endif	// !__TUvXDC_h
