/*
 *  $Id: XglDC.h,v 1.2 2002-07-25 02:38:07 ueshiba Exp $
 */
#ifndef __TUvXglDC_h
#define __TUvXglDC_h

#include <xgl/xgl.h>
#include <xil/xil.h>
#include "TU/v/CanvasPaneDC3.h"

namespace TU
{
/************************************************************************
*  class XglObject							*
************************************************************************/
class XglObject
{
  protected:
    XglObject()						;
    ~XglObject()					;
    XglObject(const XglObject&)				{++_nobjects;}
    XglObject&	operator =(const XglObject&)		{return *this;}

    static Xgl_sys_state	xglstate()		{return _xglstate;}
    
  private:
    static Xgl_sys_state	_xglstate;
    static u_int		_nobjects;
};

namespace v
{
/************************************************************************
*  class XglDC								*
************************************************************************/
class XglDC : public CanvasPaneDC3, public XglObject
{
  public:
    XglDC(CanvasPane& parentCanvasPane,
	     u_int width=0, u_int height=0)				;
    virtual		~XglDC()					;
    
			operator Xgl_object()			const	;
    Xgl_object		pcache()				const	;
    void		emptyPcache()					;
    
    virtual DC&		setSize(u_int width, u_int height,
				u_int mul,   u_int div)			;
    virtual DC&		setLayer(Layer layer)				;
    virtual DC3&	setInternal(int	   u0,	 int	v0,
				    double ku,	 double kv,
				    double near, double far)		;
    virtual DC3&	setExternal(const Vector<double>& t,
				    const Matrix<double>& Rt)		;
    virtual const DC3&	getInternal(int&    u0,	  int&	  v0,
				    double& ku,	  double& kv,
				    double& near, double& far)	const	;
    virtual const DC3&	getExternal(Vector<double>& t,
				    Matrix<double>& Rt)	const	;
    virtual DC3&	translate(double d)				;
    virtual DC3&	rotate(double angle)				;

    XglDC&		clearXgl()					;
    XglDC&		syncXgl()					;

  protected:
    virtual void	initializeGraphics()				;

  private:
    XglDC&		setViewTransform()				;
    XglDC&		setViewport()					;
    
    Xgl_win_ras		_xglwin;
    Xgl_ctx		_xglctx;
    int			_dummy;
    Xgl_trans	const	_wc_to_cc;
    Xgl_trans	const	_cc_to_vdc;
    Xgl_object		_pcache;
};

inline
XglDC::operator Xgl_object() const
{
    return _xglctx;
}

inline Xgl_object
XglDC::pcache() const
{
    return _pcache;
}

inline void
XglDC::emptyPcache()
{
    xgl_object_destroy(_pcache);
    _pcache = xgl_object_create(xglstate(), XGL_PCACHE, NULL, NULL);
    xgl_object_set(_pcache, XGL_PCACHE_CONTEXT, _xglctx, NULL);
}

/************************************************************************
*  Manipulators								*
************************************************************************/
extern XglDC&	clearXgl(XglDC&);
extern XglDC&	syncXgl(XglDC&);

template <class S> inline S&
operator <<(S& s, XglDC& (*f)(XglDC&))
{
    (*f)(s);
    return s;
}
 
}
}
#endif	/* !__TUvXglDC_h */
