/*
 *  $Id: SDLOglDC.h,v 1.2 2008-09-09 01:42:19 ueshiba Exp $
 */
#ifndef __TUvSDLOglDC_h
#define __TUvSDLOglDC_h

#include "TU/v/DC3.h"
#include <SDL/SDL.h>
#include <SDL/SDL_opengl.h>

namespace TU
{
/************************************************************************
*  class SDLOglDC							*
************************************************************************/
class SDLOglDC : public v::DC3
{
  public:
    SDLOglDC(u_int width, u_int height, u_int depth, bool fullScreen)	;
    virtual ~SDLOglDC()							;

    u_int		width()			const	{return _screen->w;}
    u_int		height()		const	{return _screen->h;}
    
    virtual v::DC3&	setInternal(int	   u0,	 int	v0,
				    double ku,	 double kv,
				    double near, double far=0.0)	;
    virtual v::DC3&	setExternal(const Point3d& t,
				    const Matrix33d& Rt)		;
    virtual const v::DC3&
			getInternal(int&    u0,	  int&	  v0,
				    double& ku,	  double& kv,
				    double& near, double& far)	const	;
    virtual const v::DC3&
			getExternal(Point3d& t, Matrix33d& Rt)	const	;
    virtual v::DC3&	translate(double d)				;
    virtual v::DC3&	rotate(double angle)				;

    void		swapBuffers()				const	;
    void		toggleFullScreen()				;

    virtual bool	callback(const SDL_Event& event)		;

  private:
    SDLOglDC&		setViewport()					;
    
    SDL_Surface* const	_screen;
    int			_scale;
    bool		_fullScreen;
};

inline void
SDLOglDC::swapBuffers() const
{
    SDL_GL_SwapBuffers();
}
    
inline void
SDLOglDC::toggleFullScreen()
{
    if (SDL_WM_ToggleFullScreen(_screen))
	_fullScreen = !_fullScreen;
}
    
}
#endif	// !__TUvSDLOglDC_h
