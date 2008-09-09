/*
 *  $Id: SDLOglDC.cc,v 1.2 2008-09-09 01:42:18 ueshiba Exp $
 */
#include "TU/SDLOglDC.h"

namespace TU
{
/************************************************************************
*  class SDLOglDC							*
************************************************************************/
SDLOglDC::SDLOglDC(u_int w, u_int h, u_int d, bool fullScreen)
    :v::DC3(v::DC3::X, 128.0),
     _screen(SDL_SetVideoMode(w, h, d,
			      SDL_OPENGL | (fullScreen ? SDL_FULLSCREEN : 0))),
     _scale(0),
     _fullScreen(fullScreen)
{
    if (_screen == 0)
	throw std::runtime_error("TU::v::SDLOglDC::SDLOglDC: Failed to set vide mode!!");

    setViewport();
    setInternal(width() / 2, height() / 2, 800.0, 800.0, 1.0, 1000.0);
    Matrix33d	Rt;
    Rt[0][0] = Rt[2][1] = 1.0;
    Rt[1][2] = -1.0;
    setExternal(Point3d(), Rt);
}

SDLOglDC::~SDLOglDC()
{
}
    
v::DC3&
SDLOglDC::setInternal(int u0, int v0, double ku, double kv,
		      double near, double far)
{
    GLdouble	matrix[4][4];
    matrix[0][0] =  2.0 * ku / width();
    matrix[1][1] = -2.0 * kv / height();
    matrix[2][0] =  2.0 * u0 / width()  - 1.0;
    matrix[2][1] = -2.0 * v0 / height() + 1.0;
    matrix[2][3] =  1.0;
    if (far > near)
    {
	matrix[2][2] = (far + near) / (far - near);
	matrix[3][2] = -2.0 * far * near / (far - near);
    }
    else
    {
	matrix[2][2] =  1.0;
	matrix[3][2] = -2.0 * near;
    }
		   matrix[0][1] = matrix[0][2] = matrix[0][3] =
    matrix[1][0]	        = matrix[1][2] = matrix[1][3] =
    matrix[3][0] = matrix[3][1]		       = matrix[3][3] = 0.0;

    glMatrixMode(GL_PROJECTION);
      glLoadMatrixd(matrix[0]);
    glMatrixMode(GL_MODELVIEW);		// Default mode should be MODELVIEW.
    
    return *this;
}

v::DC3&
SDLOglDC::setExternal(const Point3d& t, const Matrix33d& Rt)
{
  /* set rotation */
    GLdouble	matrix[4][4];
    for (int i = 0; i < 3; ++i)
    {
	for (int j = 0; j < 3; ++j)
	    matrix[i][j] = Rt[j][i];
	matrix[i][3] = matrix[3][i] = 0.0;
    }
    matrix[3][3] = 1.0;
    glLoadMatrixd(matrix[0]);
    
  /* set translation */
    glTranslated(-t[0], -t[1], -t[2]);
    
    return *this;
}

const v::DC3&
SDLOglDC::getInternal(int& u0, int& v0, double& ku, double& kv,
		      double& near, double& far) const
{
    GLdouble	matrix[4][4];
    glGetDoublev(GL_PROJECTION_MATRIX, matrix[0]);
    ku	 =	  matrix[0][0]  * width()  / 2.0;
    kv	 =       -matrix[1][1]  * height() / 2.0;
    u0	 = (1.0 + matrix[2][0]) * width()  / 2.0;
    v0	 = (1.0 - matrix[2][1]) * height() / 2.0;
    near =	 -matrix[3][2] / (1.0 + matrix[2][2]);
    far  = (matrix[2][2] != 1.0 ?
		  matrix[3][2] / (1.0 - matrix[2][2]) : 0.0);
    
    return *this;
}

const v::DC3&
SDLOglDC::getExternal(Point3d& t, Matrix33d& Rt) const
{
    GLdouble	matrix[4][4];
    glGetDoublev(GL_MODELVIEW_MATRIX, matrix[0]);
    Rt[0][0] = matrix[0][0];
    Rt[0][1] = matrix[1][0];
    Rt[0][2] = matrix[2][0];
    Rt[1][0] = matrix[0][1];
    Rt[1][1] = matrix[1][1];
    Rt[1][2] = matrix[2][1];
    Rt[2][0] = matrix[0][2];
    Rt[2][1] = matrix[1][2];
    Rt[2][2] = matrix[2][2];
    t[0] = -matrix[3][0];
    t[1] = -matrix[3][1];
    t[2] = -matrix[3][2];
    t *= Rt;
    
    return *this;
}

v::DC3&
SDLOglDC::translate(double dist)
{
  /* Since OpenGL does not support post-concatination, we have to do such a
     dirty thing. */
    GLdouble	matrix[4][4];
    glGetDoublev(GL_MODELVIEW_MATRIX, matrix[0]);	// Store the original.

    glLoadIdentity();
    GLdouble	dx = 0.0, dy = 0.0, dz = 0.0;
    switch (getAxis())
    {
      case v::DC3::X:
	dx = -dist;
	break;

      case v::DC3::Y:
	dy = -dist;
	break;

      case v::DC3::Z:
	dz = -dist;
	break;
    }
    glTranslated(dx, dy, dz);
    glMultMatrixd(matrix[0]);

    return v::DC3::translate(dist);
}

v::DC3&
SDLOglDC::rotate(double angle)
{
    GLdouble	matrix[4][4];
    glGetDoublev(GL_MODELVIEW_MATRIX, matrix[0]);	// Store the original.

    glLoadIdentity();
    glTranslated(0.0, 0.0, getDistance());
    GLdouble	nx = 0.0, ny = 0.0, nz = 0.0;
    switch (getAxis())
    {
      case v::DC3::X:
	nx = 1.0;
	break;

      case v::DC3::Y:
	ny = 1.0;
	break;

      default:
	nz = 1.0;
	break;
    }
    glRotated(-angle * 180.0 / M_PI, nx, ny, nz);
    glTranslated(0.0, 0.0, -getDistance());
    glMultMatrixd(matrix[0]);
    
    return *this;
}

SDLOglDC&
SDLOglDC::setViewport()
{
    u_int	w = width(), h = height();
    if (_scale > 0)
	for (int i = 0; i < _scale; ++i)
	{
	    w <<= 1;
	    h <<= 1;
	}
    else
	for (int i = 0; i > _scale; --i)
	{
	    w >>= 1;
	    h >>= 1;
	}
    glViewport(0, height() - h, w, h);
    return *this;
}

bool
SDLOglDC::callback(const SDL_Event& event)
{
    if (event.type == SDL_KEYDOWN)
    {
	if (event.key.keysym.mod == KMOD_LSHIFT)
	{
	    switch (event.key.keysym.sym)
	    {
	      case SDLK_n:
		*this << axis(Z) << v::translate(-0.05 * getDistance());
		break;
	      case SDLK_m:
		*this << axis(Z) << v::translate( 0.05 * getDistance());
		break;
	      case SDLK_h:
		*this << axis(X) << v::translate( 0.05 * getDistance());
		break;
	      case SDLK_j:
		*this << axis(Y) << v::translate(-0.05 * getDistance());
		break;
	      case SDLK_k:
		*this << axis(Y) << v::translate( 0.05 * getDistance());
		break;
	      case SDLK_l:
		*this << axis(X) << v::translate(-0.05 * getDistance());
		break;
	    }
	}
	else
	{
	    switch (event.key.keysym.sym)
	    {
	      case SDLK_ESCAPE:
		if (_fullScreen)
		    toggleFullScreen();
		else
		    return false;
	      case SDLK_SPACE:
		toggleFullScreen();
		break;
	      case SDLK_n:
		*this << axis(Z) << v::rotate( 5 * M_PI / 180.0);
		break;
	      case SDLK_m:
		*this << axis(Z) << v::rotate(-5 * M_PI / 180.0);
		break;
	      case SDLK_h:
		*this << axis(Y) << v::rotate( 5 * M_PI / 180.0);
		break;
	      case SDLK_j:
		*this << axis(X) << v::rotate( 5 * M_PI / 180.0);
		break;
	      case SDLK_k:
		*this << axis(X) << v::rotate(-5 * M_PI / 180.0);
		break;
	      case SDLK_l:
		*this << axis(Y) << v::rotate(-5 * M_PI / 180.0);
		break;
	      case SDLK_i:
		if (_scale < 3)
		{
		    ++_scale;
		    setViewport();
		}
		break;
	      case SDLK_o:
		if (_scale > -3)
		{
		    --_scale;
		    setViewport();
		}
		break;
	    }
	}
    }
    
    return true;
}

}

