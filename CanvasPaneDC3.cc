/*
 *  $Id$  
 */
#include "TU/v/CanvasPaneDC3.h"
#include <X11/keysym.h>

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPaneDC							*
************************************************************************/
/*
 *  Public member functions
 */
CanvasPaneDC3::CanvasPaneDC3(CanvasPane& parentCanvasPane,
			     size_t w, size_t h, float zoom)
    :CanvasPaneDC(parentCanvasPane, w, h, zoom),
     DC3(DC3::X, 128.0)
{
}

CanvasPaneDC3::~CanvasPaneDC3()
{
}

void
CanvasPaneDC3::callback(CmdId id, CmdVal val)
{
    if (id == Id_KeyPress)
	switch (val)
	{
	  case 'n':
	    *this << axis(Z) << v::rotate( 5 * M_PI / 180.0)
		  << v::repaint;
	    break;
	  case 'm':
	    *this << axis(Z) << v::rotate(-5 * M_PI / 180.0)
		  << v::repaint;
	    break;
	  case 'h':
	    *this << axis(Y) << v::rotate( 5 * M_PI / 180.0)
		  << v::repaint;
	    break;
	  case 'j':
	    *this << axis(X) << v::rotate( 5 * M_PI / 180.0)
		  << v::repaint;
	    break;
	  case 'k':
	    *this << axis(X) << v::rotate(-5 * M_PI / 180.0)
		  << v::repaint;
	    break;
	  case 'l':
	    *this << axis(Y) << v::rotate(-5 * M_PI / 180.0)
		  << v::repaint;
	    break;

	  case 'N':
	    *this << axis(Z) << v::translate(-0.05 * getDistance())
		  << v::repaint;
	    break;
	  case 'M':
	    *this << axis(Z) << v::translate( 0.05 * getDistance())
		  << v::repaint;
	    break;
	  case 'H':
	    *this << axis(X) << v::translate( 0.05 * getDistance())
		  << v::repaint;
	    break;
	  case 'J':
	    *this << axis(Y) << v::translate(-0.05 * getDistance())
		  << v::repaint;
	    break;
	  case 'K':
	    *this << axis(Y) << v::translate( 0.05 * getDistance())
		  << v::repaint;
	    break;
	  case 'L':
	    *this << axis(X) << v::translate(-0.05 * getDistance())
		  << v::repaint;
	    break;
	}
    else
	CanvasPaneDC::callback(id, val);
}

/*
 *  Protected member functions
 */
void
CanvasPaneDC3::initializeGraphics()
{
  // Initialize 2D part.
    CanvasPaneDC::initializeGraphics();
    
  // Set initial internal and external parameters.
    setInternal(width() / 2, height() / 2, 800.0, 800.0, 1.0, 1000.0);
    Matrix33d	Rt;
    Rt[0][0] = Rt[2][1] = 1.0;
    Rt[1][2] = -1.0;
    setExternal(Point3d(), Rt);
}

}
}
