/*
 *  $Id: CanvasPaneDC3.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
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
CanvasPaneDC3::CanvasPaneDC3(CanvasPane& parentCanvasPane, u_int w, u_int h)
    :CanvasPaneDC(parentCanvasPane, w, h),
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
	    *this << TU::v::axis(Z) << TU::v::rotate( 5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'm':
	    *this << TU::v::axis(Z) << TU::v::rotate(-5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'h':
	    *this << TU::v::axis(Y) << TU::v::rotate( 5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'j':
	    *this << TU::v::axis(X) << TU::v::rotate( 5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'k':
	    *this << TU::v::axis(X) << TU::v::rotate(-5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'l':
	    *this << TU::v::axis(Y) << TU::v::rotate(-5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;

	  case 'N':
	    *this << TU::v::axis(Z) << TU::v::translate(-0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'M':
	    *this << TU::v::axis(Z) << TU::v::translate( 0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'H':
	    *this << TU::v::axis(X) << TU::v::translate( 0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'J':
	    *this << TU::v::axis(Y) << TU::v::translate(-0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'K':
	    *this << TU::v::axis(Y) << TU::v::translate( 0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'L':
	    *this << TU::v::axis(X) << TU::v::translate(-0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
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
  // Set initial internal and external parameters.
    setInternal(width() / 2, height() / 2, 800.0, 800.0, 1.0, 1000.0);
    Matrix<double>	Rt(3, 3);
    Rt[0][0] = Rt[2][1] = 1.0;
    Rt[1][2] = -1.0;
    setExternal(Vector<double>(3), Rt);
}

}
}
