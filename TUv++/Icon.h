/*
 *  $Id: Icon.h,v 1.2 2002-07-25 02:38:11 ueshiba Exp $
 */
#ifndef __TUvIcon_h
#define __TUvIcon_h

#include "TU/v/Colormap.h"
#include <X11/Xutil.h>

namespace TU
{
namespace v
{
/************************************************************************
*  class Icon								*
************************************************************************/
class Icon
{
  public:
    Icon(const Colormap& colormap, const u_char data[])	;
    ~Icon()						;

    Pixmap		xpixmap()		const	{return _pixmap;}
    
  private:
    Display* const	_display;
    const Pixmap	_pixmap;
};

}
}
#endif	// !__TUvIcon_h
