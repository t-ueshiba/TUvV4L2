/*
 *  $Id$  
 */
#ifndef TU_V_ICON_H
#define TU_V_ICON_H

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
#endif	// !TU_V_ICON_H
