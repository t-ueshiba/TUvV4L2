/*
 *  $Id$  
 */
#ifndef TU_V_BITMAP_H
#define TU_V_BITMAP_H

#include "TU/v/Colormap.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class TUvBitmap							*
************************************************************************/
class Bitmap
{
  public:
    Bitmap(const Colormap& colormap, const u_char data[])	;
    ~Bitmap()							;

    Pixmap		xpixmap()		const	{return _bitmap;}
    
  private:
    Display* const	_display;
    const Pixmap	_bitmap;
};

}
}
#endif	// !TU_V_BITMAP_H
