/*
 *  $Id: Bitmap.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifndef __TUvBitmap_h
#define __TUvBitmap_h

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
#endif	// !__TUvBitmap_h
