/*
 *  $Id: ShmDC.h,v 1.1.1.1 2002-07-25 02:14:18 ueshiba Exp $
 */
#ifndef __TUvShmDC_h
#define __TUvShmDC_h

#include "TU/v/CanvasPaneDC.h"
#include <X11/extensions/XShm.h>

namespace TU
{
namespace v
{
/************************************************************************
*  class ShmDC								*
************************************************************************/
class ShmDC : public CanvasPaneDC
{
  public:
    ShmDC(CanvasPane& parentCanvasPane, u_int width=0, u_int height=0)	;
    virtual		~ShmDC()					;

  protected:
    virtual void	allocateXImage(int buffWidth, int buffHeight)	;
    virtual void	putXImage()				const	;
    char*		attachShm(u_int size)				;
    virtual void	destroyShmImage()				;
    XShmSegmentInfo*	xShmInfo()					;
    
  private:
    XShmSegmentInfo	_xShmInfo;
    u_int		_xShmSize;	// Size of shm currently allocated.
    bool		_xShmAvailable;
};

inline XShmSegmentInfo*
ShmDC::xShmInfo()
{
    return &_xShmInfo;
}

}
}
#endif	// !__TUvShmDC_h
