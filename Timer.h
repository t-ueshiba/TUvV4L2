/*
 *  $Id: Timer.h,v 1.2 2002-07-25 02:38:13 ueshiba Exp $
 */
#ifndef __TUvTimer_h
#define __TUvTimer_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class Timer								*
************************************************************************/
class Timer
{
  public:
    Timer(Object& vObject, u_long interval=0)				;
    ~Timer()								;
    
    void		start(u_long interval)				;
    void		stop()						;

  private:
    void		start()						;
    void		tick()						;

    friend void		CBtimer(XtPointer vTimer, XtIntervalId*)	;

    Object&		_vObject;
    u_long		_interval;
    XtIntervalId	_id;
};

}
}
#endif	// !__Timer_h
