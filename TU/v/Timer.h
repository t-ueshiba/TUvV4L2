/*
 *  $Id$  
 */
#ifndef TU_V_TIMER_H
#define TU_V_TIMER_H

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
#endif	// !TU_V_TIMER_H
