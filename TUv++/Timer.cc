/*
 *  $Id: Timer.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/Timer.h"
#include "TU/v/App.h"

namespace TU
{
namespace v
{
/************************************************************************
*  callback for class Timer						*
************************************************************************/
void
CBtimer(XtPointer vTimer, XtIntervalId*)
{
    ((Timer*)vTimer)->tick();
}

/************************************************************************
*  class Timer							*
************************************************************************/
/*
 *  public member functions.
 */
Timer::Timer(Object& vObject, u_long interval)
    :_vObject(vObject), _interval(interval), _id(0)
{
    if (_interval != 0)
	start();
}

Timer::~Timer()
{
    stop();
}

void
Timer::start(u_long interval)
{
    stop();
    _interval = interval;
    if (_interval != 0)
	start();
}

void
Timer::stop()
{
    if (_id != 0)
    {
	XtRemoveTimeOut(_id);
	_id = 0;
    }
}

/*
 *  private member functions.
 */
void
Timer::start()
{
    _id = XtAppAddTimeOut(XtWidgetToApplicationContext(_vObject.widget()),
			  _interval, CBtimer, (XtPointer)this);
}

void
Timer::tick()
{
    _vObject.tick();
    start();
}

}
}
