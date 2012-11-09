/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id$  
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
