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
 *  $Id: Thread.cc,v 1.4 2008-09-10 05:11:51 ueshiba Exp $  
 */
#include "TU/Thread++.h"
#include <unistd.h>

namespace TU
{
/************************************************************************
*  class Thread								*
************************************************************************/
Thread::Thread()
    :_mutex(), _cond(), _thread(), _state(Idle)
{
    pthread_mutex_init(&_mutex, NULL);
    pthread_cond_init(&_cond, NULL);

    pthread_create(&_thread, NULL, threadProc, this);
  // The Exit signal may be sent to the child thread from ~Thread()
  // before the child reaches pthread_cond_wait() when ~Thread() is
  // called immediately after Thread(). This results in the child
  // failing to catch the signal and wait it forever! So, we need some
  // delay here.
    usleep(1000);
}

Thread::~Thread()
{
    pthread_mutex_lock(&_mutex);
    _state = Exit;
    pthread_cond_signal(&_cond);	// Send Exit signal to the child.
    pthread_mutex_unlock(&_mutex);

    pthread_join(_thread, NULL);	// Wait the child terminating.
    pthread_cond_destroy(&_cond);
    pthread_mutex_destroy(&_mutex);
}

void
Thread::wait() const
{
    pthread_mutex_lock(&_mutex);
    while (_state != Idle)
	pthread_cond_wait(&_cond, &_mutex);
    pthread_mutex_unlock(&_mutex);
}

void*
Thread::threadProc(void* t)
{
    Thread*	thread = (Thread*)t;

    pthread_mutex_lock(&thread->_mutex);
    for (;;)
    {
	pthread_cond_wait(&thread->_cond, &thread->_mutex);
      	if (thread->_state == Ready)
	{
	    thread->doJob();				// Do main job.
	    thread->_state = Idle;
	    pthread_cond_signal(&thread->_cond);	// Send Idle signal
	}						// to the parent.
	else if (thread->_state == Exit)
	    break;
    }
    pthread_mutex_unlock(&thread->_mutex);

    return 0;
}

}
