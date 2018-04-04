/*!
  \file		Serial.h
  \author	Toshio UESHIBA
  \brief	クラス TU::Serial の定義と実装
*/
#ifndef TU_SERIAL_H
#define TU_SERIAL_H

#include "TU/fdstream.h"
#include <termios.h>

namespace TU
{
/************************************************************************
*  class Serial								*
************************************************************************/
//! シリアルポートクラス
class Serial : public fdstream
{
  public:
		Serial(const char*)					;
    virtual	~Serial()						;

    Serial&	i_nl2cr()						;
    Serial&	i_igncr()						;
    Serial&	i_cr2nl()						;
#if !defined(__APPLE__)
    Serial&	i_upper2lower()						;
#endif
    Serial&	i_through()						;
    Serial&	o_nl2crnl()						;
#if !defined(__APPLE__)
    Serial&	o_cr2nl()						;
    Serial&	o_lower2upper()						;
#endif
    Serial&	o_through()						;
    Serial&	c_baud(int)						;
    Serial&	c_csize(int)						;
    Serial&	c_even()						;
    Serial&	c_odd()							;
    Serial&	c_noparity()						;
    Serial&	c_stop1()						;
    Serial&	c_stop2()						;

  private:
    Serial&	set_flag(tcflag_t termios::* flag,
			 unsigned long clearbits,
			 unsigned long setbits)				;

    termios	_termios_bak;
};

}
#endif	// !TU_SERIAL_H
