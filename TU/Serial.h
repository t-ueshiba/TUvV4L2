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
/*!
  \file		Serial.h
  \brief	クラス TU::Serial の定義と実装
*/
#ifndef __TUSerial_h
#define __TUSerial_h

#include "TU/fdstream.h"
#include <termios.h>

namespace TU
{
/************************************************************************
*  class Serial								*
************************************************************************/
//! シリアルポートクラス
class __PORT Serial : public fdstream
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

#endif	// !__TUSerial_h
