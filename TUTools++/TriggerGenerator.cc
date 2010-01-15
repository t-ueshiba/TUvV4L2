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
 *  $Id: TriggerGenerator.cc,v 1.22 2010-01-15 01:21:12 ueshiba Exp $
 */
#include "TU/TriggerGenerator.h"
#include "TU/Manip.h"
#include <iomanip>
#include <cstring>

namespace TU
{
/************************************************************************
*  class TriggerGenerator						*
************************************************************************/
//! 指定されたttyをopenしてトリガ信号発生器を作る．
/*!
  \param ttyname	tty名
*/
TriggerGenerator::TriggerGenerator(const char* ttyname)
    :Serial(ttyname)
{
    i_through()
	.o_through()
#if !defined(__APPLE__)
	.o_lower2upper()
#endif
	.c_baud(9600).c_csize(8).c_noparity().c_stop1();

    fill('0');
    setf(ios_base::internal, ios_base::adjustfield);
}

//! ファームウェアのIDを出力ストリームに書き出す．
/*!
  \param out	出力ストリーム
*/
void
TriggerGenerator::showId(std::ostream& out)
{
    using namespace	std;
    
    *this << 'V' << endl;
    for (char c; get(c); )
    {
	if (c == '\n')
	    break;
	out << c;
    }
    out << endl;

    *this >> skipl;
}

//! トリガ信号を出力するチャンネルを指定する．
/*!
  \param channel	出力チャンネルに対応するビットに1を立てたビットマップ
  \return		このトリガ信号発生器
*/
TriggerGenerator&
TriggerGenerator::selectChannel(u_int channel)
{
    using namespace	std;

    *this << 'A' << setw(8) << std::hex << channel << endl;
    *this >> skipl >> skipl;
    return *this;
}

//! トリガ信号の出力間隔を指定する．
/*!
  \param interval	出力間隔(msec)．10 <= interval <=
  \return		このトリガ信号発生器
*/
TriggerGenerator&
TriggerGenerator::setInterval(u_int interval)
{
    using namespace	std;
    
    if (10 <= interval && interval <= 255)
    {
	*this << 'F' << std::dec << interval << endl;
	*this >> skipl >> skipl;
    }
    return *this;
}

//! トリガ信号を1つだけ出力する．
/*!
  \return		このトリガ信号発生器
*/
TriggerGenerator&
TriggerGenerator::oneShot()
{
    using namespace	std;
    
    *this << 'T' << endl;
    *this >> skipl >> skipl;
    return *this;
}

//! トリガ信号を連続的に出力する．
/*!
  \return		このトリガ信号発生器
*/
TriggerGenerator&
TriggerGenerator::continuousShot()
{
    using namespace	std;
    
    *this << 'R' << endl;
    *this >> skipl >> skipl;
    return *this;
}

//! トリガ信号を停止する．
/*!
  \return		このトリガ信号発生器
*/
TriggerGenerator&
TriggerGenerator::stopContinuousShot()
{
    using namespace	std;
    
    *this << 'S' << endl;
    *this >> skipl >> skipl;
    return *this;
}

//! トリガ信号発生器の状態を取得する．
/*!
  \param channel	トリガ信号を出力するチャンネルに1を立てたビットマップ
			が返される
  \param interval	トリガ信号の出力間隔(msec)が返される
  \return		トリガ信号が出力中ならばtrue，そうでなければfalse
*/
bool
TriggerGenerator::getStatus(u_int& channel, u_int& interval)
{
    using namespace	std;
    
    *this << 'I' << endl;

    char	c, token[5];	// tokenは"STOP"または"RUN"のいずれか
    *this >> c >> std::hex >> channel >> c;
    *this >> c >> std::dec >> interval >> c >> token >> skipl >> skipl;
    
    return !strcmp(token, "RUN");
}

}

