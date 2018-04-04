/*!
  \file		TriggerGenerator.cc
  \author	Toshio UESHIBA
  \brief	クラス TU::TriggerGenerator の実装
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

