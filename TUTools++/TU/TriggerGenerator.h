/*!
  \file		TriggerGenerator.h
  \author	Toshio UESHIBA
  \brief	クラス TU::TriggerGenerator の定義と実装
*/
#ifndef __TU_TRIGGERGENERATOR_H
#define __TU_TRIGGERGENERATOR_H

#include "TU/Serial.h"

namespace TU
{
/************************************************************************
*  class TriggerGenerator						*
************************************************************************/
//! 東通産業製トリガ信号発生器を表すクラス
class TriggerGenerator : public Serial
{
  public:
    TriggerGenerator(const char* ttyname)				;

    void		showId(std::ostream& out)			;
    TriggerGenerator&	selectChannel(u_int channel)			;
    TriggerGenerator&	setInterval(u_int interval)			;
    TriggerGenerator&	oneShot()					;
    TriggerGenerator&	continuousShot()				;
    TriggerGenerator&	stopContinuousShot()				;
    bool		getStatus(u_int& channel, u_int& interval)	;
};

}
#endif	// !__TU_TRIGGERGENERATOR_H
