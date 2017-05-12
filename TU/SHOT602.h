/*!
  \file		SHOT602.h
  \author	Toshio UESHIBA
  \brief	クラス TU::SHOT602 の定義と実装
*/
#ifndef __TU_SHOT602_H
#define __TU_SHOT602_H

#include "TU/Serial.h"

namespace TU
{
/************************************************************************
*  class SHOT602							*
************************************************************************/
//! シグマ光機製パルスモータコントローラSHOT-604を制御するクラス
class SHOT602 : public Serial
{
  public:
  //! 軸
    enum Axis
    {
	Axis_1,		//!< 第1軸
	Axis_2,		//!< 第2軸
	Axis_Both	//!< 第1,2軸の両方
    };

  //! スピード
    enum Speed
    {
	LowSpeed,	//!< 低速
	HighSpeed	//!< 高速
    };

  public:
    SHOT602(const char* ttyname)					;

  // ファームウェアバージョン
    void	showId(std::ostream& out)				;

  // ホームポジション検出
    SHOT602&	findHome(Axis axis, bool dir, bool dir2=true)		;
    SHOT602&	setOrigin(Axis axis)					;
    
  // 状態検出
    bool	isBusy()						;
    int		where(Axis axis)					;
    bool	atLimit(Axis axis)					;
    bool	getStatus(int& position1, int& position2,
			  bool& limit1, bool& limit2)			;

  // 速度設定
    SHOT602&	setSpeed(Speed speed,
			 u_int bottom1=0,   u_int top1=0,
			 u_int duration1=0, u_int bottom2=0,
			 u_int top2=0,	    u_int duration2=0)		;
    
  // 移動
    SHOT602&	stop(Axis axis)						;
    SHOT602&	emergencyStop()						;
    SHOT602&	jog(Axis axis, bool dir, bool dir2=true)		;
    SHOT602&	move(Axis axis, int val, int val2=0, bool block=false)	;

  // 励磁
    SHOT602&	setHold(Axis axis, bool on1, bool on2=true)		;

  private:
    SHOT602&	putCommand(Axis axis, char command,
			   const char* arg, const char* arg2,
			   bool putDelimieter=true)			;
};
    
}
#endif	// !__TU_SHOT602_H
