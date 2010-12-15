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
 *  $Id: SHOT602.h,v 1.1 2010-12-15 03:55:57 ueshiba Exp $
 */
#ifndef __TUSHOT602_h
#define __TUSHOT602_h

#include "TU/Serial.h"

namespace TU
{
/************************************************************************
*  class SHOT602							*
************************************************************************/
//! シグマ光機製パルスモータコントローラSHOT-604を制御するクラス
class __PORT SHOT602 : public Serial
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
    SHOT602&	move(Axis axis, int val, int val2=0)			;

  // 励磁
    SHOT602&	setHold(Axis axis, bool on1, bool on2=true)		;

  private:
    SHOT602&	putCommand(Axis axis, char command,
			   const char* arg, const char* arg2,
			   bool putDelimieter=true)			;
};
    
}

#endif	// !__TUSHOT602_h
