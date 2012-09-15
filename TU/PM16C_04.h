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
 *  $Id: PM16C_04.h,v 1.1 2012-09-15 03:59:19 ueshiba Exp $
 */
/*!
  \file		PM16C_04.h
  \brief	クラス TU::PM16C_04 の定義と実装
*/
#ifndef __TUPM16C_04_h
#define __TUPM16C_04_h

#include "TU/Serial.h"

namespace TU
{
/************************************************************************
*  class PM16C_04							*
************************************************************************/
//! ツジ電子製パルスモータコントローラPM16C_04を制御するクラス
class __PORT PM16C_04 : public Serial
{
  public:
  //! 軸
    enum Axis
    {
	Axis_A,		//!< A軸
	Axis_B,		//!< B軸
	Axis_C,		//!< C軸
	Axis_D		//!< D軸
    };

  //! スピードモード
    enum Speed
    {
	Speed_Low,	//!< 低速
	Speed_Medium,	//!< 中速
	Speed_High	//!< 高速
    };

  public:
    PM16C_04(const char* ttyname)					;

  // ファームウェアバージョン
    void	showId(std::ostream& out)				;

  // Local/Remoteモード
    PM16C_04&	setMode(bool remote)					;
    bool	isRemoteMode()						;

  // 位置
    PM16C_04&	setPosition(u_int channel, int position)		;
    int		getPosition(u_int channel)				;
    
  // スピード
    PM16C_04&	setSpeed(Speed speed)					;
    PM16C_04&	setSpeedValue(u_int channel, Speed speed, u_int val)	;
    u_int	getSpeedValue(u_int channel, Speed speed)		;

  // ソフトウェアリミットスイッチ
    PM16C_04&	enableSoftwareLimitSwitch(u_int channel,
					  int positionP, int positionN)	;
    PM16C_04&	disableSoftwareLimitSwitch(u_int channel)		;
    bool	isEnabledSoftwareLimitSwitch(u_int channel)		;
    int		getSoftwareLimitSwitchPositionP(u_int channel)		;
    int		getSoftwareLimitSwitchPositionN(u_int channel)		;
    
  // ハードウェアリミットスイッチ，ホームスイッチ
    PM16C_04&	enableHardwareLimitSwitch(u_int channel, bool dir,
					  bool normallyClose)		;
    PM16C_04&	disableHardwareLimitSwitch(u_int channel, bool dir)	;
    bool	isEnabledHardwareLimitSwitch(u_int channel,
					     bool dir)			;
    bool	getHardwareLimitSwitchPolarity(u_int channel,
					       bool dir)		;
    PM16C_04&	setHomeSwitchPolarity(u_int channel, bool normallyClose);
    bool	getHomeSwitchPolarity(u_int channel)			;

  // バックラッシュ補正
    PM16C_04&	setBacklashCorrectionStep(u_int channel, u_int steps)	;
    u_int	getBacklashCorrectionStep(u_int channel)		;

  // Hold off機能（停止時の非通電）
    PM16C_04&	enableHoldOff(u_int channel)				;
    PM16C_04&	disableHoldOff(u_int channel)				;
    bool	isEnabledHoldOff(u_int channel)				;

  // ホームポジション検出
    PM16C_04&	setHomeSearchDirection(u_int channel, bool dir)		;
    bool	getHomeSearchDirection(u_int channel)			;
    PM16C_04&	setHomeOffset(u_int channel, u_int offset)		;
    u_int	getHomeOffset(u_int channel)				;
    bool	isHomeFound(u_int channel)				;
    bool	isHomeFoundFromFront(u_int channel)			;
    int		getHomePosition(u_int channel)				;
    PM16C_04&	findHome(Axis axis)					;
    PM16C_04&	goHome(Axis axis)					;
    
  // 軸とチャンネルの関係
    PM16C_04&	setChannel(Axis axis, u_int channel)			;
    u_int	getChannel(Axis axis)					;
    void	getChannel(u_int& channel_A, u_int& channel_B,
			   u_int& channel_C, u_int& channel_D)		;

  // 軸の状態
    int		where(Axis axis)					;
    bool	isBusy(Axis axis)					;
    bool	isPulseEmitted(Axis axis)				;
    bool	isPulseStopped(Axis axis)				;
    bool	atLimit(Axis axis, bool dir)				;
    bool	atHome(Axis axis)					;
    
  // 移動
    PM16C_04&	stop(Axis axis)						;
    PM16C_04&	jog(Axis axis, bool dir)				;
    PM16C_04&	scanWithConstantSpeed(Axis axis, bool dir)		;
    PM16C_04&	scan(Axis axis, bool dir)				;
    PM16C_04&	pause(Axis axis, bool on)				;
    PM16C_04&	holdOff(Axis axis, bool set)				;
    PM16C_04&	scanAndStopAtHome(Axis axis, bool dir)			;
    PM16C_04&	move(Axis axis, bool relative,
		     int val, bool correctBacklash)			;

  // Parallel I/Oポート
    PM16C_04&	enableParallelIO()					;
    PM16C_04&	disableParallelIO()					;
    bool	isEnabledParallelIO()					;
    u_int	readParallelIO()					;
    PM16C_04&	writeParallelIO(u_int val)				;
    
  private:
    PM16C_04&	setLimitSwitchConf(u_int channel, u_int conf)		;
    u_int	getLimitSwitchConf(u_int channel)			;
    u_int	getHomeStatus(u_int channel)				;
    u_int	getHardwareLimitSwitchStatus(Axis axis)			;
    u_int	getControllerStatus(Axis axis)				;
    PM16C_04&	move(Axis axis, const char* motion)			;

  private:
    enum	{DELAY = 50000};
};
    
}

#endif	// !__TUPM16C_04_h
