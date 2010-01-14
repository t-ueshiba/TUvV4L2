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
 *  $Id: PM16C_04.cc,v 1.7 2010-01-14 11:13:09 ueshiba Exp $
 */
#include "TU/PM16C_04.h"
#include "TU/Manip.h"
#include <iomanip>
#include <stdexcept>
#include <cstring>

namespace TU
{
using namespace	std;
    
/************************************************************************
*  static functions							*
************************************************************************/
static void
checkChannel(u_int channel)
{
    if (channel >= 16)
	throw runtime_error("channel# must be less than 16!");
}
    
/************************************************************************
*  class PM16C_04							*
************************************************************************/
//! 指定されたttyをopenしてパルスモータコントローラを作る．
/*!
  \param ttyname	tty名
*/
PM16C_04::PM16C_04(const char* ttyname)
    :Serial(ttyname)
{
    i_igncr()
	.o_nl2crnl()
#if !defined(__APPLE__)
	.o_lower2upper()
#endif
	.c_baud(9600).c_csize(8).c_noparity().c_stop1();

    setMode(true);
    usleep(DELAY);

  // paddingとして符号と数値の間に'0'を出力
    fill('0');
    setf(ios_base::internal, ios_base::adjustfield);
}
    
//! ファームウェアのIDを出力ストリームに書き出す．
/*!
  \param out	出力ストリーム
*/
void
PM16C_04::showId(std::ostream& out)
{
    *this << "VER?" << endl;
    for (char c; fdstream::get(c); )
    {
	if (c == '\n')
	    break;
	out << c;
    }
    out << endl;
}

/*
 *  Local/Remoteモード
 */
//! コントローラのLOCAL/REMOTEモードを設定する．
/*!
  \param remote	trueならREMOTEモードに，falseならLOCALモードに設定
*/
PM16C_04&
PM16C_04::setMode(bool remote)
{
    *this << "S1" << (remote ? 'R' : 'L') << endl;

    return *this;
}

//! コントローラがREMOTEモードであるか調べる．
/*!
  \return	REMOTEモードならtrue，LOCALモードならfalse
*/
bool
PM16C_04::isRemoteMode()
{
    return getHardwareLimitSwitchStatus(Axis_A) & 0x8;
}

/*
 *  位置
 */
//! 指定されたチャンネルの位置を設定する．
/*!
  \param channel	チャンネル
  \param position	位置
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::setPosition(u_int channel, int position)
{
    checkChannel(channel);
    *this << "S5" << std::hex << channel << "PS"
	  << setw(8) << std::showpos << std::dec << position
	  << endl;
    usleep(DELAY);

    return *this;
}

//! 指定されたチャンネルの現在位置を調べる．
/*!
  \param channel	チャンネル
  \return		現在位置
*/
int
PM16C_04::getPosition(u_int channel)
{
    checkChannel(channel);
    *this << "S4" << std::hex << channel << "PS" << endl;
    int	position;
    *this >> std::dec >> position >> skipl;
    
    return position;
}
    
/*
 *  スピード
 */
//! すべての軸のスピードモード(LOW/MEDIUM/HIGH)を設定する．
/*!
  \param speed		スピードモード
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::setSpeed(Speed speed)
{
    *this << "S3"
	  << (speed == Speed_Low ? '4' : speed == Speed_Medium ? '5' : '6')
	  << endl;

    return *this;
}

//! 指定されたチャネルのスピードの値を設定する．
/*!
  \param channel	チャンネル
  \param speed		スピードモード
  \param val		スピードの値
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::setSpeedValue(u_int channel, Speed speed, u_int val)
{
    checkChannel(channel);
    *this << "SP"
	  << (speed == Speed_Low ? 'L' : speed == Speed_Medium ? 'M' : 'H')
	  << std::hex << channel
	  << setw(6) << std::noshowpos << std::dec << val
	  << endl;
    
    return *this;
}
    
//! 指定されたチャネルのスピードの値を調べる．
/*!
  \param channel	チャンネル
  \param speed		スピードモード
  \return		スピードの値
*/
u_int
PM16C_04::getSpeedValue(u_int channel, Speed speed)
{
    checkChannel(channel);
    *this << "SP"
	  << (speed == Speed_Low ? 'L' : speed == Speed_Medium ? 'M' : 'H')
	  << '?' << std::hex << channel << endl;
    char	c;
    u_int	val;
    *this >> c >> std::dec >> val >> skipl;

    return val;
}

/*
 *  ソフトウェアリミットスイッチ
 */
//! ソフトウェアリミットスイッチを有効化し．その位置を指定する．
/*!
  \param channel	チャンネル
  \param PositionP	正方向リミットスイッチの位置
  \param PositionN	負方向リミットスイッチの位置
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::enableSoftwareLimitSwitch(u_int channel,
				    int positionP, int positionN)
{
    setLimitSwitchConf(channel, getLimitSwitchConf(channel) | 0x20);
    *this << "S5" << std::hex << channel << "FL"
	  << setw(8) << std::showpos << std::dec << positionP
	  << endl;
    *this << "S5" << std::hex << channel << "BL"
	  << setw(8) << std::showpos << std::dec << positionN
	  << endl;
    usleep(DELAY);

    return *this;
}
    
//! ソフトウェアリミットスイッチを無効化する．
/*!
  \param channel	チャンネル
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::disableSoftwareLimitSwitch(u_int channel)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) & 0xdf);
}
    
//! ソフトウェアリミットスイッチが有効か調べる．
/*!
  \param channel	チャンネル
  \return		有効であればtrue, 無効であればfalse
*/
bool
PM16C_04::isEnabledSoftwareLimitSwitch(u_int channel)
{
    return getLimitSwitchConf(channel) & 0x20;
}

//! 正方向ソフトウェアリミットスイッチの位置を調べる．
/*!
  \param channel	チャンネル
  \return		正方向リミットスイッチの位置
*/
int
PM16C_04::getSoftwareLimitSwitchPositionP(u_int channel)
{
    checkChannel(channel);
    *this <<"S4" << std::hex << channel << "FL" << endl;
    int	position;
    *this >> std::dec >> position >> skipl;

    return position;
}
    
//! 負方向ソフトウェアリミットスイッチの位置を調べる．
/*!
  \param channel	チャンネル
  \return		負方向リミットスイッチの位置
*/
int
PM16C_04::getSoftwareLimitSwitchPositionN(u_int channel)
{
    checkChannel(channel);
    *this <<"S4" << std::hex << channel << "BL" << endl;
    int	position;
    *this >> std::dec >> position >> skipl;

    return position;
}
    
/*
 *  ハードウェアリミットスイッチ
 */
//! ハードウェアリミットスイッチを有効化し．その極性を設定する．
/*!
  \param channel	チャンネル
  \param dir		正方向リミットスイッチならtrue，
			負方向リミットスイッチならfalse
  \param normallyClose	スイッチが働いていないときにcloseならtrue，openならtrue
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::enableHardwareLimitSwitch(u_int channel, bool dir,
				    bool normallyClose)
{
    u_int	conf = getLimitSwitchConf(channel) | (dir ? 0x08: 0x10);
    if (normallyClose)
	return setLimitSwitchConf(channel, conf | (dir ? 0x01 : 0x02));
    else
	return setLimitSwitchConf(channel, conf & (dir ? 0xfe : 0xfd));
}

//! ハードウェアリミットスイッチを無効化する．
/*!
  \param channel	チャンネル
  \param dir		正方向リミットスイッチならtrue，
			負方向リミットスイッチならfalse
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::disableHardwareLimitSwitch(u_int channel, bool dir)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) &
				       (dir ? 0xf7 : 0xef));
}

//! ハードウェアリミットスイッチが有効であるか調べる．
/*!
  \param channel	チャンネル
  \param dir		正方向リミットスイッチならtrue，
			負方向リミットスイッチならfalse
  \return		有効ならtrue, 無効ならfalse
*/
bool
PM16C_04::isEnabledHardwareLimitSwitch(u_int channel, bool dir)
{
    return getLimitSwitchConf(channel) & (dir ? 0x08 : 0x10);
}

//! ハードウェアリミットスイッチの極性を調べる．
/*!
  \param channel	チャンネル
  \param dir		正方向リミットスイッチならtrue，
			負方向リミットスイッチならfalse
  \return		スイッチが働いていないときcloseならtrue, openならfalse
*/
bool
PM16C_04::getHardwareLimitSwitchPolarity(u_int channel, bool dir)
{
    return getLimitSwitchConf(channel) & (dir ? 0x01 : 0x02);
}

//! ハードウェアホームポジションスイッチの極性を設定する．
/*!
  \param channel	チャンネル
  \param normallyClose	スイッチが働いていないときにcloseならtrue，openならtrue
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::setHomeSwitchPolarity(u_int channel, bool normallyClose)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) | 0x04);
}

//! ハードウェアホームポジションスイッチの極性を調べる．
/*!
  \param channel	チャンネル
  \return		スイッチが働いていないときcloseならtrue, openならfalse
*/
bool
PM16C_04::getHomeSwitchPolarity(u_int channel)
{
    return getLimitSwitchConf(channel) & 0x04;
}

/*
 *  バックラッシュ補正
 */
//! バックラッシュ補正のステップ数を設定する．
/*!
  \param channel	チャンネル
  \param steps		ステップ数
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::setBacklashCorrectionStep(u_int channel, u_int steps)
{
    checkChannel(channel);
    *this << 'B' << std::hex << channel
	  << setw(8) << std::showpos << std::dec << steps
	  << endl;

    return *this;
}

//! バックラッシュ補正のステップ数を調べる．
/*!
  \param channel	チャンネル
  \return		ステップ数
*/
u_int
PM16C_04::getBacklashCorrectionStep(u_int channel)
{
    checkChannel(channel);
    *this << 'B' << std::hex << channel << '?' << endl;
    char	c;
    u_int	val;
    *this >> c >> std::dec >> val >> skipl;

    return val;
}

/*
 *  Hold off機能（停止時の非通電）
 */
//! Hold off機能（停止時の非通電）を有効化する．
/*!
  \param channel	チャンネル
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::enableHoldOff(u_int channel)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) & 0xbf);
}

//! Hold off機能（停止時の非通電）を無効化する．
/*!
  \param channel	チャンネル
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::disableHoldOff(u_int channel)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) | 0x40);
}

//! Hold off機能（停止時の非通電）が有効か調べる．
/*!
  \param channel	チャンネル
  \return		有効ならtrue, 無効ならfalse
*/
bool
PM16C_04::isEnabledHoldOff(u_int channel)
{
    return getLimitSwitchConf(channel) & 0x40;
}

/*
 *  ホームポジション検出
 */
//! どちら側からホームポジション検出を行うかを設定する．
/*!
  \param channel	チャンネル
  \param dir		正方向からであればtrue, 負方向からであればfalse
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::setHomeSearchDirection(u_int channel, bool dir)
{
    checkChannel(channel);
    *this << 'D' << (dir ? 'P' : 'N') << std::hex << channel << endl;

    return *this;
}
    
//! どちら側からホームポジション検出を行うのか調べる．
/*!
  \param channel	チャンネル
  \return		正方向からであればtrue, 負方向からであればfalse
*/
bool
PM16C_04::getHomeSearchDirection(u_int channel)
{
    return getHomeStatus(channel) & 0x1;
}
    
//! ホームポジション検出時のオフセット値を設定する．
/*!
  \param channel	チャンネル
  \param offset		オフセット値
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::setHomeOffset(u_int channel, u_int offset)
{
    checkChannel(channel);
    *this << "GF" << std::hex << channel
	  << setw(4) << std::noshowpos << std::dec << offset
	  << endl;

    return *this;
}
    
//! ホームポジション検出時のオフセット値を調べる．
/*!
  \param channel	チャンネル
  \return		オフセット値
*/
u_int
PM16C_04::getHomeOffset(u_int channel)
{
    checkChannel(channel);
    *this << "GF?" << std::hex << channel << endl;
    char	c;
    u_int	offset;
    *this >> c >> std::dec >> offset >> skipl;

    return offset;
}
    
//! ホームポジションが検出済みか調べる．
/*!
  \param channel	チャンネル
  \return		検出済みならtrue, 未検出ならfalse
*/
bool
PM16C_04::isHomeFound(u_int channel)
{
    return getHomeStatus(channel) & 0x4;
}
    
//! 検出済みのホームポジションがどちらから検出されたのか調べる．
/*!
  \param channel	チャンネル
  \return		正方向からならtrue, 負方向からならfalse
*/
bool
PM16C_04::isHomeFoundFromFront(u_int channel)
{
    return getHomeStatus(channel) & 0x2;
}

//! 検出済みのホームポジションの位置を調べる．
/*!
  \param channel	チャンネル
  \return		ホームポジション位置
*/
int
PM16C_04::getHomePosition(u_int channel)
{
    checkChannel(channel);
    *this << "HP?" << std::hex << channel << endl;
    int	position;
    *this >> std::dec >> position >> skipl;

    return position;
}

//! ホームポジションを検出する（高精度だが長時間を要す）．
/*
  \param axis	ホームポジション検出を実行する軸
  \return	このコントローラ
*/
PM16C_04&
PM16C_04::findHome(Axis axis)
{
    const u_int	channel = getChannel(axis);
    if (!isEnabledHardwareLimitSwitch(channel, getHomeSearchDirection(channel)))
	throw runtime_error("PM16C_04::findHome(): hardware limit switch is disabled!");

    *this << "FHP" << (axis == Axis_A ? 'A' :
		       axis == Axis_B ? 'B' :
		       axis == Axis_C ? 'C' : 'D')
	  << endl;

    while (isBusy(axis))
	;

    return *this;
}
    
//! ホームポジションに移動する．
/*
  \param axis	ホームポジションに移動する軸
  \return	このコントローラ
*/
PM16C_04&
PM16C_04::goHome(Axis axis)
{
    if (!isHomeFound(getChannel(axis)))
	throw runtime_error("PM16C_04::goHome(): home position is not found yet!");

    *this << "RTHP" << (axis == Axis_A ? 'A' :
			axis == Axis_B ? 'B' :
			axis == Axis_C ? 'C' : 'D')
	  << endl;

    while (isBusy(axis))
	;
    
    return *this;
}
    
/*
 *  軸とチャンネルの関係
 */
//! 指定した軸に指定したチャンネルを結びつける．
/*!
  \param axis		軸
  \param channel	チャンネル
  \return		このコントローラ
*/
PM16C_04&
PM16C_04::setChannel(Axis axis, u_int channel)
{
    checkChannel(channel);
    *this << "S1" << (axis == Axis_A ? '1' :
		      axis == Axis_B ? '2' :
		      axis == Axis_C ? '5' : '6')
	  << std::hex << channel
	  << endl;
    usleep(DELAY);
    
    return *this;
}

//! 指定した軸にどのチャンネルが結びつけられているか調べる．
/*!
  \param axis		軸
  \return		チャンネル
*/
u_int
PM16C_04::getChannel(Axis axis)
{
    u_int	channel_A, channel_B, channel_C, channel_D;
    getChannel(channel_A, channel_B, channel_C, channel_D);

    return (axis == Axis_A ? channel_A :
	    axis == Axis_B ? channel_B :
	    axis == Axis_C ? channel_C : channel_D);
}
    
//! 4つの軸にそれぞれどのチャンネルが結びつけられているか調べる．
/*!
  \param channel_A	#Axis_Aに結びつけられたチャンネルが返される
  \param channel_B	#Axis_Bに結びつけられたチャンネルが返される
  \param channel_C	#Axis_Cに結びつけられたチャンネルが返される
  \param channel_D	#Axis_Dに結びつけられたチャンネルが返される
*/
void
PM16C_04::getChannel(u_int& channel_A, u_int& channel_B,
		     u_int& channel_C, u_int& channel_D)
{
    *this << "S10" << endl;
    char	c;
    u_int	channel;
    *this >> c >> std::hex >> channel >> skipl;
    channel_A = (channel >> 12) & 0xf;
    channel_B = (channel >>  8) & 0xf;
    channel_C = (channel >>  4) & 0xf;
    channel_D =  channel	& 0xf;
}

/*
 *  軸の状態
 */
//! 指定した軸の現在位置を調べる．
/*!
  \param axis	軸
  \return	現在位置
*/
int
PM16C_04::where(Axis axis)
{
    *this << "S2" << (axis == Axis_A ? '0' :
		      axis == Axis_B ? '2' :
		      axis == Axis_C ? '4' : '6')
	  << 'D'
	  << endl;
    int	position;
    *this >> std::dec >> position >> skipl;

    return position;
}

//! 指定した軸において何らかのコマンドが実行中か調べる．
/*!
  \param axis	軸
  \return	実行中ならtrue, そうでなければfalse
*/
bool
PM16C_04::isBusy(Axis axis)
{
    return getControllerStatus(axis) & 0x1;
}

//! 指定した軸においてパルスが発生中か調べる．
/*!
  \param axis	軸
  \return	発生中ならtrue, そうでなければfalse
*/
bool
PM16C_04::isPulseEmitted(Axis axis)
{
    return getControllerStatus(axis) & 0x2;
}

//! 指定した軸において発生中であったパルスが停止したか調べる．
/*!
  \param axis	軸
  \return	停止したならtrue, そうでなければfalse
*/
bool
PM16C_04::isPulseStopped(Axis axis)
{
    return getControllerStatus(axis) & 0x4;
}

//! 指定した軸においてリミットスイッチがONであるか調べる．
/*!
  \param axis	軸
  \param dir	正方向リミットスイッチならtrue, 負方向リミットスイッチならfalse
  \return	ONならtrue, OFFならfalse
*/
bool
PM16C_04::atLimit(Axis axis, bool dir)
{
    return !(getHardwareLimitSwitchStatus(axis) & (dir ? 0x1 : 0x2));
}

//! 指定した軸においてホームポジションスイッチがONであるか調べる．
/*!
  \param axis	軸
  \return	ONならtrue, OFFならfalse
*/
bool
PM16C_04::atHome(Axis axis)
{
    return !(getHardwareLimitSwitchStatus(axis) & 0x4);
}

/*
 *  移動
 */
//! 指定した軸を減速停止する．
/*!
  \param axis	軸
  \return	このコントローラ　
*/
PM16C_04&
PM16C_04::stop(Axis axis)
{
    return move(axis, "40");
}
    
//! 指定した軸をjog動作させる．
/*!
  \param axis	軸
  \param dir	正方向ならtrue, 負方向ならfalse
  \return	このコントローラ　
*/
PM16C_04&
PM16C_04::jog(Axis axis, bool dir)
{
   return move(axis, (dir ? "08" : "09"));
}
    
//! 指定した軸を一定速度でスキャンする．
/*!
  \param axis	軸
  \param dir	正方向ならtrue, 負方向ならfalse
  \return	このコントローラ　
*/
PM16C_04&
PM16C_04::scanWithConstantSpeed(Axis axis, bool dir)
{
    return move(axis, (dir ? "0C" : "0D"));
}
    
//! 指定した軸を台形速度パターンでスキャンする．
/*!
  \param axis	軸
  \param dir	正方向ならtrue, 負方向ならfalse
  \return	このコントローラ　
*/
PM16C_04&
PM16C_04::scan(Axis axis, bool dir)
{
    return move(axis, (dir ? "0E" : "0F"));
}
    
//! 指定した軸を一時停止する．
/*!
  \param axis	軸
  \param on	
  \return	このコントローラ　
*/
PM16C_04&
PM16C_04::pause(Axis axis, bool on)
{
    return move(axis, (on ? "16" : "17"));
}
    
//! 指定した軸のHold off機能（停止時の非通電）を設定する．
/*!
  \param axis	軸
  \param set	機能を使用するならtrue, 使用しないならfalsse	
  \return	このコントローラ
*/
PM16C_04&
PM16C_04::holdOff(Axis axis, bool set)
{
    return move(axis, (set ? "18" : "19"));
}

//! 指定した軸をスキャンしながらホームポジションを検出する（高速だが低精度）．
/*!
  \param axis	軸
  \param dir	正方向にスキャンするならtrue, 負方向ならfalse
  \return	このコントローラ
*/
PM16C_04&
PM16C_04::scanAndStopAtHome(Axis axis, bool dir)
{
    return move(axis, (dir ? "1E" : "1F"));
}

//! 指定した軸を移動する．
/*!
  \param axis			軸
  \param relative		相対的な移動ならtrue, 絶対的な移動ならfalse
  \param val			移動量（相対的な移動）
				または目標位置（絶対的な移動）
  \param correctBacklash	停止時にバックラッシュ補正を行うならtrue,
				行わないならfase
  \return			このコントローラ
*/
PM16C_04&
PM16C_04::move(Axis axis, bool relative, int val, bool correctBacklash)
{
    *this << "S3" << (axis == Axis_A ? '2' :
		      axis == Axis_B ? '3' :
		      axis == Axis_C ? 'A' : 'B')
	  << (relative ? 'R' : 'A')
	  << setw(8) << std::showpos << std::dec << val;
    if (correctBacklash)
	*this << 'B';
    *this << endl;

    return *this;
}

/*
 *  Parallel I/Oポート
 */
//! パラレルI/Oポートを有効化する．
/*!
  \return	このコントローラ
*/
PM16C_04&
PM16C_04::enableParallelIO()
{
    *this << "PIO" << endl;

    return *this;
}
    
//! パラレルI/Oポートを無効化する．
/*!
  \return	このコントローラ
*/
PM16C_04&
PM16C_04::disableParallelIO()
{
    *this << "COM" << endl;

    return *this;
}

//! パラレルI/Oポートが有効か調べる．
/*!
  \return	有効ならtrue, 無効ならfalse
*/
bool
PM16C_04::isEnabledParallelIO()
{
    *this << "PIO?" << endl;
    char	response[4];
    getline(response, sizeof(response));

    return !strncmp(response, "PIO", 3);
}

//! パラレルI/Oポートから読み込む．
/*!
  \return	読み込んだ値
*/
u_int
PM16C_04::readParallelIO()
{
    *this << "RD" << endl;
    char	c;
    u_int	val;
    *this >> c >> std::hex >> val >> skipl;

    return val;
}
    
//! パラレルI/Oポートに書き出す．
/*!
  \param val	書き出す値
  \return	このコントローラ
*/
PM16C_04&
PM16C_04::writeParallelIO(u_int val)
{
    *this << 'W' << setw(4) << std::hex << (val & 0xffff)
	  << endl;

    return *this;
}
    
/*
 *  private member functions
 */
u_int
PM16C_04::getLimitSwitchConf(u_int channel)
{
    checkChannel(channel);
    *this << "S4" << std::hex << channel << 'D' << endl;
    char	c;
    u_int	conf;
    *this >> c >> std::hex >> conf >> skipl;

    return conf >> 16;
}

PM16C_04&
PM16C_04::setLimitSwitchConf(u_int channel, u_int conf)
{
    checkChannel(channel);
    *this << "S5" << std::hex << channel << 'D'
	  << setw(2) << std::hex << conf
	  << endl;
    usleep(DELAY);

    return *this;
}

u_int
PM16C_04::getHardwareLimitSwitchStatus(Axis axis)
{
    *this << "S6" << endl;
    char	c;
    u_int	status;
    *this >> c >> std::hex >> status >> skipl;

    return (status >> (axis == Axis_A ?  8 :
		       axis == Axis_B ? 12 :
		       axis == Axis_C ?  0 : 4)) & 0xf;
}
    
u_int
PM16C_04::getHomeStatus(u_int channel)
{
    checkChannel(channel);
    *this << "G?" << std::hex << channel << endl;
    char	c;
    u_int	status;
    *this >> c >> std::hex >> status >> skipl;

    return status;
}
    
u_int
PM16C_04::getControllerStatus(Axis axis)
{
    *this << "S2" << (axis == Axis_A ? '1' :
		      axis == Axis_B ? '3' :
		      axis == Axis_C ? '5' : '7')
	  << endl;
    char	c;
    u_int	status;
    *this >> c >> std::hex >> status >> skipl;

    return status;
}
    
PM16C_04&
PM16C_04::move(Axis axis, const char* motion)
{
    *this << "S3" << (axis == Axis_A ? '0' :
		      axis == Axis_B ? '1' :
		      axis == Axis_C ? '8' : '9')
	  << motion << endl;

    return *this;
}
    
}
