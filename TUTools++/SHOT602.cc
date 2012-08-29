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
 *  $Id: SHOT602.cc,v 1.6 2012-08-29 21:17:08 ueshiba Exp $
 */
#include "TU/SHOT602.h"
#include "TU/Manip.h"
#include <stdexcept>
#include <cstdlib>

namespace TU
{
using namespace	std;
    
/************************************************************************
*  class SHOT602							*
************************************************************************/
//! 指定されたttyをopenしてパルスモータコントローラを作る．
/*!
  \param ttyname	tty名
*/
SHOT602::SHOT602(const char* ttyname)
    :Serial(ttyname)
{
    i_igncr()
	.o_nl2crnl()
#if !defined(__APPLE__)
	.o_lower2upper()
#endif
	.c_baud(9600).c_csize(8).c_noparity().c_stop1();

    setSpeed(HighSpeed);
}
    
//! ファームウェアのIDを出力ストリームに書き出す．
/*!
  \param out	出力ストリーム
*/
void
SHOT602::showId(std::ostream& out)
{
    *this << "?:V" << endl;
    for (char c; fdstream::get(c); )
    {
	if (c == '\n')
	    break;
	out << c;
    }
    out << endl;
}

/*
 *  ホームポジション検出
 */
//! ホームポジションを検出する．
/*!
  ホームポジションが検出されるまでホスト側に制御を返さない．
  \param axis	ホームポジション検出を実行する軸
  \param dir	正方向からであればtrue, 負方向からであればfalse
  \param dir2	axisが #Axis_Both の場合の第2軸について，
		正方向からであればtrue, 負方向からであればfalse
  \return	このコントローラ
*/
SHOT602&
SHOT602::findHome(Axis axis, bool dir, bool dir2)
{
    putCommand(axis, 'H', (dir ? "+" : "-"), (dir2 ? "+" : "-"));
    while (isBusy())
	;

    return *this;
}
    
//! 現在位置を座標原点に設定する．
/*
  \param axis	ホームポジションに移動する軸
  \return	このコントローラ
*/
SHOT602&
SHOT602::setOrigin(Axis axis)
{
    return putCommand(axis, 'R', "", "");
}
    
/*
 *  状態検出
 */
//! 何らかのコマンドが実行中か調べる．
/*!
  \return	実行中ならtrue, そうでなければfalse
*/
bool
SHOT602::isBusy()
{
    *this << "!:" << endl;
    char	c;
    *this >> c >> skipl;
    
    return (c == 'B');
}

//! 現在位置を調べる．
/*!
  \param axis	軸
  \return	現在位置
*/
int
SHOT602::where(Axis axis)
{
    int		position1, position2;
    bool	limit1, limit2;
    getStatus(position1, position2, limit1, limit2);

    switch (axis)
    {
      case Axis_1:
	return position1;
      case Axis_2:
	return position2;
    }

    throw runtime_error("SHOT602::where(): unknown axis!");

    return 0;
}

//! 指定した軸においてリミットスイッチがONであるか調べる．
/*!
  \param axis	軸
  \return	ONならtrue, OFFならfalse
*/
bool
SHOT602::atLimit(Axis axis)
{
    int		position1, position2;
    bool	limit1, limit2;
    getStatus(position1, position2, limit1, limit2);

    switch (axis)
    {
      case Axis_1:
	return limit1;
      case Axis_2:
	return limit2;
    }

    throw runtime_error("SHOT602::atLimit(): unknown axis!");

    return false;
}

//! コントローラの状態を調べる．
/*!
  \param position1	第1軸の現在位置が返される
  \param position2	第2軸の現在位置が返される
  \param limit1		第1軸のリミットスイッチがONならtrue, OFFならfalseが返される
  \param limit2		第2軸のリミットスイッチがONならtrue, OFFならfalseが返される
  \return		このコントローラ
 */
bool
SHOT602::getStatus(int& position1, int& position2, bool& limit1, bool& limit2)
{
    *this << "Q:" << endl;

    char	s[256];
  // 第1軸の位置
    getline(s, sizeof(s), ',');
    position1 = (s[0] != '-' ? atoi(s) : -atoi(s + 1));

  // 第2軸の位置
    getline(s, sizeof(s), ',');
    position2 = (s[0] != '-' ? atoi(s) : -atoi(s + 1));

  // コマンドまたはパラメータのエラー
    getline(s, sizeof(s), ',');
    if (s[0] == 'X')
	throw runtime_error("SHOT602::getStatus(): command/parameter error!");

  // リミットスイッチの状態
    getline(s, sizeof(s), ',');
    switch (s[0])
    {
      case 'L':
	limit1 = true;
	limit2 = false;
	break;
      case 'M':
	limit1 = false;
	limit2 = true;
	break;
      case 'W':
	limit1 = limit2 = true;
	break;
      default:
	limit1 = limit2 = false;
	break;
    }

  // BUSY/READY状態
    getline(s, sizeof(s));
    return (s[0] == 'B');
}

/*
 *  速度設定
 */
//! 速度モードとパラメータを設定する．
/*!
  \param speed		低速/高速モード
  \param bottom1	第1軸の起動速度，0を与えるとデフォルト値に設定
  \param top1		第1軸の巡航速度，0を与えるとデフォルト値に設定
  \param duration1	第1軸の加減速時間，0を与えるとデフォルト値に設定
  \param bottom2	第2軸の起動速度，0を与えるとデフォルト値に設定
  \param top2		第2軸の巡航速度，0を与えるとデフォルト値に設定
  \param duration2	第2軸の加減速時間，0を与えるとデフォルト値に設定
*/
SHOT602&
SHOT602::setSpeed(Speed speed,
		  u_int bottom1, u_int top1, u_int duration1,
		  u_int bottom2, u_int top2, u_int duration2)
{
    const u_int	LowSpeedMin		= 1,
		LowSpeedMax		= 200,
		HighSpeedMin		= 50,
		HighSpeedMax		= 20000,
		DefaultBottomSpeed	= 500,
		DefaultTopSpeed		= 5000,
		DefaultDuration		= 200;

    if (speed == LowSpeed)
    {
	bottom1 = (bottom1 == 0		 ? DefaultBottomSpeed :
		   bottom1 < LowSpeedMin ? LowSpeedMin :
		   bottom1 > LowSpeedMax ? LowSpeedMax :
		   bottom1);
	top1	= (top1 == 0	      ? DefaultTopSpeed :
		   top1 < LowSpeedMin ? LowSpeedMin :
		   top1 > LowSpeedMax ? LowSpeedMax :
		   top1);
	bottom2 = (bottom2 == 0		 ? DefaultBottomSpeed :
		   bottom2 < LowSpeedMin ? LowSpeedMin :
		   bottom2 > LowSpeedMax ? LowSpeedMax :
		   bottom2);
	top2	= (top2 == 0	      ? DefaultTopSpeed :
		   top2 < LowSpeedMin ? LowSpeedMin :
		   top2 > LowSpeedMax ? LowSpeedMax :
		   top2);
    }
    else
    {
	bottom1 = (bottom1 == 0		  ? DefaultBottomSpeed :
		   bottom1 < HighSpeedMin ? HighSpeedMin :
		   bottom1 > HighSpeedMax ? HighSpeedMax :
		   bottom1);
	top1	= (top1 == 0	       ? DefaultTopSpeed :
		   top1 < HighSpeedMin ? HighSpeedMin :
		   top1 > HighSpeedMax ? HighSpeedMax :
		   top1);
	bottom2 = (bottom2 == 0		  ? DefaultBottomSpeed :
		   bottom2 < HighSpeedMin ? HighSpeedMin :
		   bottom2 > HighSpeedMax ? HighSpeedMax :
		   bottom2);
	top2	= (top2 == 0	       ? DefaultTopSpeed :
		   top2 < HighSpeedMin ? HighSpeedMin :
		   top2 > HighSpeedMax ? HighSpeedMax :
		   top2);
    }

    if (bottom1 > top1 || bottom2 > top2)
	throw runtime_error("SHOT602::setSpeed(): bottom speed must be lower than top speed!");
    
    if (duration1 == 0)
	duration1 = DefaultDuration;
    if (duration2 == 0)
	duration2 = DefaultDuration;

    *this << "D:" << (speed == LowSpeed ? 1 : 2)
	  << 'S' << bottom1 << 'F' << top1 << 'R' << duration1
	  << 'S' << bottom2 << 'F' << top2 << 'R' << duration2
	  << endl;

    return *this;
}

/*
 *  移動
 */
//! 指定した軸を減速停止する．
/*!
  \param axis	軸
  \return	このコントローラ　
*/
SHOT602&
SHOT602::stop(Axis axis)
{
    return putCommand(axis, 'L', "", "");
}
    
//! 全軸を非常停止する．
/*!
  \return	このコントローラ　
*/
SHOT602&
SHOT602::emergencyStop()
{
    *this << "L:E" << endl;

    return *this;
}
    
//! 指定した軸をjog動作させる．
/*!
  \param axis	軸
  \param dir	正方向ならtrue, 負方向ならfalse
  \param dir2	axisが #Axis_Both の場合の第2軸について，
		正方向ならtrue, 負方向ならfalse
  \return	このコントローラ　
*/
SHOT602&
SHOT602::jog(Axis axis, bool dir, bool dir2)
{
    putCommand(axis, 'J', (dir ? "+" : "-"), (dir2 ? "+" : "-"));
    *this << 'G' << endl;

    return *this;
}
    
//! 指定した軸を移動する．
/*!
  \param axis	軸
  \param val	移動量
  \param val2	axisが #Axis_Both の場合の第2軸の移動量
  \param block	移動が完了するまでリターンしないならtrue, 直ちにリターンするならfalse
  \return	このコントローラ
*/
SHOT602&
SHOT602::move(Axis axis, int val, int val2, bool block)
{
    putCommand(axis, 'M', "", "", false);

    if (val >= 0)
	*this << "+P" << val;
    else
	*this << "-P" << -val;

    if (axis == Axis_Both)
    {
	if (val2 >= 0)
	    *this << "+P" << val2;
	else
	    *this << "-P" << -val2;
    }

    *this << endl << 'G' << endl;
    if (block)
	while (isBusy())
	    ;
    
    return *this;
}

/*
 *  励磁
 */
//! 指定した軸のhold（励磁）/free（非励磁）を設定する．
/*!
  \param axis	軸
  \param on	励磁するならtrue, 励磁しないならfalse
  \param on2	axisが #Axis_Both の場合の第2軸について，
		励磁するならtrue, 励磁しないならfalse
  \return	このコントローラ
*/
SHOT602&
SHOT602::setHold(Axis axis, bool on, bool on2)
{
    return putCommand(axis, 'C', (on ? "1" : "0"), (on2 ? "1" : "0"));
}

/*
 *  private member functions
 */
SHOT602&
SHOT602::putCommand(Axis axis, char command,
		    const char* arg, const char* arg2, bool putDelimiter)
{
    *this << command;

    switch (axis)
    {
      case Axis_1:
	*this << ":1" << arg;
	break;
      case Axis_2:
	*this << ":2" << arg;
	break;
      default:
	*this << ":W" << arg << arg2;
	break;
    }

    if (putDelimiter)
	*this << endl;
    
    return *this;
}
    
}
