/*
 *  $Id: Manus.cc,v 1.2 2002-07-25 02:38:01 ueshiba Exp $
 */
#include "TU/Can++.h"

namespace TU
{
/************************************************************************
*  static data								*
************************************************************************/
static const char *messages[] =
{
    "No messages.",						//  0

    "Warning: stuck gripper.",					//  1
    "Warning: wrong area.",					//  2
    "Warning: arm folded stretched.",				//  3
    "Warning: blocked DOF.",					//  4
    "Warning: maximum M1 rotation.",				//  5

    "General: folded.",						//  6
    "General: unfolded.",					//  7
    "General: gripper ready initialising.",			//  8
    "General: absolute measuring ready.",			//  9

    "Error:   I/O 80c522 error.",				// 10
    "Error:   absolute encoder error.",				// 11
    "Error:   move without user input error.",			// 12
    "Error:   unknown error."					// 13
};

/************************************************************************
*  static functions							*
************************************************************************/
inline int
toInt(u_char hi, u_char lo)
{
    return int(short((hi << 8) | lo));
}

inline u_int
toUint(u_char hi, u_char lo)
{
    return u_int((hi << 8) | lo);
}

inline int
limit(int val, int min, int max)
{
    return (val < min ? min : val > max ? max : val);
}

inline int
sign(int val)
{
    return (val > 0 ? 1 : val < 0 ? -1 : 0);
}

inline int
abs(int val)
{
    return (val > 0 ? val : -val);
}

static bool
isZero(const Manus::Speed& speed)
{
    for (int i = 0; i < speed.dim(); ++i)
	if (speed[i] != 0)
	    return false;
    return true;
}

/************************************************************************
*  class Manus								*
************************************************************************/
//! Manusマニピュレータノードを生成する
/*!
  マニピュレータは#STILLモードに初期化される．
  \param dev	Manusマニピュレータのデバイス名(ex. /dev/can0)
*/
Manus::Manus(const char* dev)
    :Can(dev), _mode(STILL), _status(OK), _pos(), _upDown(0), _speed()
{
    stillMode();
}

//! 折り畳まれているマニピュレータを拡げる
/*!
  拡げ終わると，台座は静止に，すべての軸の速度指令値は0にそれぞれリセット
  されて自動的に#JOINTモードになる．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::foldOut()
{
    _mode = FOLD_OUT;
    _upDown = 0;
    _speed = 0;
    do
    {
	tick();
    } while (_status != WRONG_AREA && _status != UNFOLDED);
    
    return jointMode();
}

//! マニピュレータを格納位置に折り畳む
/*!
  折り畳みが完了すると，台座は静止に，すべての軸の速度指令値は0にそれぞれ
  リセットされて自動的に#JOINTモードになる．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::foldIn()
{
    Position	ref;
    ref[0] =  -590;
    ref[1] =   610;
    ref[2] =  -550;
    ref[3] = -1550;
    ref[4] =  1040;
    ref[5] = -1155;
    ref[6] = 25000;
    jointMode().moveTo(ref);
    
    _mode = FOLD_IN;
    _upDown = 0;
    _speed = 0;
    do
    {
	tick();
    } while (_status != WRONG_AREA && _status != FOLDED);
    
    return jointMode();
}

//! マニピュレータを#STILLモードに移行する
/*!
  台座は静止に，すべての軸の速度指令値は0にそれぞれリセットされる．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::stillMode()
{
    _mode = STILL;
    _upDown = 0;
    _speed = 0;
    return tick();
}

//! マニピュレータを#CARTESIANモードに移行する
/*!
  移行に失敗した場合は，#JOINTモードに移行する．
  台座は静止に，すべての軸の速度指令値は0にそれぞれリセットされる．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::cartesianMode()
{
    _mode = CARTESIAN;
    _upDown = 0;
    _speed = 0;
    tick();
    if (_status == WRONG_AREA)
	jointMode();
    return *this;
}

//! マニピュレータを#JOINTモードに移行する
/*!
  台座は静止に，すべての軸の速度指令値は0にそれぞれリセットされる．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::jointMode()
{
    _mode = JOINT;
    _upDown = 0;
    _speed = 0;
    return tick();
}

//! 台座を上昇させるように設定する
/*!
  すべての軸の速度指令値は0にリセットされる．
  次のtick()の呼び出しで実際に台座が上昇する．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::setBaseUp()
{
    _upDown = -1;
    _speed = 0;
    return *this;
}

//! 台座を静止させるように設定する
/*!
  すべての軸の速度指令値は0にリセットされる．
  次のtick()の呼び出しで実際に台座が静止する．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::setBaseStill()
{
    _upDown = 0;
    _speed = 0;
    return *this;
}

//! 台座を下降させるように設定する
/*!
  すべての軸の速度指令値は0にリセットされる．
  次のtick()の呼び出しで実際に台座が下降する．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::setBaseDown()
{
    _upDown = 1;
    _speed = 0;
    return *this;
}

//! 速度指令値を設定する
/*!
  #CARTESIAN/#JOINTモードの場合は，指定された速度値が#SpeedLimits
  の範囲に収まらなければ，収まるように最大/最小値が設定される．
  #CARTESIAN/#JOINTモード以外の場合は，速度値は0に設定される．どちらの場合も
  台座は静止にリセットされる．次のtick()の呼び出しで実際に速度指令がマニ
  ピュレータに送られる．
  \param speed	速度指令値．#CARTESIANモードの場合は各座標軸の速度．そうでない
		場合は関節角速度．
  \return	このManusマニピュレータオブジェクト．
*/
Manus&
Manus::setSpeed(const Speed& speed)
{
    _upDown = 0;
    switch (_mode)
    {
      case CARTESIAN:
	for (int i = 0; i < 3; ++i)
	    _speed[i] = limit(speed[i],
			      -MAX_SPEED_CART_XYZ, MAX_SPEED_CART_XYZ);
	for (int i = 3; i < 6; ++i)
	    _speed[i] = limit(speed[i],
			      -MAX_SPEED_CART_YPR, MAX_SPEED_CART_YPR);
	_speed[6] = limit(speed[6], -MAX_SPEED_CART_GRIP, MAX_SPEED_CART_GRIP);
        break;
      case JOINT:
	for (int i = 0; i < 3; ++i)
	    _speed[i] = limit(speed[i],
			      -MAX_SPEED_JOINT_012, MAX_SPEED_JOINT_012);
	for (int i = 3; i < 6; ++i)
	    _speed[i] = limit(speed[i],
			      -MAX_SPEED_JOINT_345, MAX_SPEED_JOINT_345);
	_speed[6] = limit(speed[6],
			  -MAX_SPEED_JOINT_GRIP, MAX_SPEED_JOINT_GRIP);
        break;
      default:
	_speed = 0;
	break;
    }
    return *this;
}

//! マニピュレータの制御ループを1回だけ回す
/*!
  マニピュレータの現在位置を読み込む．読み込んだ現在位置は，position()で知る
  ことができる．さらに，#CARTESIAN/#JOINTモードの
  場合は，現在の速度指令値をマニピュレータに送る．
  \return		このManusマニピュレータオブジェクト．
  \exception Error	マニピュレータにエラーが生じた．
*/
Manus&
Manus::tick()
{
    for (int n = 0; n < 3; ++n)
	switch (receive())
	{
	  case 0x350:	// status, message and first 3 coordinates/angles.
	    _status = toStatus(get(0), get(1));
	    if ((_status & 0xc0) == 0xc0)
		throw Error(_status);
	    _pos[0] = toInt(get(2), get(3));
	    _pos[1] = toInt(get(4), get(5));
	    _pos[2] = toInt(get(6), get(7));
	    break;

	  case 0x360:	// last 4 coordinates/angles.
	    _pos[3] = toInt(get(0), get(1));
	    _pos[4] = toInt(get(2), get(3));
	    _pos[5] = toInt(get(4), get(5));
	    _pos[6] = toUint(get(6), get(7));
	    break;

	  case 0x37f:	// prompt for selecting control box.
	    setId(_mode);
	    if (_mode == CARTESIAN || _mode == JOINT)
	    {
		put(_upDown);
		for(int i = 0; i < _speed.dim(); ++i)
		    put(_speed[i]);
	    }
	    send();
	    break;
	}

    return *this;
}

//! マニピュレータの台座を最高点まで上げる
/*!
  #CARTESIANまたは#JOINTモードでのみ有効．
  \return			このManusマニピュレータオブジェクト．
  \exception std::runtime_error	#CARTESIANまたは#JOINTモードでない．
*/
Manus&
Manus::baseUp()
{
    if (_mode != CARTESIAN && _mode == JOINT)
	throw std::runtime_error("TU::Manus::up(): Neighter CARTESIAN nor JOINT mode!!");
    
    setBaseUp();
    for (int n = 0; n < 700; ++n)
	tick();

    return *this;
}

//! マニピュレータの台座を最低点まで下げる
/*!
  #CARTESIANまたは#JOINTモードでのみ有効．
  \return			このManusマニピュレータオブジェクト．
  \exception std::runtime_error	#CARTESIANまたは#JOINTモードでない．
*/
Manus&
Manus::baseDown()
{
    if (_mode != CARTESIAN && _mode == JOINT)
	throw std::runtime_error("Manus::down(): Neighter CARTESIAN nor JOINT mode!!");
    
    setBaseDown();
    for (int n = 0; n < 700; ++n)
	tick();

    return *this;
}

//! マニピュレータを目標位置まで動かす
/*!
  #CARTESIANまたは#JOINTモードでのみ有効．
  \param ref			目標位置
  \return			このManusマニピュレータオブジェクト．
  \exception std::runtime_error	#CARTESIANまたは#JOINTモードでない．
*/
Manus&
Manus::moveTo(const Position& ref)
{
  /*    for (int i = _pos.dim(); --i >= 0; )
    {
	Speed	speed;
	for (;;)
	{
	    setSpeed(speed).tick();
	    speed[i] = ref[i] - _pos[i];
	    cerr << "    " << i << "-th axis: "
		 << ref[i] << " - " << _pos[i] << " = " << speed[i] << endl;
	    if (abs(speed[i]) < 10)
		break;
	    speed[i] = sign(speed[i]);
	}
	}*/

    Speed	speed;
    for (;;)
    {
	setSpeed(speed).tick();

	speed = ref - _pos;

	cerr << " moveTo: position =" << _pos
	     << "         error    =" << speed;
	
	int	i;
	for (i = 0; i < 6; ++i)
	    if (abs(speed[i]) >= 20)
		break;
	if (i == 6)
	    break;
	
	for (i = 0; i < speed.dim(); ++i)
	    speed[i] = sign(speed[i]);
    } 
    
    return *this;
}

//! マニピュレータの状態をメッセージ文字列に変換する
/*!
  \param status	マニピュレータの状態．
  \return	メッセージ文字列．
*/
const char*
Manus::message(Status status)
{
    switch (status)
    {
      case OK:
	return messages[0];

      // Warnings.
      case STUCK_GRIPPER:
	return messages[1];
      case WRONG_AREA:
	return messages[2];
      case ARM_FOLDED_STRETCHED:
	return messages[3];
      case BLOCKED_DOF:
	return messages[4];
      case MAXIMUM_M1_ROTATION:
	return messages[5];
	
      // General messages.
      case FOLDED:
	return messages[6];
      case UNFOLDED:
	return messages[7];
      case GRIPPER_REDAY_INITIALISING:
	return messages[8];
      case ABSOLUTE_MEASURING_READY:
	return messages[9];

      // Errors.
      case IO_80C552_ERROR:
	return messages[10];
      case ABSOLUTE_ENCODER_ERROR:
	return messages[11];
      case MOVE_WITHOUT_USER_INPUT_ERROR:
	return messages[12];
    }
    return messages[13];
}

Manus::Status
Manus::toStatus(u_char hi, u_char lo)
{
    hi &= 0xc0;

    if (hi == 0)
	return OK;
    
    switch (hi | lo)
    {
      // Warnings.
      case STUCK_GRIPPER:
	return STUCK_GRIPPER;
      case WRONG_AREA:
	return WRONG_AREA;
      case ARM_FOLDED_STRETCHED:
	return ARM_FOLDED_STRETCHED;
      case BLOCKED_DOF:
	return BLOCKED_DOF;
      case MAXIMUM_M1_ROTATION:
	return MAXIMUM_M1_ROTATION;
	
      // General messages.
      case FOLDED:
	return FOLDED;
      case UNFOLDED:
	return UNFOLDED;
      case GRIPPER_REDAY_INITIALISING:
	return GRIPPER_REDAY_INITIALISING;
      case ABSOLUTE_MEASURING_READY:
	return ABSOLUTE_MEASURING_READY;

      // Errors.
      case IO_80C552_ERROR:
	return IO_80C552_ERROR;
      case ABSOLUTE_ENCODER_ERROR:
	return ABSOLUTE_ENCODER_ERROR;
      case MOVE_WITHOUT_USER_INPUT_ERROR:
	return MOVE_WITHOUT_USER_INPUT_ERROR;
    }
    
    return UNKNOWN_ERROR;
}

}
#ifdef __GNUG__
#  include "TU/Geometry++.cc"

namespace TU
{
template
CoordBase<int, 7u>&	CoordBase<int, 7u>::operator =(double);
}

#endif
