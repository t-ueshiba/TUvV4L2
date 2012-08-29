/*
 *  $Id: Can++.h,v 1.4 2012-08-29 21:16:49 ueshiba Exp $
 */
#ifndef __TUCanPP_h
#define __TUCanPP_h

#include <sys/types.h>
#include <can4linux.h>
#include "TU/Vector++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class Can								*
************************************************************************/
/*!
  CAN(Control Area Network)のノードを表すクラス．
*/
class Can
{
  public:
  //! 通信速度
    enum Baud
    {
	B10k	=   10,		//!< 10k baud
	B20k	=   20,		//!< 20k baud
	B40k	=   40,		//!< 40k baud
	B50k	=   50,		//!< 50k baud
	B100k	=  100,		//!< 100k baud
	B125k	=  125,		//!< 125k baud
	B250k	=  250,		//!< 250k baud
	B500k	=  500,		//!< 500k baud
	B800k	=  800,		//!< 800k baud
	B1000k	= 1000,		//!< 1000k baud
    };
    
  public:
    Can(const char* dev)				;
    ~Can()						;

    Can&		setBaud(Baud baud)		;
    
  protected:
    u_long		nreceive()			;
    u_long		receive()			;
    
  //! メッセージのIDを返す
  /*!
    \return 現在読み込まれているメッセージのID．*/
    u_long		id()			const	{return _msg.id;}
  //! メッセージに含まれるデータのバイト数を返す
  /*!
    \return 現在読み込まれているメッセージに含まれるデータのバイト数．*/
    u_int		nbytes()		const	{return _msg.length;}
    u_char		get(u_int i)		const	;
    
    Can&		setId(u_long id)		;
    Can&		put(u_char c)			;
    const Can&	send()			const	;
    
  private:
    Can(const Can&)					;
    Can&		operator =(const Can&)	;
    
    const int	_fd;
    canmsg_t	_msg;
};

/************************************************************************
*  class Manus								*
************************************************************************/
/*!
  Manusマニピュレータ(Exact Dynamics社)を表すクラス．
*/
class Manus : public Can
{
  public:
  //! Manusの現在位置
    typedef Vector<int, FixedSizedBuf<int, 7> >	Position;
  //! Manusへの速度指令値
    typedef Vector<int, FixedSizedBuf<int, 7> >	Speed;

  //! Manusの動作モード
    enum Mode
    {
	STILL	  = 0x370,	//!< control box 0: startup/initialization.
	CARTESIAN = 0x371,	//!< control box 1: cartesian control.
	JOINT	  = 0x374,	//!< control box 4: joint control.
	FOLD_OUT  = 0x375,	//!< control box 5: folding out.
	FOLD_IN	  = 0x376	//!< control box 6: folding in.
    };

  //! Manusの状態
    enum Status
    {
	OK			= 0x00,	//!< 正常

      // Warnings.
	STUCK_GRIPPER		= 0x40,	//!< グリッパが障害物に衝突
	WRONG_AREA		= 0x41,	//!< 変な姿勢からcartesian/foldに移行
	ARM_FOLDED_STRETCHED	= 0x42,	//!< アームが延びきった
	BLOCKED_DOF		= 0x43,	//!< 過負荷/衝突
	MAXIMUM_M1_ROTATION	= 0x44,	//!< 回転角の限度を越えた

      // General messages.
	FOLDED				= 0x80,	//!< fold状態
	UNFOLDED			= 0x81,	//!< unfold状態
	GRIPPER_REDAY_INITIALISING	= 0x82,	//!< gripper ready
	ABSOLUTE_MEASURING_READY	= 0x83,	//!< cartesian mode ready

      // Errors.
	IO_80C552_ERROR			= 0xc1,	//!< user I/Oのエラー
	ABSOLUTE_ENCODER_ERROR		= 0xc4,	//!< エンコーダのエラー
	MOVE_WITHOUT_USER_INPUT_ERROR	= 0xcf,	//!< 入力がないのに動いた
	UNKNOWN_ERROR			= 0xc5	//!< その他のエラー
    };

  //! Manusへの速度指令の最大値
    enum SpeedLimits
    {
	MAX_SPEED_CART_XYZ	= 127,	//!< xyz軸の最大速度(cartesian mode)
	MAX_SPEED_CART_YPR	=  10,	//!< ypr軸の最大速度(cartesian mode)
	MAX_SPEED_CART_GRIP	=  15,	//!< グリッパの最大速度(cartesian mode)
	MAX_SPEED_JOINT_012	=  10,	//!< 012軸の最大速度(joint mode)
	MAX_SPEED_JOINT_345	=  10,	//!< 345軸の最大速度(joint mode)
	MAX_SPEED_JOINT_GRIP	=  15	//!< グリッパの最大速度(joint mode)
    };

  //! cartesianモードでの各軸の座標値の最大/最小値
    enum CartesianLimits
    {
	MIN_CART_XYZ	=   -720,	//!< xyz軸の最小値
	MAX_CART_XYZ	=    720,	//!< xyz軸の最大値
	MIN_CART_YAW	=  -1800,	//!< yaw軸の最小値
	MAX_CART_YAW	=   1800,	//!< yaw軸の最大値
	MIN_CART_PITCH	=   -900,	//!< pitch軸の最小値
	MAX_CART_PITCH	=    900,	//!< pitch軸の最大値
	MIN_CART_ROLL	=  -1800,	//!< roll軸の最小値
	MAX_CART_ROLL	=   1800,	//!< roll軸の最大値
	MIN_CART_GRIP	=  28100,	//!< grip軸の最大値
	MAX_CART_GRIP	=  54000	//!< grip軸の最大値
    };
    
  //! jointモードでの各軸の座標値の最大/最小値
    enum JointLimits
    {
	MIN_JOINT_012	=  -1800,	//!< 第012軸の最小値
	MAX_JOINT_012	=   1800,	//!< 第012軸の最大値
	MIN_JOINT_3	=  -1800,	//!< 第3軸(yaw)の最小値
	MAX_JOINT_3	=   1800,	//!< 第3軸(yaw)の最大値
	MIN_JOINT_4	=      0,	//!< 第4軸の最小値
	MAX_JOINT_4	=   1266,	//!< 第4軸の最大値
	MIN_JOINT_5	=  -1800,	//!< 第5軸(roll)の最小値
	MAX_JOINT_5	=   1800,	//!< 第5軸(roll)の最大値
	MIN_JOINT_GRIP	=  28100,	//!< grip軸の最小値
	MAX_JOINT_GRIP	=  54000	//!< grip軸の最大値
    };

  //! Manusのエラー
    class Error : public std::runtime_error
    {
      public:
      //! エラーオブジェクトを生成する
      /*!
	\param stat エラーを表すManusの状態変数. */
	Error(Status stat)
	    :std::runtime_error(Manus::message(stat)), status(stat)	{}
	
      //! エラーを表すManusの状態変数
	const Status	status;
    };
    
  public:
    Manus(const char* dev)				;

  //! 現在のマニピュレータの動作モードを返す
  /*!
    \return 現在のモード．*/
    Mode		mode()			const	{return _mode;}
  //! 現在のマニピュレータの状態を返す
  /*!
    \return 現在の状態．*/
    Status		status()		const	{return _status;}
			operator bool()		const	;
  //! 現在のマニピュレータの位置を返す
  /*!
    \return 現在の位置．#CARTESIANモードの場合はハンドのcatesian座標．
    そうでない場合は関節座標．*/
    const Position&	position()		const	{return _pos;}
  //! 現在のマニピュレータへの速度指令値を返す
  /*!
    \return 現在の速度指令値．#CARTESIANモードの場合は各座標軸の速度．
    そうでない場合は関節角速度．*/
    const Speed&	speed()			const	{return _speed;}

    Manus&		foldOut()			;
    Manus&		foldIn()			;
    Manus&		stillMode()			;
    Manus&		cartesianMode()			;
    Manus&		jointMode()			;
    Manus&		setBaseUp()			;
    Manus&		setBaseStill()			;
    Manus&		setBaseDown()			;
    Manus&		setSpeed(const Speed& speed)	;
    Manus&		tick()				;
    Manus&		baseUp()			;
    Manus&		baseDown()			;
    Manus&		moveTo(const Position& ref)	;

    static const char*	message(Status status)		;
    
  private:
    static Status	toStatus(u_char hi, u_char low)	;
    
    Mode	_mode;
    Status	_status;
    Position	_pos;		// current cartesian coordinates/joint angles.
    int		_upDown;	// up: -1, down: 1, still: 0.
    Speed	_speed;		// speed given to the manipulator.
};

//! 現在のマニピュレータがwarningもしくはerror状態にないか調べる
/*!
  \return	warning状態にもerror状態にもなければtrueを返す．
		そうでなければfalseを返す．
*/
inline
Manus::operator bool() const
{
    return (_status & 0x40 ? true : false);
}
 
}
#endif	/* !__TUCanPP_h	*/
