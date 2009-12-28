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
 *  $Id: PM16C_04.cc,v 1.1 2009-12-28 01:40:22 ueshiba Exp $
 */
#include <cstdlib>
#include <stdarg.h>
#include "TU/PM16C_04.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static void
checkChannel(u_int channel)
{
    if (channel >= 16)
	throw std::runtime_error("channel# must be less than 16!");
}
    
/************************************************************************
*  class PM16C_04							*
************************************************************************/
PM16C_04::PM16C_04(const char* ttyname, bool echo)
    :Serial(ttyname), _echo(echo)
{
    i_igncr()
	.o_nl2crnl()
#if !defined(__APPLE__)
	.o_lower2upper()
#endif
	.c_baud(9600).c_csize(8).c_noparity().c_stop1();

    setMode(true);
    usleep(DELAY);
}
    
void
PM16C_04::showId() const
{
    using namespace	std;

    put("VER?\n");
    Serial::get(_response, BUFFER_SIZE);
    cerr << _response;
}

/*
 *  Local/Remoteモード
 */
PM16C_04&
PM16C_04::setMode(bool remote)
{
    put(remote ? "S1R\n" : "S1L\n");

    return *this;
}

bool
PM16C_04::isRemoteMode() const
{
    return getHardwareLimitSwitchStatus(Axis_A) & 0x8;
}

/*
 *  位置
 */
PM16C_04&
PM16C_04::setPosition(u_int channel, int position)
{
    checkChannel(channel);
    put("S5%XPS%+08d\n", channel, position);
    usleep(DELAY);

    return *this;
}

int
PM16C_04::getPosition(u_int channel) const
{
    checkChannel(channel);
    return put("S4%XPS\n", channel).get<int>("%d");
}
    
/*
 *  スピード
 */
PM16C_04&
PM16C_04::setSpeed(Speed speed)
{
    switch (speed)
    {
      case Speed_Low:
	put("S34\n");
	break;
      case Speed_Medium:
	put("S35\n");
	break;
      default:
	put("S36\n");
	break;
    }

    return *this;
}

PM16C_04&
PM16C_04::setSpeedValue(u_int channel, Speed speed, u_int val)
{
    checkChannel(channel);

    switch (speed)
    {
      case Speed_Low:
	put("SPL%X%06d\n", channel, val);
	break;
      case Speed_Medium:
	put("SPM%X%06d\n", channel, val);
	break;
      default:
	put("SPH%X%06d\n", channel, val);
	break;
    }

    return *this;
}
    
u_int
PM16C_04::getSpeedValue(u_int channel, Speed speed) const
{
    checkChannel(channel);

    switch (speed)
    {
      case Speed_Low:
	return put("SPL?%X\n", channel).get<u_int>("%d");
      case Speed_Medium:
	return put("SPM?%X\n", channel).get<u_int>("%d");
    }
    return put("SPH?%X\n", channel).get<u_int>("%d");
}

/*
 *  ソフトウェアリミットスイッチ
 */
PM16C_04&
PM16C_04::enableSoftwareLimitSwitch(u_int channel,
				    int positionP, int positionN)
{
    setLimitSwitchConf(channel, getLimitSwitchConf(channel) | 0x20)
	.put("S5%XFL%+08d\n", channel, positionP)
	.put("S5%XBL%+08d\n", channel, positionN);
    usleep(DELAY);

    return *this;
}
    
PM16C_04&
PM16C_04::disableSoftwareLimitSwitch(u_int channel)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) & 0xdf);
}
    
bool
PM16C_04::isEnabledSoftwareLimitSwitch(u_int channel) const
{
    return getLimitSwitchConf(channel) & 0x20;
}

int
PM16C_04::getSoftwareLimitSwitchPositionP(u_int channel) const
{
    checkChannel(channel);
    return put("S4%XFL\n", channel).get<int>("%d");
}
    
int
PM16C_04::getSoftwareLimitSwitchPositionN(u_int channel) const
{
    checkChannel(channel);
    return put("S4%XBL\n", channel).get<int>("%d");
}
    
/*
 *  ハードウェアリミットスイッチ
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

PM16C_04&
PM16C_04::disableHardwareLimitSwitch(u_int channel, bool dir)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) &
				       (dir ? 0xf7 : 0xef));
}

bool
PM16C_04::isEnabledHardwareLimitSwitch(u_int channel, bool dir) const
{
    return getLimitSwitchConf(channel) & (dir ? 0x08 : 0x10);
}

bool
PM16C_04::getHardwareLimitSwitchPolarity(u_int channel, bool dir) const
{
    return getLimitSwitchConf(channel) & (dir ? 0x01 : 0x02);
}

PM16C_04&
PM16C_04::setHomeSwitchPolarity(u_int channel, bool normallyClose)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) | 0x04);
}

bool
PM16C_04::getHomeSwitchPolarity(u_int channel) const
{
    return getLimitSwitchConf(channel) & 0x04;
}

/*
 *  バックラッシュ補正
 */
PM16C_04&
PM16C_04::setBacklashCorrectionStep(u_int channel, u_int val)
{
    checkChannel(channel);
    put("B%X%+05d\n", channel, val);

    return *this;
}

u_int
PM16C_04::getBacklashCorrectionStep(u_int channel) const
{
    checkChannel(channel);
    return put("B%X?\n", channel).get<u_int>("%d");
}

/*
 *  Hold off機能（停止時の非通電）
 */
PM16C_04&
PM16C_04::enableHoldOff(u_int channel)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) & 0xbf);
}

PM16C_04&
PM16C_04::disableHoldOff(u_int channel)
{
    return setLimitSwitchConf(channel, getLimitSwitchConf(channel) | 0x40);
}

bool
PM16C_04::isEnabledHoldOff(u_int channel) const
{
    return getLimitSwitchConf(channel) & 0x40;
}

/*
 *  原点検出
 */
PM16C_04&
PM16C_04::setHomeSearchDirection(u_int channel, bool dir)
{
    checkChannel(channel);
    put("D%c%X\n", (dir ? 'P' : 'N'), channel);

    return *this;
}
    
bool
PM16C_04::getHomeSearchDirection(u_int channel) const
{
    return getHomeStatus(channel) & 0x1;
}
    
PM16C_04&
PM16C_04::setHomeOffset(u_int channel, u_int offset)
{
    checkChannel(channel);
    put("GF%X%04d\n", channel, offset);

    return *this;
}
    
u_int
PM16C_04::getHomeOffset(u_int channel) const
{
    checkChannel(channel);
    return put("GF?%X\n", channel).get<u_int>("%d");
}
    
bool
PM16C_04::isHomeFound(u_int channel) const
{
    return getHomeStatus(channel) & 0x4;
}
    
bool
PM16C_04::isHomeFoundFromFront(u_int channel) const
{
    return getHomeStatus(channel) & 0x2;
}

int
PM16C_04::getHomePosition(u_int channel) const
{
    checkChannel(channel);
    return put("HP?%X\n", channel).get<int>("%d");
}
    
PM16C_04&
PM16C_04::findHome(Axis axis)
{
    using namespace	std;

    const u_int	channel = getChannel(axis);
    if (!isEnabledHardwareLimitSwitch(channel, getHomeSearchDirection(channel)))
	throw runtime_error("PM16C_04::findHome(): hardware limit switch is disabled!");
    
    switch (axis)
    {
      case Axis_A:
	put("FHPA\n");
	break;
      case Axis_B:
	put("FHPB\n");
	break;
      case Axis_C:
	put("FHPC\n");
	break;
      default:
	put("FHPD\n");
	break;
    }

    while (isBusy(axis))
	;

    return *this;
}
    
PM16C_04&
PM16C_04::goHome(Axis axis)
{
    using namespace	std;

    if (!isHomeFound(getChannel(axis)))
	throw runtime_error("PM16C_04::goHome(): home position is not found yet!");
    
    switch (axis)
    {
      case Axis_A:
	put("RTHPA\n");
	break;
      case Axis_B:
	put("RTHPB\n");
	break;
      case Axis_C:
	put("RTHPC\n");
	break;
      default:
	put("RTHPD\n");
	break;
    }

    while (isBusy(axis))
	;
    
    return *this;
}
    
/*
 *  軸とチャンネルの関係
 */
PM16C_04&
PM16C_04::setChannel(Axis axis, u_int channel)
{
    checkChannel(channel);

    switch (axis)
    {
      case Axis_A:
	put("S11%X\n", channel);
	break;
      case Axis_B:
	put("S12%X\n", channel);
	break;
      case Axis_C:
	put("S15%X\n", channel);
	break;
      default:
	put("S16%X\n", channel);
	break;
    }
    usleep(DELAY);
    
    return *this;
}

u_int
PM16C_04::getChannel(Axis axis) const
{
    u_int	channel_A, channel_B, channel_C, channel_D;
    getChannel(channel_A, channel_B, channel_C, channel_D);

    switch(axis)
    {
      case Axis_A:
	return channel_A;
      case Axis_B:
	return channel_B;
      case Axis_C:
	return channel_C;
    }

    return channel_D;
}
    
void
PM16C_04::getChannel(u_int& channel_A, u_int& channel_B,
		     u_int& channel_C, u_int& channel_D) const
{
    u_int	channel = put("S10\n").get<u_int>("%x");
    channel_A = (channel >> 12) & 0xf;
    channel_B = (channel >>  8) & 0xf;
    channel_C = (channel >>  4) & 0xf;
    channel_D =  channel	& 0xf;
}

/*
 *  軸の状態
 */
int
PM16C_04::where(Axis axis) const
{
    switch (axis)
    {
      case Axis_A:
	return put("S20D\n").get<int>("%d");
      case Axis_B:
	return put("S22D\n").get<int>("%d");
      case Axis_C:
	return put("S24D\n").get<int>("%d");
    }
    return put("S26D\n").get<int>("%d");
}

bool
PM16C_04::isBusy(Axis axis) const
{
    return getControllerStatus(axis) & 0x1;
}

bool
PM16C_04::isPulseEmitted(Axis axis) const
{
    return getControllerStatus(axis) & 0x2;
}

bool
PM16C_04::isPulseStopped(Axis axis) const
{
    return getControllerStatus(axis) & 0x4;
}

bool
PM16C_04::atLimit(Axis axis, bool dir) const
{
    return !(getHardwareLimitSwitchStatus(axis) & (dir ? 0x1 : 0x2));
}

bool
PM16C_04::atHome(Axis axis) const
{
    return !(getHardwareLimitSwitchStatus(axis) & 0x4);
}

/*
 *  移動
 */
PM16C_04&
PM16C_04::stop(Axis axis)
{
    return move(axis, "40");
}
    
PM16C_04&
PM16C_04::jog(Axis axis, bool dir)
{
   return move(axis, (dir ? "08" : "09"));
}
    
PM16C_04&
PM16C_04::scanWithConstantSpeed(Axis axis, bool dir)
{
    return move(axis, (dir ? "0C" : "0D"));
}
    
PM16C_04&
PM16C_04::scan(Axis axis, bool dir)
{
    return move(axis, (dir ? "0E" : "0F"));
}
    
PM16C_04&
PM16C_04::pause(Axis axis, bool on)
{
    return move(axis, (on ? "16" : "17"));
}
    
PM16C_04&
PM16C_04::holdOff(Axis axis, bool set)
{
    return move(axis, (set ? "18" : "19"));
}
    
PM16C_04&
PM16C_04::scanAndStopAtHome(Axis axis, bool dir)
{
    return move(axis, (dir ? "1E" : "1F"));
}
    
PM16C_04&
PM16C_04::move(Axis axis, bool relative, int val, bool correctBacklash)
{
    const char	rel   = (relative ? 'R' : 'A');
    const char*	rmbkl = (correctBacklash ? "B" : "");
    
    switch (axis)
    {
      case Axis_A:
	put("S32%c%+08d%s\n", rel, val, rmbkl);
	break;
      case Axis_B:
	put("S33%c%+08d%s\n", rel, val, rmbkl);
	break;
      case Axis_C:
	put("S3A%c%+08d%s\n", rel, val, rmbkl);
	break;
      default:
	put("S3B%c%+08d%s\n", rel, val, rmbkl);
	break;
    }

    return *this;
}

/*
 *  Parallel I/Oポート
 */
PM16C_04&
PM16C_04::enableParallelIO()
{
    put("PIO\n");

    return *this;
}
    
PM16C_04&
PM16C_04::disableParallelIO()
{
    put("COM\n");

    return *this;
}
    
bool
PM16C_04::isEnabledParallelIO() const
{
    put("PIO?\n");
    Serial::get(_response, BUFFER_SIZE);
    return !strncmp(_response, "PIO", 3);
}
    
u_int
PM16C_04::readParallelIO() const
{
    return put("RD\n").get<u_int>("%x");
}
    
PM16C_04&
PM16C_04::writeParallelIO(u_int val)
{
    put("W%04X", val & 0xffff);

    return *this;
}
    
/*
 *  private member functions
 */
u_int
PM16C_04::getLimitSwitchConf(u_int channel) const
{
    checkChannel(channel);
    return put("S4%XD\n", channel).get<u_int>("%x") >> 16;
}

PM16C_04&
PM16C_04::setLimitSwitchConf(u_int channel, u_int conf)
{
    checkChannel(channel);
    put("S5%XD%02X\n", channel, conf);
    usleep(DELAY);

    return *this;
}

u_int
PM16C_04::getHardwareLimitSwitchStatus(Axis axis) const
{
    u_int	status = put("S6\n").get<u_int>("%x");

    switch (axis)
    {
      case Axis_A:
	return (status >>  8) & 0xf;
      case Axis_B:
	return (status >> 12) & 0xf;
      case Axis_C:
	return status & 0xf;
    }
    return (status >> 4) & 0xf;
}
    
u_int
PM16C_04::getHomeStatus(u_int channel) const
{
    checkChannel(channel);
    return put("G?%X\n", channel).get<u_int>("%x");
}
    
u_int
PM16C_04::getControllerStatus(Axis axis) const
{
    switch (axis)
    {
      case Axis_A:
	return put("S21\n").get<u_int>("%x");
      case Axis_B:
	return put("S23\n").get<u_int>("%x");
      case Axis_C:
	return put("S25\n").get<u_int>("%x");
    }
    return put("S27\n").get<u_int>("%x");
}
    
PM16C_04&
PM16C_04::move(Axis axis, const char* motion)
{
    switch (axis)
    {
      case Axis_A:
	put("S30%s\n", motion);
	break;
      case Axis_B:
	put("S31%s\n", motion);
	break;
      case Axis_C:
	put("S38%s\n", motion);
	break;
      default:
	put("S39%s\n", motion);
	break;
    }

    return *this;
}
    
const PM16C_04&
PM16C_04::put(const char* format, ...) const
{
    va_list	args;
    va_start(args, format);
    if (_echo)
	vfprintf(stderr, format, args);
    vfprintf(fp(), format, args);
    va_end(args);

    return *this;
}
    
}
