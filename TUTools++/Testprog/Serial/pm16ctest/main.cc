/*
 *  $Id: main.cc,v 1.4 2012-08-15 07:58:34 ueshiba Exp $
 */
#include <unistd.h>
#include <cstdlib>
#include <iomanip>
#include <exception>
#include "TU/PM16C_04.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static PM16C_04::Axis
char2axis(char c)
{
    switch (c)
    {
      case 'b':
      case 'B':
	return PM16C_04::Axis_B;
      case 'c':
      case 'C':
	return PM16C_04::Axis_C;
      case 'd':
      case 'D':
	return PM16C_04::Axis_D;
    }

    return PM16C_04::Axis_A;
}

static PM16C_04::Speed
char2speed(char c)
{
    switch (c)
    {
      case 'l':
      case 'L':
	return PM16C_04::Speed_Low;
      case 'm':
      case 'M':
	return PM16C_04::Speed_Medium;
    }

    return PM16C_04::Speed_High;
}

static void
putPrompt(PM16C_04::Axis axis, u_int channel, PM16C_04::Speed speed)
{
    using namespace	std;
    
    switch (axis)
    {
      case PM16C_04::Axis_A:
	cerr << "(axis-A,";
	break;
      case PM16C_04::Axis_B:
	cerr << "(axis-B,";
	break;
      case PM16C_04::Axis_C:
	cerr << "(axis-C,";
	break;
      default:
	cerr << "(axis-D,";
	break;
    }

    cerr << "ch-" << channel << ',';
    
    switch (speed)
    {
      case PM16C_04::Speed_Low:
	cerr << "spd-L)>> ";
	break;
      case PM16C_04::Speed_Medium:
	cerr << "spd-M)>> ";
	break;
      default:
	cerr << "spd-H)>> ";
	break;
    }
}

}

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    const char*		ttyname	= "/dev/ttyS0";
    PM16C_04::Axis	axis	= PM16C_04::Axis_A;
    extern char*	optarg;
    
    for (int c; (c = getopt(argc, argv, "d:ABCD")) != -1; )
	switch (c)
	{
	  case 'd':
	    ttyname = optarg;
	    break;
	  case 'A':
	    axis = PM16C_04::Axis_A;
	    break;
	  case 'B':
	    axis = PM16C_04::Axis_B;
	    break;
	  case 'C':
	    axis = PM16C_04::Axis_C;
	    break;
	  case 'D':
	    axis = PM16C_04::Axis_D;
	    break;
	}

    try
    {
	PM16C_04	stage(ttyname);
	u_int		channel = stage.getChannel(axis);
	PM16C_04::Speed	speed	= PM16C_04::Speed_Medium;

	stage.disableHoldOff(channel);
	
	stage.enableHardwareLimitSwitch(channel, true, true)
	    .enableHardwareLimitSwitch(channel, false, true)
	    .setHomeSwitchPolarity(channel, false);
	
	stage.enableSoftwareLimitSwitch(channel, 2000, -300000);
	stage.disableSoftwareLimitSwitch(channel);
	cerr << " Software limit switch: "
	     << stage.getSoftwareLimitSwitchPositionP(channel)
	     << ", " << stage.getSoftwareLimitSwitchPositionN(channel)
	     << ": " << (stage.isEnabledSoftwareLimitSwitch(channel) ?
			 "enaled." : "disabled")
	     << endl;

	cerr << " Current position: " << stage.getPosition(channel) << endl;

	putPrompt(axis, channel, speed);
	char	command[128];
	while (cin >> command)
	{
	    switch (command[0])
	    {
	      case 'V':
		stage.showId(cout);
		break;
		
	      // 操作対象となる軸をセット
	      case 'X':
		axis = char2axis(command[1]);
		channel = stage.getChannel(axis);
		break;

	      // 軸とチャンネルの関係づけ
	      case 'C':
		if (command[1] != '\0')
		{
		    channel = atoi(command + 1);
		    stage.setChannel(axis, channel);
		}
		else
		{
		    u_int	channel_A, channel_B, channel_C, channel_D;
		    stage.getChannel(channel_A, channel_B,
				     channel_C, channel_D);
		    cout <<  " ch-A: " << channel_A
			 << ", ch-B: " << channel_B
			 << ", ch-C: " << channel_C
			 << ", ch-D: " << channel_D
			 << endl;
		}
		break;

	      // スピード
	      case 'L':
	      case 'M':
	      case 'H':
		speed = char2speed(command[0]);
		stage.setSpeed(speed);
		break;
	      case 'S':
		if (command[1] == '\0')
		    cout << " speed: "
			 << stage.getSpeedValue(channel, speed)
			 << endl;
		else
		    stage.setSpeedValue(channel, speed,
					atoi(command + 1));
		break;

	      // 状態の問い合わせ
	      case 'w':
		if (command[1] == '\0')
		    cout << " position: " << stage.where(axis)
			 << endl;
		else
		    stage.setPosition(channel, atoi(command + 1));
		break;

	      // 原点検出
	      case 'h':
		stage.setHomeSearchDirection(channel, (command[1] == 'f'));
		stage.findHome(axis);
		cout << " Home offset: " << stage.getHomeOffset(channel)
		     << ", Home position: " << stage.getHomePosition(channel)
		     << endl;
		stage.setPosition(channel, 0);
		break;
	      case 'g':
		stage.goHome(axis);
		cout << " Home offset: " << stage.getHomeOffset(channel)
		     << ", Home position: " << stage.getHomePosition(channel)
		     << endl;
		stage.setPosition(channel, 0);
		break;
		
	      // 移動
	      case 's':
		if (command[1] == '\0')
		    stage.stop(axis);
		else
		    stage.scan(axis, command[1] == 'f');
		break;
	      case 'a':
		stage.move(axis, false, atoi(command + 1), false);
		break;
	      case 'A':
		stage.move(axis, false, atoi(command + 1), true);
		break;
	      case 'r':
		stage.move(axis, true,  atoi(command + 1), false);
		break;
	      case 'R':
		stage.move(axis, true,  atoi(command + 1), true);
		break;

	      // バックラッシュ除去
	      case 'B':
		if (command[1] == '\0')
		    cout << " Backlash correction step: "
			 << stage.getBacklashCorrectionStep(channel)
			 << endl;
		else
		    stage.setBacklashCorrectionStep(channel, atoi(command + 1));
		break;

	      case '?':
	      default:
		cout << "Commands.\n"
		     << "  V:       show firmware version.\n"
		     << "  X<axis>: make <axis> current.\n"
		     << "  C<ch>:   attach <ch> to the current axis.\n"
		     << "  L:       make low speed current.\n"
		     << "  M:       make medium speed current.\n"
		     << "  H:       make high speed current.\n"
		     << "  S:       show value of the current speed.\n"
		     << "  S<val>:  set <val> to the current speed.\n"
		     << "  w:       show position of the current axis.\n"
		     << "  w<val>:  set <val> to the position of the current channel.\n"
		     << "  h[fb]:   find home position from front/back.\n"
		     << "  g:       go back to home.\n"
		     << "  s:       stop.\n"
		     << "  s[fb]:   scan forward/backward.\n"
		     << "  a<pos>   move to <pos>.\n"
		     << "  A<pos>   move to <pos> with backlash removal.\n"
		     << "  r<inc>   move by <inc>.\n"
		     << "  R<inc>   move by <inc> with backlash removal.\n"
		     << "  B:       show backlash correction step.\n"
		     << "  B<val>:  set <val> to the backlash correction step.\n"
		     << endl;
		break;
	    }

	    putPrompt(axis, channel, speed);
	}
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
    }
	
    return 0;
}
