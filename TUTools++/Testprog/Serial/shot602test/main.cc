/*
 *  $Id: main.cc,v 1.2 2010-12-17 00:53:43 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
#include <exception>
#include "TU/SHOT602.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static SHOT602::Axis
char2axis(char c)
{
    switch (c)
    {
      case '1':
	return SHOT602::Axis_1;
      case '2':
	return SHOT602::Axis_2;
      case 'w':
      case 'W':
	return SHOT602::Axis_Both;
    }

    return SHOT602::Axis_1;
}

static void
putPrompt(SHOT602::Axis axis, SHOT602::Speed speed)
{
    using namespace	std;
    
    switch (axis)
    {
      case SHOT602::Axis_1:
	cerr << "(axis-1,";
	break;
      case SHOT602::Axis_2:
	cerr << "(axis-2,";
	break;
      default:
	cerr << "(axis-W,";
	break;
    }

    switch (speed)
    {
      case SHOT602::LowSpeed:
	cerr << "spd-L)>> ";
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
    SHOT602::Axis	axis	= SHOT602::Axis_1;
    extern char*	optarg;
    
    for (int c; (c = getopt(argc, argv, "d:12W")) != EOF; )
	switch (c)
	{
	  case 'd':
	    ttyname = optarg;
	    break;
	  case '1':
	    axis = SHOT602::Axis_1;
	    break;
	  case '2':
	    axis = SHOT602::Axis_2;
	    break;
	  case 'W':
	    axis = SHOT602::Axis_Both;
	    break;
	}

    try
    {
	SHOT602		stage(ttyname);
	SHOT602::Speed	speed = SHOT602::HighSpeed;
	
	putPrompt(axis, speed);
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
		break;

	      // スピードをセット
	      case 'L':
	      case 'H':
	      {
		speed = (command[0] == 'L' ? SHOT602::LowSpeed
					   : SHOT602::HighSpeed);
		u_int	top = (isdigit(command[1]) ? atoi(command + 1) : 0);
		stage.setSpeed(speed, 0, top, 0, 0, top, 0);
	      }
	        break;
	      
	      // 位置の問い合わせ
	      case 'w':
		cout << " position: " << stage.where(axis)
		     << endl;
		break;

	      // 原点検出
	      case 'h':
		stage.findHome(axis, (command[1] == '+'));
		break;
		
	      // 移動
	      case 's':
		stage.stop(axis);
		break;
	      case 'e':
		stage.emergencyStop();
		break;
	      case 'j':
		stage.jog(axis, (command[1] != '-'));
		break;
	      case 'r':
		stage.move(axis, atoi(command + 1));
		break;
	      case 'R':
		stage.move(axis, atoi(command + 1), 0, true);
		break;

	      case '?':
	      default:
		cout << "Commands.\n"
		     << "  V:       show firmware version.\n"
		     << "  X<axis>: make <axis> current.\n"
		     << "  w:       show position of the current axis.\n"
		     << "  h[+-]:   find home position from front/back.\n"
		     << "  s:       stop.\n"
		     << "  j[+-]:   jog.\n"
		     << "  r<inc>   move by <inc>.\n"
		     << "  R<inc>   move by <inc> and wait completion.\n"
		     << endl;
		break;
	    }

	    putPrompt(axis, speed);
	}
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
    }
	
    return 0;
}
