/*
 *  $Id: main.cc,v 1.4 2012-08-13 07:15:07 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
#include "TU/v/vIIDC++.h"
#include "MyCmdWindow.h"

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	TU;
    
    v::App		vapp(argc, argv);
    int			speed = 400;
    u_int64_t		uniqId = 0;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "s:")) != -1; )
	switch (c)
	{
	  case 's':
	    speed = atoi(optarg);
	    break;
	}
    
    extern int	optind;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
  // Main job.
    try
    {
	IIDCCamera	camera(uniqId);

	switch (speed)
	{
	  case 100:
	    camera.setSpeed(IIDCCamera::SPD_100M);
	    break;
	  case 200:
	    camera.setSpeed(IIDCCamera::SPD_200M);
	    break;
	  default:
	    camera.setSpeed(IIDCCamera::SPD_400M);
	    break;
	  case 800:
	    camera.setSpeed(IIDCCamera::SPD_800M);
	    break;
	  case 1600:
	    camera.setSpeed(IIDCCamera::SPD_1_6G);
	    break;
	  case 3200:
	    camera.setSpeed(IIDCCamera::SPD_3_2G);
	    break;
	}
	
	v::MyCmdWindow<IIDCCamera, RGB>	myWin(vapp, camera);
	vapp.run();

	camera.continuousShot(false);

	std::cout << camera;
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
