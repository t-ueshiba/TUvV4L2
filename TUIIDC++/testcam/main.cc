/*
 *  $Id: main.cc,v 1.4 2012-08-13 07:15:07 ueshiba Exp $
 */
#include <cstdlib>
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
    IIDCCamera::Speed	speed = IIDCCamera::SPD_400M;
    u_int64_t		uniqId = 0;
    
  // Parse command options.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "b")) != -1; )
	switch (c)
	{
	  case 'b':
	    speed = IIDCCamera::SPD_800M;
	    break;
	}
    
    extern int	optind;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
  // Main job.
    try
    {
	IIDCCamera	camera(uniqId);
	camera.setSpeed(speed);
	
	v::MyCmdWindow<IIDCCamera, u_char>	myWin(vapp, camera);
	vapp.run();

	std::cout << camera;
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
