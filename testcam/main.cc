/*
 *  $Id$
 */
#include <cstdlib>
#include "TU/v/vV4L2++.h"
#include "MyCmdWindow.h"

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	TU;
    
    v::App		vapp(argc, argv);
    const char*		dev = "/dev/video0";

  // Parse command line.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "d:")) != -1; )
	switch (c)
	{
	  case 'd':
	    dev = optarg;
	    break;
	}

  // Main job.
    try
    {
	V4L2Camera				camera(dev);
	v::MyCmdWindow<V4L2Camera, RGB>		myWin(vapp, camera);
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
