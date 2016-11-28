/*
 *  $Id: main.cc,v 1.2 2012-08-13 07:13:12 ueshiba Exp $
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
    
    v::App	vapp(argc, argv);
    
  // Main job.
    try
    {
	extern int		optind;
	Array<V4L2Camera>	cameras(argc - optind);
	for (auto& camera : cameras)
	    camera.initialize(argv[optind++]);

	if (cameras.size() == 0)
	    throw std::runtime_error("One or more cameras must be specified!!");

	v::MyCmdWindow<Array<V4L2Camera>, u_char>	myWin(vapp, cameras);
	vapp.run();

	std::cout << cameras;
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
