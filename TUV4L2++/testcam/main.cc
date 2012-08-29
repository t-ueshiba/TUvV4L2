/*
 *  $Id: main.cc,v 1.3 2012-06-29 03:10:04 ueshiba Exp $
 */
#include <cstdlib>
#include "MyCmdWindow.h"

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
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
    
    try
    {
	V4L2Camera	camera(dev);
	
	v::MyCmdWindow	myWin(vapp, camera);
	vapp.run();

	cout << camera;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}