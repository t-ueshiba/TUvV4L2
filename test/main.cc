/*
 *  $Id$
 */
#include <cstdlib>
#include "TU/IIDCCameraArray.h"

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    const char*		cameraName = "BlackFly";
    const char*		configDirs = nullptr;
    int			ncameras   = -1;

    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "c:d:")) != -1; )
	switch (c)
	{
	  case 'c':
	    cameraName = optarg;
	    break;
	  case 'd':
	    configDirs = optarg;
	    break;
	}
	
    try
    {
	IIDCCamera	cameras[2];

	for (const auto& camera : cameras)
	    cerr << camera;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
