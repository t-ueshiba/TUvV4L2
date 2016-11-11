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
    IIDCCamera::Speed	speed	   = IIDCCamera::SPD_400M;
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
#if 1
	IIDCCameraArray	cameras(cameraName, configDirs, speed, ncameras);
#else
	IIDCCameraArray	cameras;
	cin >> cameras;
#endif
	for (auto camera : cameras)
	    cerr << hex << camera->globalUniqueId() << endl;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
