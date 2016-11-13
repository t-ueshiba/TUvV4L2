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
#if 0
#  if 1
	IIDCCameraArray	cameras(cameraName, configDirs, speed, ncameras);
#  else
	IIDCCameraArray	cameras;
	cin >> cameras;
#  endif
	for (auto camera : cameras)
	    cerr << hex << camera->globalUniqueId() << endl;
#else
	uint64_t	uniqId0 = 0x00b09d0100f908b2;
	uint64_t	uniqId1 = 0x00b09d0100f908b3;
	IIDCCamera	cameras[] = {{IIDCCamera::Monocular, uniqId0},
				     {IIDCCamera::Monocular, uniqId1}};
	for (const auto& camera : cameras)
	    cerr << hex << camera.globalUniqueId() << endl;
#endif
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
