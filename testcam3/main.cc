/*
 *  $Id: main.cc,v 1.4 2012-08-13 07:13:18 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
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
    IIDCCamera::Type	type = IIDCCamera::Binocular;
    IIDCCamera::Speed	speed = IIDCCamera::SPD_400M;

  // Parse command line.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "b123")) != -1; )
	switch (c)
	{
	  case 'b':
	    speed = IIDCCamera::SPD_800M;
	    break;
	  case '1':
	    type = IIDCCamera::Monocular;
	    break;
	  case '2':
	    type = IIDCCamera::Binocular;
	    break;
	}
    extern int	optind;
    u_int64_t	uniqId = 0;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
    try
    {
	IIDCCamera		camera(type, uniqId, speed);

	cerr << "0x" << hex << setw(16) << setfill('0')
	     << camera.globalUniqueId() << dec << endl;
	
	v::MyCmdWindow<u_char>	myWin(vapp, camera, type);
	vapp.run();

	camera.continuousShot(false);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
