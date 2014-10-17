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
    
    v::App			vapp(argc, argv);
    Ieee1394Camera::Type	type = Ieee1394Camera::Binocular;
    Ieee1394Node::Speed		speed = Ieee1394Node::SPD_400M;
    u_int			delay = 1;

  // Parse command line.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "bd:123")) != -1; )
	switch (c)
	{
	  case 'b':
	    speed = Ieee1394Node::SPD_800M;
	    break;
	  case 'd':
	    delay = atoi(optarg);
	    break;
	  case '1':
	    type = Ieee1394Camera::Monocular;
	    break;
	  case '2':
	    type = Ieee1394Camera::Binocular;
	    break;
	}
    extern int	optind;
    u_int64_t	uniqId = 0;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
    try
    {
	Ieee1394Camera		camera(type, uniqId, speed, delay);

	cerr << "0x" << hex << setw(16) << setfill('0')
	     << camera.globalUniqueId() << dec << endl;
	
	v::MyCmdWindow<u_char>	myWin(vapp, camera, type);
	vapp.run();

	camera.stopContinuousShot();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
