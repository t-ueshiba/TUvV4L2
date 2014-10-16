/*
 *  $Id: main.cc,v 1.4 2012-08-13 07:15:07 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
#include "TU/v/vIeee1394++.h"
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
    Ieee1394Node::Speed	speed = Ieee1394Node::SPD_400M;
    u_int		delay = 1;

  // Parse command line.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "bd:")) != EOF; )
	switch (c)
	{
	  case 'b':
	    speed = Ieee1394Node::SPD_800M;
	    break;
	  case 'd':
	    delay = atoi(optarg);
	    break;
	}
    extern int	optind;
    u_int64_t	uniqId = 0;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
  // Main job.
    try
    {
	Ieee1394Camera	camera(Ieee1394Camera::Monocular, uniqId, speed, delay);

	v::MyCmdWindow<Ieee1394Camera, u_char>	myWin(vapp, camera);
	vapp.run();

	camera.stopContinuousShot();

	cerr << "0x" << hex << setw(16) << setfill('0')
	     << camera.globalUniqueId() << dec << ' ' << camera;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
