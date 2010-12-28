/*
 *  $Id: main.cc,v 1.3 2010-12-28 11:47:48 ueshiba Exp $
 */
#include <unistd.h>
#include <stdlib.h>
#include "MyCmdWindow.h"
#include "TU/Ieee1394++.h"
#include <iomanip>

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    v::App		vapp(argc, argv);
    const char*		triggerDev = "/dev/ttyS0";
    bool		i1394b = false;
    u_int		delay = 1;

  // Parse command line.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "bd:t:")) != EOF; )
	switch (c)
	{
	  case 'b':
	    i1394b = true;
	    break;
	  case 'd':
	    delay = atoi(optarg);
	    break;
	  case 't':
	    triggerDev = optarg;
	    break;
	}
    extern int	optind;
    u_int64_t	uniqId = 0;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
    try
    {
#if defined(UseTrigger)
	TriggerGenerator	trigger(triggerDev);
#endif
	Ieee1394Camera		camera(Ieee1394Camera::Monocular,
				       i1394b, uniqId, delay);

	v::MyCmdWindow	myWin(vapp, camera
#if defined(UseTrigger)
			      , trigger
#endif
			     );
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
