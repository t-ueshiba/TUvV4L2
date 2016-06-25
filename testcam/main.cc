/*
 *  $Id: main.cc,v 1.4 2012-08-13 07:15:07 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
#include "TU/v/vIIDC++.h"
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
    IIDCCamera::Speed	speed = IIDCCamera::SPD_400M;

  // Parse command line.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "b")) != EOF; )
	switch (c)
	{
	  case 'b':
	    speed = IIDCCamera::SPD_800M;
	    break;
	}
    extern int	optind;
    u_int64_t	uniqId = 0;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
  // Main job.
    try
    {
	IIDCCamera	camera(IIDCCamera::Monocular, uniqId, speed);

	v::MyCmdWindow<IIDCCamera, u_char>	myWin(vapp, camera);
	vapp.run();

	camera.continuousShot(false);

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
