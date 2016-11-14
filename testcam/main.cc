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
    
    v::App	vapp(argc, argv);
    u_int64_t	uniqId = 0;
    extern int	optind;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
  // Main job.
    try
    {
	IIDCCamera	camera(uniqId);

	v::MyCmdWindow<IIDCCamera, u_char>	myWin(vapp, camera);
	vapp.run();

	camera.continuousShot(false);

	cerr << camera;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
