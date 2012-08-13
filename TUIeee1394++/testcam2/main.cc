/*
 *  $Id: main.cc,v 1.2 2012-08-13 07:13:12 ueshiba Exp $
 */
#include <unistd.h>
#include <stdlib.h>
#include <iomanip>
#include <stdexcept>
#include "TU/Ieee1394CameraArray.h"
#include "MyCmdWindow.h"

namespace TU
{
/************************************************************************
*  class CameraArray							*
************************************************************************/
class CameraArray : public Array<Ieee1394Camera*>
{
  public:
    CameraArray(char* argv[], int argc,
		Ieee1394Node::Speed speed, u_int delay)			;
    ~CameraArray()							;
};

CameraArray::CameraArray(char* argv[], int argc,
			 Ieee1394Node::Speed speed, u_int delay)
    :Array<Ieee1394Camera*>(argc)
{
    for (int i = 0; i < dim(); ++i)
	(*this)[i] = new Ieee1394Camera(Ieee1394Camera::Monocular,
					strtoull(argv[i], 0, 0), speed, delay);
}

CameraArray::~CameraArray()
{
    for (int i = 0; i < dim(); ++i)
	delete (*this)[i];
}

}
/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    v::App		vapp(argc, argv);
    char*		triggerDev = "/dev/ttyS0";
    Ieee1394Node::Speed	speed = Ieee1394Node::SPD_400M;
    u_int		delay = 1;
    bool		sync = false;
    
  // Parse command options.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "bd:t:s")) != EOF; )
	switch (c)
	{
	  case 'b':
	    speed = Ieee1394Node::SPD_800M;
	    break;
	  case 'd':
	    delay = atoi(optarg);
	    break;
	  case 't':
	    triggerDev = optarg;
	    break;
	  case 's':
	    sync = true;
	    break;
	}
    
    extern int		optind;
    if (argc - optind == 0)
    {
	cerr << "One or more cameras must be specified!!" << endl;
	return 1;
    }
    
  // Main job.
    try
    {
	CameraArray		cameras(argv + optind, argc - optind,
					speed, delay);
#ifdef UseTrigger
	TriggerGenerator	trigger(triggerDev);
#endif
	for (int i = 0; i < cameras.dim(); ++i)
	    cerr << "camera " << i << ": uniqId = "
		 << hex << setw(16) << setfill('0')
		 << cameras[i]->globalUniqueId() << dec << endl;
	
	v::MyCmdWindow	myWin(vapp, cameras, sync
#ifdef UseTrigger
			      , trigger
#endif
			     );
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
