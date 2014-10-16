/*
 *  $Id: main.cc,v 1.2 2012-08-13 07:13:12 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
#include "TU/v/vIeee1394++.h"
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
    for (size_t i = 0; i < size(); ++i)
	(*this)[i] = new Ieee1394Camera(Ieee1394Camera::Monocular,
					strtoull(argv[i], 0, 0), speed, delay);
}

CameraArray::~CameraArray()
{
    for (size_t i = 0; i < size(); ++i)
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
    Ieee1394Node::Speed	speed = Ieee1394Node::SPD_400M;
    u_int		delay = 1;
    
  // Parse command options.
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
    
    extern int		optind;
    if (argc - optind == 0)
    {
	cerr << "One or more cameras must be specified!!" << endl;
	return 1;
    }
    
  // Main job.
    try
    {
	CameraArray	cameras(argv + optind, argc - optind, speed, delay);
	for (size_t i = 0; i < cameras.dim(); ++i)
	    cerr << "camera " << i << ": uniqId = "
		 << hex << setw(16) << setfill('0')
		 << cameras[i]->globalUniqueId() << dec << endl;
	
	v::MyCmdWindow<Ieee1394Camera, u_char>	myWin(vapp, cameras);
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
