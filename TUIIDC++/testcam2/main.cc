/*
 *  $Id: main.cc,v 1.2 2012-08-13 07:13:12 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
#include "TU/v/vIIDC++.h"
#include "MyCmdWindow.h"

namespace TU
{
/************************************************************************
*  class CameraArray							*
************************************************************************/
class CameraArray : public Array<IIDCCamera*>
{
  public:
    CameraArray(char* argv[], int argc, IIDCCamera::Speed speed)	;
    ~CameraArray()							;
};

CameraArray::CameraArray(char* argv[], int argc, IIDCCamera::Speed speed)
    :Array<IIDCCamera*>(argc)
{
    for (size_t i = 0; i < size(); ++i)
	(*this)[i] = new IIDCCamera(strtoull(argv[i], 0, 0));
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
    IIDCCamera::Speed	speed = IIDCCamera::SPD_400M;
    
  // Parse command options.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "b")) != EOF; )
	switch (c)
	{
	  case 'b':
	    speed = IIDCCamera::SPD_800M;
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
	CameraArray	cameras(argv + optind, argc - optind, speed);
	v::MyCmdWindow<IIDCCamera, u_char>	myWin(vapp, cameras);
	vapp.run();

	for (auto camera : cameras)
	    cout << *camera;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
