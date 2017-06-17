/*
 *  $Id: main.cc 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include <cstdlib>
#include "MyCmdWindow.h"

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

  // 本当のお仕事．
    try
    {
	v::App				vapp(argc, argv);
	v::MyCmdWindow<u_char, float>	myWin(vapp);
      //v::MyCmdWindow<RGB, float>	myWin(vapp);
	vapp.run();

    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
