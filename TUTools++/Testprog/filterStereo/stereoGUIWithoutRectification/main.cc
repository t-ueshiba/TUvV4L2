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

    bool		color = false;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "c")) != -1; )
	switch (c)
	{
	  case 'c':
	    color = true;
	    break;
	}
	
  // 本当のお仕事．
    try
    {
	v::App	vapp(argc, argv);

	if (color)
	{
	    v::MyCmdWindow<RGB, float>		myWin(vapp);
	    vapp.run();
	}
	else
	{
	    v::MyCmdWindow<u_char, float>	myWin(vapp);
	    vapp.run();
	}
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
