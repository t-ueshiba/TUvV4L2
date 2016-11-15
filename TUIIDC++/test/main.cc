/*
 *  $Id$
 */
#include "TU/IIDC++.h"

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    try
    {
	TU::IIDCCamera	cameras[2];

	
	for (auto& camera : cameras)
	{
	    camera.initialize();
	    std::cout << camera;
	}
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }
    
    return 0;
}
