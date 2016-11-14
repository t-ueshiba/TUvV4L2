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

	for (const auto& camera : cameras)
	    std::cerr << camera;
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }
    
    return 0;
}
