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
	TU::Array<TU::IIDCCamera>	cameras;

	std::cin >> cameras;
	std::cout << cameras;
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }
    
    return 0;
}
