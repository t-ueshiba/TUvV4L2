/*
 *  $Id$
 */
#include "TU/tuple.h"

namespace TU
{
void
doJob()
{
    using tuple_type = std::tuple<int, float, double>;

    constexpr size_t	N = 1000;
    tuple_type		x[N];
    
    for (size_t i = 0; i < N; ++i)
	x[i] = std::make_tuple(i, i + 0.11, i + 0.222);
    
    tuple_type	y;
    
    for (size_t n = 0; n < 1000000; ++n)
    {
	y = std::make_tuple(0, 0.0, 0.0);
	
	for (size_t i = 0; i < N; ++i)
	    y += x[i];
    }

    std::cout << y << std::endl;
}
}


int
main()
{
    TU::doJob();
    
    return 0;
}
