/*
 *  $Id: main.cc,v 1.1 2002-07-25 04:36:10 ueshiba Exp $
 */
#include <cstdlib>
#include <iostream>
#include <cmath>
#include "TU/Random.h"

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    int		histogram[100];
    const int	m = sizeof(histogram)/sizeof(histogram[0]);
    for (int i = 0; i < m; ++i)
	histogram[i] = 0;

    int		n = (argc == 1 ? 10000 : atoi(argv[1]));
    Random	random;
    for (int i = 0; i < n; ++i)
    {
	double	val = random.gaussian();
	int	index = floor(10.0*val) + m/2;
	if (index >= 0 && index < m)
	    ++histogram[index];
    }
    for (int i = 0; i < m; ++i)
	cout << 0.1 * (i - m/2) << ' ' << histogram[i] << endl;
    
    return 0;
    
}
