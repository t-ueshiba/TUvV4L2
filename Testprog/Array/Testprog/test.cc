#include "TU/Array++.h"

int
main()
{
    TU::Array<float>	f(10), f1;

    f = 0;
    f[0] = 0; f[1] = 1; f[2] = 2;
    f1 = f;
    
    std::cout << f1;

    return 0;
}
