/*
 *  $Id: heaptest.cc,v 1.4 2010-10-07 00:26:10 ueshiba Exp $
 */
#include "TU/Heap.h"

namespace TU
{
struct ordered
{
    int
    operator ()(const int& item0, const int& item1) const
    {
	return item0 < item1;
    }
};
}

int
main()
{
    using namespace	std;
    using namespace	TU;
    
    Array<int>	a;
    cerr << ">> ";
    cin >> a;

    TU::sort(a, std::less<int>());
  //TU::sort(a, ordered());
    cout << a;

    return 0;
}
