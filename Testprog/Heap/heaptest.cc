/*
 *  $Id: heaptest.cc,v 1.2 2008-09-02 05:13:16 ueshiba Exp $
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

main()
{
    using namespace	std;
    using namespace	TU;
    
    Array<int>	a;
    cerr << ">> ";
    cin >> a;

    TU::sort(a, ordered());
    cout << a;
}
