#include "TU/Array++.h"

namespace TU
{
template <class T> std::ostream&
operator <<(std::ostream& out , const Array2<T>& a)
{
    for (int i = 0; i < a.dim(); ++i)
	out << a[i];
    return out;
}
	
}

main()
{
    using namespace	std;
    using namespace	TU;
    
    Array2<Array<int> >	a(3, 2);
    
    a[0][0] = 1;
    a[0][1] = 2;
    a[1][0] = 10;
    a[1][1] = 20;
    a[2][0] = 100;
    a[2][1] = 200;

    cout << a;
    
    Array2<Array<int> >	b(a);
//    Array2<int, Array<int> >	b;
//    b = a;
    
    a[1][0] = 11;
    a[1][1] = 21;

    cout << a;
    cout << b;
}
