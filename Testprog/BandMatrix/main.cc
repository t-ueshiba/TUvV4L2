/*
 *  $Id$
 */
#include "TU/BandMatrix++.h"

int
main()
{
    using namespace	std;
    using namespace	TU;
    
    BandMatrix<double, 1, 2>	A(5);
    A(0, 0) = 10; A(0, 1) = 11; A(0, 2) = 12;
    A(1, 0) = 19; A(1, 1) = 20; A(1, 2) = 21; A(1, 3) = 22;
    A(2, 1) = 29; A(2, 2) = 30; A(2, 3) = 31; A(2, 4) = 32;
    A(3, 2) = 39; A(3, 3) = 40; A(3, 4) = 41;
    A(4, 3) = 49; A(4, 4) = 50;
    
    cout << A << A.L() << A.U();
    
    cout << "*********" << endl;
    
    A.decompose();
    cout << A.L() << A.U() << A.L() * A.U();

    
    cerr << ">> ";
    for (Vector<double> b; cin >> b; )
    {
	A.substitute(b);
	cout << b << b * A.L() * A.U();
	cerr << ">> ";
    }
    
    return 1;
}
