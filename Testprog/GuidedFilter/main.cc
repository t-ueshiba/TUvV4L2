/*
 *  $Id: main.cc,v 1.1 2012-07-23 00:45:48 ueshiba Exp $
 */
#include <cstdlib>
#include "TU/GuidedFilter.h"

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    size_t		w = 3;
    float		e = 0.01;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "w:e:")) != -1; )
	switch (c)
	{
	  case 'w':
	    w = atoi(optarg);
	    break;
	  case 'e':
	    e = atof(optarg);
	    break;
	}
     
  // guided filterを2つの1D arrayに適用する．
    Array<int>	a;
    cerr << ">> ";
    cin >> a;

    Array<u_char>	b(a.size());
    for (u_int i = 0; i < b.size(); ++i)
	b[i] = a[i]; //+ 1;

    Array<float>	c(a.size());
  /*guidedFilter(a.begin(), a.end() + 1 - w,
    b.begin(), b.end() + 1 - w, c.begin(), w, e);*/
    guidedFilter(a.begin(), a.end() + 1 - w, c.begin(), w, e);
    cout << c;

  // guided filterを2つの2D arrayに適用する．
    Array2<Array<short> >	A;
    cerr << ">> ";
    cin >> A;

    Array2<Array<u_char> >	B(A.nrow(), A.ncol());
    for (u_int i = 0; i < B.nrow(); ++i)
	for (u_int j = 0; j < B.ncol(); ++j)
	    B[i][j] = A[i][j];// + 1;

    Array2<Array<float> >	C(A.nrow(), A.ncol());
  //guidedFilter2(A.begin(), A.end() + 1 - w,
  //	  B.begin(), B.end() + 1 - w, C.begin(), w, w, e);
    guidedFilter2(A.begin(), A.end() + 1 - w, C.begin(), w, w, e);
    cout << C;

    return 0;
}
