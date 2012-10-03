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

    GuidedFilter<float>	gf(w, e);
    Array<float>	c(a.size());
    gf.convolve(a.begin(), a.end(), b.begin(), b.end(), c.begin() + w - 1);
  //gf.convolve(a.begin(), a.end(), c.begin() + w - 1);
    cout << c;
  /*
  // guided filterを2つの2D arrayに適用する．
    Array2<Array<short> >	A;
    cerr << ">> ";
    cin >> A;

    Array2<Array<u_char> >	B(A.nrow(), A.ncol());
    for (u_int i = 0; i < B.nrow(); ++i)
	for (u_int j = 0; j < B.ncol(); ++j)
	    B[i][j] = A[i][j];// + 1;

    GuidedFilter2<float>	gf2(w, w, e);
    Array2<Array<float> >	C(A.nrow(), A.ncol());
  //gf2.convolve(A.begin(), A.end(), B.begin(), B.end(), C.begin());
    gf2.convolve(A.begin(), A.end(), C.begin());
    cout << C;
  */
    return 0;
}
