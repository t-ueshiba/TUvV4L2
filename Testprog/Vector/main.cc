/*
 *  $Id: mtest.cc,v 1.4 2010-03-03 01:36:57 ueshiba Exp $
 */
#include "TU/Vector++.h"

int
main()
{
    using namespace	std;
    using namespace	TU;
    
    Vector<double>	u, v;
    Matrix<double>	A, B;
    Matrix<float>	F;

//    cout << setprecision(8);
    cout.setf(ios::scientific, ios::floatfield);

    for (;;)
    {
	cerr << "\n i: inner product, addition, subtraction"
	     << "\n o: outer product"
	     << "\n p: partial vector"
	     << "\n I: inverse matrix"
	     << "\n G: generalized inverse matrix"
	     << "\n A: adjoint matrix"
	     << "\n E: eigen vector & eigen value"
	     << "\n S: sigular value decomposition"
	  //	     << "\n C: matrix type conversion"
	     << "\n P: partial matrix"
	     << "\n T: tridiagonalize matrix"
	     << "\n B: bidiagonalize matrix"
	     << "\n Q: QR decomposition of a matrix"
	     << "\n L: Cholesky decomposition of a matrix"
	     << "\nSelect function >> ";
	char	c;
	if (!(cin >> c))
	    break;
	cin.get();	// discard '\n'

	try {	
	switch (c)
	{
	  case 'i':
	    cerr << " u >> ";
	    cin >> u;
	    cerr << " v >> ";
	    cin >> v;
	    cout << "u   = " << u
		 << "v   = " << v
		 << "u.v = " << u * v << '\n'
		 << "u+v = " << u + v
		 << "u-v = " << u - v << endl;
	    break;

	  case 'o':
	    cerr << " u >> ";
	    cin >> u;
	    cerr << " v >> ";
	    cin >> v;
	    cout << "u   = " << u
		 << "v   = " << v
		 << "u^v = " << (u ^ v) << endl;
	    break;

	  case 'p':
	  {
	    cerr << " u >> ";
	    cin >> u;
	    cerr << " i >> ";
	    int i;
	    cin >> i;
	    cerr << " d >> ";
	    int d;
	    cin >> d;
	    cout << "u = " << u
		 << "u(" << i << ", " << d << ") = " << u(i, d) << endl;
	  }
	    break;

	  case 'I':
	    cerr << " A >> ";
	    cin >> A;
	    B = A.inv();
	    cout << "--- A ---\n"	    << A
		 << "--- A.inv() ---\n"	    << B
		 << "--- A * A.inv() ---\n" << A * B << endl;
	    break;

	  case 'G':
	    cerr << " A >> ";
	    cin >> A;
	    B = A.pinv();
	    cout << "--- A ---\n"		<< A
		 << "--- A.pinv(...) --\n"	<< B
		 << "--- A * A.pinv(...) ---\n" << A * B
		 << "--- A.pinv(...) * A ---\n" << B * A << endl;
	    break;

	  case 'A':
	    cerr << " A >> ";
	    cin >> A;
	    B = A.adj();
	    cout << "--- A ---\n"	    << A
		 << "--- A.adj() ---\n"	    << B
		 << "--- A * A.adj() ---\n" << A * B << endl;
	    break;

	  case 'E':
	    cerr << " A >> ";
	    cin >> A;
	    B = A.eigen(u, true);
	    cout << "--- A ---\n"	   << A
		 << "--- Ut * U ---\n"	   << B * B.trns()
		 << "--- Ut * A * U ---\n" << B * A * B.trns()
		 << "  Eigen-values = "	   << u << endl;
	    break;

	  case 'S':
	  {
	    cerr << " A >> ";
	    cin >> A;
	    SVDecomposition<double> svd(A);
	    cout << "--- A ---\n"	   << A
		 << "--- Ut * U ---\n"	   << svd.Ut() * svd.Ut().trns()
		 << "--- Vt * V ---\n"	   << svd.Vt() * svd.Vt().trns()
		 << "--- Vt * A * U ---\n" << svd.Vt() * A * svd.Ut().trns()
		 << "  Singular-values = " << svd.diagonal() << endl;
	  }
	    break;

	  /*	  case 'C':
	    cerr << " A >> ";
	    cin >> A;
	    F = A;
	    cout << "--- A ---\n" << A
		 << "--- F ---\n" << F << endl;
		 break;*/

	  case 'P':
	  {
	    cerr << " A >> ";
	    cin >> A;
	    cerr << " i >> ";
	    int i;
	    cin >> i;
	    cerr << " j >> ";
	    int j;
	    cin >> j;
	    cerr << " r >> ";
	    int r;
	    cin >> r;
	    cerr << " c >> ";
	    int c;
	    cin >> c;
	    cout << "--- A ---\n" << A
		 << "--- A(" << i << ", " << j
		 << ", " << r << ", " << c << ") ---\n"
		 << A(i, j, r, c) << endl;
	  }
	    break;  
	    
	  case 'T':
	  {
	    cerr << " A >> ";
	    cin >> A;
	    TriDiagonal<double> tri(A);
	    cout << "--- A ---\n"	   << A
		 << "--- Ut * U ---\n"	   << tri.Ut() * tri.Ut().trns()
		 << "--- Ut * A * U ---\n" << tri.Ut() * A * tri.Ut().trns()
		 << "  Diagonal     = "	   << tri.diagonal()
		 << "  Off-diagonal = "	   << tri.off_diagonal() << endl;
	  }
	    break;
	    
	  case 'B':
	  {
	    cerr << " A >> ";
	    cin >> A;
	    BiDiagonal<double> bi(A);
	    cout << "--- A ---\n"	   << A
		 << "--- Ut * U ---\n"	   << bi.Ut() * bi.Ut().trns()
		 << "--- Vt * V ---\n"	   << bi.Vt() * bi.Vt().trns()
		 << "--- Vt * A * U ---\n" << bi.Vt() * A * bi.Ut().trns()
		 << "  Diagonal     = "	   << bi.diagonal()
		 << "  Off-diagonal = "	   << bi.off_diagonal() << endl;
	  }
	    break;

	  case 'Q':
	  {
	      cerr << " A >> ";
	      cin >> A;
	      QRDecomposition<double> qr(A);
	      cout << "--- A ---\n"	 << A
		   << "--- Qt * Q ---\n" << qr.Qt() * qr.Qt().trns()
		   << "--- Rt ---\n"	 << qr.Rt()
		   << "--- Rt * Qt ---\n" << qr.Rt() * qr.Qt() << endl;
	  }
	    break;

	  case 'L':
	  {
	      cerr << " A >> ";
	      cin >> A;
	      B = A.cholesky();
	      cout << "--- A ---\n"	 << A
		   << "--- Lt ---\n"	 << B
		   << "--- L * Lt ---\n" << B.trns() * B << endl;
	  }
	    break;
	}
	}
	catch (std::exception& err)
	{
	    cerr << err.what() << endl;
	}
    }

	return 0;
}
