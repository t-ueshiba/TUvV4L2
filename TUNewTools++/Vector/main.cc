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
	     << "\n P: partial matrix"
	     << "\n I: inverse matrix"
	     << "\n A: adjoint matrix"
	     << "\n L: Cholesky decomposition of a matrix"
	     << "\n Q: QR decomposition of a matrix"
	     << "\n T: tridiagonalize matrix"
	     << "\n E: eigen vector & eigen value"
	     << "\n B: bidiagonalize matrix"
	     << "\n S: sigular value decomposition"
	     << "\n G: generalized inverse matrix"
	  //	     << "\n C: matrix type conversion"
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
		 << "u+v = " << evaluate(u + v)
		 << "u-v = " << evaluate(u - v)
		 << endl;
	    break;

	  case 'o':
	    cerr << " u >> ";
	    cin >> u;
	    cerr << " v >> ";
	    cin >> v;
	    cout << "u   = " << u
		 << "v   = " << v
		 << "u^v = " << (u ^ v)
		 << endl;
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
		 << "u(" << i << ", " << d << ") = " << u(i, d)
		 << endl;
	  }
	    break;

	  case 'P':
	  {
	    cerr << " A >> ";
	    cin >> A;
	    cerr << " i >> ";
	    int i;
	    cin >> i;
	    cerr << " r >> ";
	    int r;
	    cin >> r;
	    cerr << " j >> ";
	    int j;
	    cin >> j;
	    cerr << " c >> ";
	    int c;
	    cin >> c;
	    cout << "--- A ---\n" << A
		 << "--- A(" << i << ", " << r
		 << ", " << j << ", " << c << ") ---\n"
		 << A(i, r, j, c)
		 << endl;
	  }
	    break;  
	    
	  case 'I':
	    cerr << " A >> ";
	    cin >> A;
	    B = inverse(A);
	    cout << "--- A ---\n"	       << A
		 << "--- inverse(A) ---\n"     << B
		 << "--- A * inverse(A) ---\n" << evaluate(A * B)
		 << endl;
	    break;

	  case 'A':
	    cerr << " A >> ";
	    cin >> A;
	    B = adjoint(A);
	    cout << "--- A ---\n"	       << A
		 << "--- adjoint(A) ---\n"     << B
		 << "--- A * adjoint(A) ---\n" << evaluate(A * B)
		 << endl;
	    break;

	  case 'L':
	  {
	      cerr << " A >> ";
	      cin >> A;
	      B = cholesky(A);
	      cout << "--- A ---\n"	 << A
		   << "--- Lt ---\n"	 << B
		   << "--- L * Lt ---\n" << evaluate(transpose(B) * B)
		   << endl;
	  }
	    break;

	  case 'Q':
	  {
	      cerr << " A >> ";
	      cin >> A;
	      QRDecomposition<double> qr(A);
	      cout << "--- A ---\n"	 << A
		   << "--- Qt * Q ---\n" << evaluate(qr.Qt() *
						     transpose(qr.Qt()))
		   << "--- Rt ---\n"	 << qr.Rt()
		   << "--- Rt * Qt ---\n" << evaluate(qr.Rt() * qr.Qt())
		   << endl;
	  }
	    break;

	  case 'T':
	  {
	    cerr << " A >> ";
	    cin >> A;
	    TriDiagonal<double> tri(A);
	    cout << "--- A ---\n"	   << A
		 << "--- Ut * U ---\n"	   << evaluate(tri.Ut() *
						       transpose(tri.Ut()))
		 << "--- Ut * A * U ---\n" << evaluate(tri.Ut() * A *
						       transpose(tri.Ut()))
		 << "  Diagonal     = "	   << tri.diagonal()
		 << "  Off-diagonal = "	   << tri.off_diagonal()
		 << endl;
	  }
	    break;

	  case 'E':
	    cerr << " A >> ";
	    cin >> A;
	    B = eigen(A, u, true);
	    cout << "--- A ---\n"	   << A
		 << "--- Ut * U ---\n"	   << evaluate(B * transpose(B))
		 << "--- Ut * A * U ---\n" << evaluate(B * A * transpose(B))
		 << "  Eigen-values = "	   << u
		 << endl;
	    break;

	  case 'B':
	  {
	    cerr << " A >> ";
	    cin >> A;
	    BiDiagonal<double> bi(A);
	    cout << "--- A ---\n"	   << A
		 << "--- Ut * U ---\n"	   << evaluate(bi.Ut() *
						       transpose(bi.Ut()))
		 << "--- Vt * V ---\n"	   << evaluate(bi.Vt() *
						       transpose(bi.Vt()))
		 << "--- Vt * A * U ---\n" << evaluate(bi.Vt() * A *
						       transpose(bi.Ut()))
		 << "  Diagonal     = "	   << bi.diagonal()
		 << "  Off-diagonal = "	   << bi.off_diagonal()
		 << endl;
	  }
	    break;

	  case 'S':
	  {
	    cerr << " A >> ";
	    cin >> A;
	    SVDecomposition<double> svd(A);
	    cout << "--- A ---\n"	   << A
		 << "--- Ut * U ---\n"	   << evaluate(svd.Ut() *
						       transpose(svd.Ut()))
		 << "--- Vt * V ---\n"	   << evaluate(svd.Vt() *
						       transpose(svd.Vt()))
		 << "--- Vt * A * U ---\n" << evaluate(svd.Vt() * A *
						       transpose(svd.Ut()))
		 << "  Singular-values = " << svd.diagonal()
		 << endl;
	  }
	    break;

	  case 'G':
	    cerr << " A >> ";
	    cin >> A;
	    B = pseudo_inverse(A);
	    cout << "--- A ---\n"		      << A
		 << "--- pseudo_inverse(A) --\n"      << B
		 << "--- A * pseudo_inverse(A) ---\n" << evaluate(A * B)
		 << "--- pseudo_inverse(A) * A ---\n" << evaluate(B * A)
		 << endl;
	    break;

	  /*	  case 'C':
	    cerr << " A >> ";
	    cin >> A;
	    F = A;
	    cout << "--- A ---\n" << A
		 << "--- F ---\n" << F << endl;
		 break;*/
	}
	}
	catch (std::exception& err)
	{
	    cerr << err.what() << endl;
	}
    }

	return 0;
}
