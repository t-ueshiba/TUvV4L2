/*
 *  $Id: main.cc,v 1.1 2011-09-07 05:01:16 ueshiba Exp $
 */
#include "TU/SparseMatrix++.h"

namespace TU
{
template <bool SYM, class T> SparseMatrix<T, SYM>
makeSparseMatrix(const Matrix<T>& D)
{
    SparseMatrix<T, SYM>	S;
    S.beginInit();
    for (u_int i = 0; i < D.nrow(); ++i)
    {
	S.setRow();
	
	for (u_int j = (SYM ? i : 0); j < D.ncol(); ++j)
	    if ((D[i][j] != T(0)) || (SYM && (i == j)))
		S.setCol(j, D[i][j]);
    }
    S.endInit();

    return S;
}

template <class T, bool SYM> Matrix<T>
makeDenseMatrix(const SparseMatrix<T, SYM>& S)
{
    Matrix<T>	D(S.nrow(), S.ncol());
    for (u_int i = 0; i < S.nrow(); ++i)
	for (u_int j = 0; j < S.ncol(); ++j)
	    D[i][j] = S(i, j);

    return D;
}
    
template <class T, bool SYM> void
composeTest()
{
    using namespace	std;
    
    Matrix<T>	A;
    cerr << "A>> " << flush;
    cin >> A;
    SparseMatrix<T, SYM>	S = makeSparseMatrix<SYM>(A);
    cerr << "--- S ---\n" << S;
    SparseMatrix<T, true>	SSt = S.compose();
    cerr << "--- S*St ---\n" << SSt;
    cerr << "--- error ---\n" << makeDenseMatrix(SSt) - A * A.trns();

    Matrix<T>	B;
    cerr << "B>> " << flush;
    cin >> B;
    B.symmetrize();
    SparseMatrix<T, true>	W = makeSparseMatrix<true>(B);
    cerr << "--- W ---\n" << W;
    SparseMatrix<T, true>	SWSt = S.compose(W);
    cerr << "--- S*W*St ---\n" << SWSt;
    cerr << "--- error ---\n" << makeDenseMatrix(SWSt) - A * B * A.trns();
}

template <class T, bool SYM> void
addTest()
{
    using namespace	std;
    
    Matrix<T>	A, B;
    cerr << "A>> " << flush;
    cin >> A;
    cerr << "B>> " << flush;
    cin >> B;
    SparseMatrix<T, SYM>	Sa = makeSparseMatrix<SYM>(A);
    cerr << "--- Sa ---\n" << Sa;
    SparseMatrix<T, SYM>	Sb = makeSparseMatrix<SYM>(B);
    cerr << "--- Sb ---\n" << Sb;

    SparseMatrix<T, SYM>	U = Sa + Sb;
    cerr << "--- Sa + Sb ---\n" << U;
    cerr << "--- error ---\n" << makeDenseMatrix(U) - (A + B);

    U = Sa - Sb;
    cerr << "--- S - T ---\n" << U;
    cerr << "--- error ---\n" << makeDenseMatrix(U) - (A - B);
}

}

int
main()
{
    using namespace	std;
    using namespace	TU;

    typedef double	value_type;
    
    for (;;)
    {
	cerr << "c: compose test(symmetric)\n"
	     << "C: compose test(non-symmetric)\n"
	     << "a: add test(symmetric)\n"
	     << "A: add test(non-symmetric)\n"
	     << "\n>> " << flush;

	char	c;
	if (!(cin >> c))
	    break;
	cin.get();	// discard '\n'

	try
	{
	    switch (c)
	    {
	      case 'c':
		composeTest<value_type, true>();
		break;
	      case 'C':
		composeTest<value_type, false>();
		break;
	      case 'a':
		addTest<value_type, true>();
		break;
	      case 'A':
		addTest<value_type, false>();
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
