/*
 *  $Id: Minimize++.h,v 1.5 2003-03-14 02:26:07 ueshiba Exp $
 */
#ifndef __TUMinimizePP_h
#define __TUMinimizePP_h

#include "TU/Vector++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class NullConstraint							*
************************************************************************/
template <class T, class AT>
class NullConstraint
{
  public:
    Vector<T> operator ()(const AT&)	const	{return Vector<T>(0);}
    Matrix<T> jacobian(const AT&)	const	{return Matrix<T>(0, 0);}
};

/************************************************************************
*  function minimizeSquare						*
*    -- Compute x st. ||f(x)||^2 -> min under g(x) = 0.			*
************************************************************************/
template <class F, class G, class AT> Matrix<typename F::T>
minimizeSquare(const F& f, const G& g, AT& x,
	       int niter_max=100, double tol=1.5e-8)
{
    using namespace		std;
    typedef typename F::T	T;		// element type.

    Vector<T>	fval   = f(x);			// function value.
    T		sqr    = fval * fval;		// square value.
    T		lambda = 1.0e-4;		// L-M parameter.

    for (int n = 0; n++ < niter_max; )
    {
	const Matrix<T>&	J    = f.jacobian(x);	// Jacobian.
	const Vector<T>&	Jtf  = fval * J;
	const Vector<T>&	gval = g(x);		// constraint residual.
	const u_int		xdim = J.ncol(), gdim = gval.dim();
	Matrix<T>		A(xdim + gdim, xdim + gdim);

	A(0, 0, xdim, xdim) = J.trns() * J;
	A(xdim, 0, gdim, xdim) = g.jacobian(x);
	A(0, xdim, xdim, gdim) = A(xdim, 0, gdim, xdim).trns();

	Vector<T>		diagA(xdim);
	for (int i = 0; i < xdim; ++i)
	    diagA[i] = A[i][i];			// Keep diagonal elements.

	for (;;)
	{
	  // Compute dx: update for parameters x to be estimated.
	    for (int i = 0; i < xdim; ++i)
		A[i][i] = (1.0 + lambda) * diagA[i];	// Augument diagonals.
	    Vector<T>	dx(xdim + gdim);
	    dx(0, xdim) = Jtf;
	    dx(xdim, gdim) = gval;
	    dx.solve(A);

	  // Compute updated parameters and function value to it.
	    AT			x_new(x);
	    f.update(x_new, dx(0, xdim));
	    const Vector<T>&	fval_new = f(x_new);
	    const T		sqr_new  = fval_new * fval_new;
#ifdef TUMinimizePP_DEBUG
	    cerr << "val^2 = " << sqr << ", gval = " << gval
		 << "  (update: val^2 = " << sqr_new
		 << ", lambda = " << lambda << ")" << endl;
#endif
	    if (fabs(sqr_new - sqr) <=
		tol * (fabs(sqr_new) + fabs(sqr) + 1.0e-10))
	    {
		for (int i = 0; i < xdim; ++i)
		    A[i][i] = diagA[i];
		return A(0, 0, xdim, xdim).pinv(1.0e8);
	    }

	    if (sqr_new < sqr)
	    {
		x	= x_new;		// Update parameters.
		fval	= fval_new;		// Update function value.
		sqr	= sqr_new;		// Update square value.
		lambda *= 0.1;			// Decrease L-M parameter.
		break;
	    }
	    else
		lambda *= 10.0;			// Increase L-M parameter.
	}
    }
    throw std::runtime_error("minimizeSquare: maximum iteration limit exceeded!");
    return Matrix<T>(0, 0);
}

/************************************************************************
*  function minimizeSquareSparse					*
*    -- Compute a and b st. sum||f(a, b[j])||^2 -> min under g(a) = 0.	*
************************************************************************/
template <class F, class G, class ATA, class ATB> Matrix<typename F::T>
minimizeSquareSparse(const F& f, const G& g, ATA& a, Array<ATB>& b,
		     int niter_max=100, double tol=1.5e-8)
{
    using namespace		std;
    typedef typename F::T	T;		// element type.
    typedef typename F::JT	JT;		// Jacobian type.

    Array<Vector<T> >		fval(b.dim());	// function values.
    T				sqr = 0;	// sum of squares.
    for (int j = 0; j < b.dim(); ++j)
    {
	fval[j] = f(a, b[j], j);
	sqr    += fval[j] * fval[j];
    }
    T	lambda = 1.0e-7;			// L-M parameter.

    for (int n = 0; n++ < niter_max; )
    {
	const u_int		adim = f.adim();
	JT			U(f.adims(), f.adims());
	Vector<T>		Jtf(adim);
	Array<Matrix<T> >	V(b.dim());
	Array<Matrix<T> >	W(b.dim());
	Array<Vector<T> >	Ktf(b.dim());
	for (int j = 0; j < b.dim(); ++j)
	{
	    const JT&		J  = f.jacobianA(a, b[j], j);
	    const JT&		Jt = J.trns();
	    const Matrix<T>&	K  = f.jacobianB(a, b[j], j);

	    U     += Jt * J;
	    Jtf   += fval[j] * J;
	    V[j]   = K.trns() * K;
	    W[j]   = Jt * K;
	    Ktf[j] = fval[j] * K;
	}

      	const Vector<T>&	gval = g(a);
	const u_int		gdim = gval.dim();
	Matrix<T>		A(adim + gdim, adim + gdim);
	
	A(adim, 0, gdim, adim) = g.jacobian(a);
	A(0, adim, adim, gdim) = A(adim, 0, gdim, adim).trns();

	for (;;)
	{
	  // Compute da: update for parameters a to be estimated.
	    A(0, 0, adim, adim) = U;
	    for (int i = 0; i < adim; ++i)
		A[i][i] *= (1.0 + lambda);		// Augument diagonals.

	    Vector<T>		da(adim + gdim);
	    da(0, adim) = Jtf;
	    da(adim, gdim) = gval;
	    Array<Matrix<T> >	VinvWt(b.dim());
	    Array<Vector<T> >	VinvKtf(b.dim());
	    for (int j = 0; j < b.dim(); ++j)
	    {
		Matrix<T>	Vinv = V[j];
		for (int k = 0; k < Vinv.dim(); ++k)
		    Vinv[k][k] *= (1.0 + lambda);	// Augument diagonals.
		Vinv = Vinv.inv();
		VinvWt[j]  = Vinv * W[j].trns();
		VinvKtf[j] = Vinv * Ktf[j];
		A(0, 0, adim, adim) -= W[j] * VinvWt[j];
		da(0, adim) -= W[j] * VinvKtf[j];
	    }
	    da.solve(A);

	  // Compute updated parameters and function value to it.
	    ATA			a_new(a);
	    f.updateA(a_new, da(0, adim));
	    Array<ATB>		b_new(b);
	    Array<Vector<T> >	fval_new(b.dim());
	    T			sqr_new = 0;
	    for (int j = 0; j < b.dim(); ++j)
	    {
		const Vector<T>& db = VinvKtf[j] - VinvWt[j] * da(0, adim);
		f.updateB(b_new[j], db);
		fval_new[j] = f(a_new, b_new[j], j);
		sqr_new	   += fval_new[j] * fval_new[j];
	    }
#ifdef TUMinimizePP_DEBUG
	    cerr << "val^2 = " << sqr << ", gval = " << gval
		 << "  (update: val^2 = " << sqr_new
		 << ", lambda = " << lambda << ")" << endl;
#endif
	    if (fabs(sqr_new - sqr) <=
		tol * (fabs(sqr_new) + fabs(sqr) + 1.0e-10))
	    {
		u_int		bdim = 0;
		for (int j = 0; j < b.dim(); ++j)
		    bdim += V[j].dim();
		Matrix<T>	S(adim + bdim, adim + bdim);
		Matrix<T>	Sa(S, 0, 0, adim, adim);
		Sa = U;
		for (int j = 0; j < b.dim(); ++j)
		{
		    VinvWt[j] = V[j].inv() * W[j].trns();
		    Sa -= W[j] * VinvWt[j];
		}
		for (int jj = adim, j = 0; j < b.dim(); ++j)
		{
		    const Matrix<T>&	VinvWtSa = VinvWt[j] * Sa;
		    for (int kk = adim, k = 0; k <= j; ++k)
		    {
			S(jj, kk, VinvWtSa.nrow(), VinvWt[k].nrow())
			     = VinvWtSa * VinvWt[k].trns();
			kk += VinvWt[k].nrow();
		    }
		    S(jj, jj, V[j].nrow(), V[j].nrow()) += V[j].inv();
		    jj += VinvWt[j].nrow();
		}
		Sa = Sa.pinv(1.0e8);
		for (int jj = adim, j = 0; j < b.dim(); ++j)
		{
		    S(jj, 0, VinvWt[j].nrow(), adim) = -VinvWt[j] * Sa;
		    jj += VinvWt[j].nrow();
		}
		    
		return S.symmetrize() *= sqr;
	    }
	    
	    if (sqr_new < sqr)
	    {
		a = a_new;			// Update parameters.
		b = b_new;
		fval = fval_new;		// Update function values.
		sqr = sqr_new;			// Update residual.
		lambda *= 0.1;			// Decrease L-M parameter.
		break;
	    }
	    else
		lambda *= 10.0;			// Increase L-M parameter.
	}
    }
    throw std::runtime_error("minimizeSquareSparse: maximum iteration limit exceeded!");

    return Matrix<T>(0, 0);
}

/************************************************************************
*  function minimizeSquareSparseDebug					*
*    -- Compute a and b st. sum||f(a, b[j])||^2 -> min under g(a) = 0.	*
************************************************************************/
template <class F, class G, class ATA, class ATB>  Matrix<typename F::T>
minimizeSquareSparseDebug(const F& f, const G& g, ATA& a, Array<ATB>& b,
			  int niter_max=100, double tol=1.5e-8)
{
    using namespace		std;
    typedef typename F::T	T;		// element type.

    Array<Vector<T> >		fval(b.dim());	// function values.
    T				sqr = 0;	// sum of squares.
    for (int j = 0; j < b.dim(); ++j)
    {
	fval[j] = f(a, b[j], j);
	sqr    += fval[j] * fval[j];
    }
    T	lambda = 1.0e-7;			// L-M parameter.

    for (int n = 0; n++ < niter_max; )
    {
	const u_int		adim = f.adim();
	const u_int		bdim = f.bdim() * b.dim();
      	const Vector<T>&	gval = g(a);
	const u_int		gdim = gval.dim();
	Matrix<T>		U(adim, adim);
	Vector<T>		Jtf(adim);
	Array<Matrix<T> >	V(b.dim());
	Array<Matrix<T> >	W(b.dim());
	Array<Vector<T> >	Ktf(b.dim());
	Matrix<T>		A(adim + bdim + gdim, adim + bdim + gdim);
	for (int j = 0; j < b.dim(); ++j)
	{
	    const Matrix<T>&	J  = f.jacobianA(a, b[j], j);
	    const Matrix<T>&	Jt = J.trns();
	    const Matrix<T>&	K  = f.jacobianB(a, b[j], j);

	    U     += Jt * J;
	    Jtf   += fval[j] * J;
	    V[j]   = K.trns() * K;
	    W[j]   = Jt * K;
	    Ktf[j] = fval[j] * K;

	    A(0, adim + j*f.bdim(), adim, f.bdim()) = W[j];
	    A(adim + j*f.bdim(), 0, f.bdim(), adim) = W[j].trns();
	}
	A(adim + bdim, 0, gdim, adim) = g.jacobian(a);
	A(0, adim + bdim, adim, gdim) = A(adim + bdim, 0, gdim, adim).trns();

	for (;;)
	{
	  // Compute da: update for parameters a to be estimated.
	    A(0, 0, adim, adim) = U;
	    for (int i = 0; i < adim; ++i)
		A[i][i] *= (1.0 + lambda);
	    for (int j = 0; j < b.dim(); ++j)
	    {
		A(adim + j*f.bdim(), adim + j*f.bdim(), f.bdim(), f.bdim())
		    = V[j];
		for (int k = 0; k < f.bdim(); ++k)
		    A[adim + j*f.bdim() + k][adim + j*f.bdim() + k]
			*= (1.0 + lambda);
	    }

	    Vector<T>	dx(adim + bdim + gdim);
	    dx(0, adim) = Jtf;
	    for (int j = 0; j < b.dim(); ++j)
		dx(adim + j*f.bdim(), f.bdim()) = Ktf[j];
	    dx(adim + bdim, gdim) = gval;
	    dx.solve(A);
	    
	  // Compute updated parameters and function value to it.
	    ATA			a_new(a);
	    f.updateA(a_new, dx(0, adim));
	    Array<ATB>		b_new(b);
	    Array<Vector<T> >	fval_new(b.dim());
	    T			sqr_new = 0;
	    for (int j = 0; j < b.dim(); ++j)
	    {
		const Vector<T>& db = dx(adim + j*f.bdim(), f.bdim());
	      /*		cerr << "*** check:  "
				<< (dx(0, adim) * W[j] + V[j] * db - Ktf[j]);*/
		f.updateB(b_new[j], db);
		fval_new[j] = f(a_new, b_new[j], j);
		sqr_new	   += fval_new[j] * fval_new[j];
	    }
#ifdef TUMinimizePP_DEBUG
	    cerr << "val^2 = " << sqr << ", gval = " << gval
		 << "  (update: val^2 = " << sqr_new
		 << ", lambda = " << lambda << ")" << endl;
#endif
	    if (fabs(sqr_new - sqr) <=
		tol * (fabs(sqr_new) + fabs(sqr) + 1.0e-10))
	    {
		A(0, 0, adim, adim) = U;
		for (int j = 0; j < b.dim(); ++j)
		    A(adim + j*f.bdim(), adim + j*f.bdim(), f.bdim(), f.bdim())
			= V[j];
		Vector<T>	evalue;
		A(0, 0, adim + bdim, adim + bdim).eigen(evalue);
		cerr << evalue;
		return A(0, 0, adim + bdim, adim + bdim).pinv(1.0e8) *= sqr;
	    }

	    if (sqr_new < sqr)
	    {
		a = a_new;			// Update parameters.
		b = b_new;
		fval = fval_new;		// Update function values.
		sqr = sqr_new;			// Update residual.
		lambda *= 0.1;			// Decrease L-M parameter.
		break;
	    }
	    else
		lambda *= 10.0;			// Increase L-M parameter.
	}
    }
    throw std::runtime_error("minimizeSquareSparseDebug: maximum iteration limit exceeded!");

    return Matrix<T>(0, 0);
}
 
}
#endif	/* !__TUMinimizePP_h	*/
