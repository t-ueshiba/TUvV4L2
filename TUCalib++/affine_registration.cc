/*
 *  $Id$
 */
#include "TU/Vector++.h"

namespace TU
{
double
affine_registration(const Matrix<double>& data0,
		    const Matrix<double>& data1,
		    Matrix<double>& At, Vector<double>& b)
{
    u_int	npoints = data0.nrow(), d = data0.ncol();

  // Compute centroids of the input data.
    Vector<double>	centroid0(d), centroid1(d);
    for (int i = 0; i < npoints; ++i)
    {
	centroid0 += data0[i];
	centroid1 += data1[i];
    }
    centroid0 /= npoints;
    centroid1 /= npoints;

  // Compute deviations from the centroid.
    Matrix<double>	delta0(data0), delta1(data1);
    for (int i = 0; i < npoints; ++i)
    {
	delta0[i] -= centroid0;
	delta1[i] -= centroid1;
    }
    
  // Compute moment matrix and vector.
    Matrix<double>	M(d*d, d*d), At_tmp(d, d);
    Vector<double>	a((double*)At_tmp, d*d);
    for (int i = 0; i < npoints; ++i)
	for (int j = 0; j < d; ++j)
	    for (int k = 0; k < d; ++k)
	    {
		for (int l = 0; l < d; ++l)
		    M[d*j+l][d*k+l] += delta1[i][j] * delta1[i][k];
		a[d*j+k] += delta1[i][j]*delta0[i][k];
	    }

  // Compute affine transformation.
    a.solve(M);
    At = At_tmp;
    b = centroid0 - centroid1 * At;
    
    return (delta0 -= delta1 * At).square() / npoints;
}
 
}
