/*
 *  $Id: euclidean_registration.cc,v 1.1.1.1 2002-07-25 02:14:15 ueshiba Exp $
 */
#include "TU/Vector++.h"
#include <stdexcept>

namespace TU
{
double
euclidean_registration(const Matrix<double>& data0,
		       const Matrix<double>& data1,
		       Matrix<double>& Rt, Vector<double>& t)
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
    
  // Compute correlation matrix.
    Matrix<double>	K(d, d);
    for (int i = 0; i < npoints; ++i)
	K += delta0[i] % delta1[i];

  // Compute motion parameters.
    SVDecomposition<double>	svd(K);
    for (int i = 0; i < d; ++i)
	if (svd[i] < 0.0)
	    throw std::domain_error("TU::euclidean_registration: negative singular value!");
    Rt = svd.Ut().trns() * svd.Vt();
    t = centroid0 - centroid1 * Rt;
    
    return (delta0 -= delta1 * Rt).square() / npoints;
}
 
}
