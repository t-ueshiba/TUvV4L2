/*
 *  $Id$
 */
#include "TU/Vector++.h"
#include <stdexcept>

namespace TU
{
void
get_flengths(const Matrix<double>& F0, double& kl, double& kr,
	     u_int commonFocalLengths)
{
  // Compute normalized epipoles.
    SVDecomposition<double>	svdF(F0);
    Vector<double>		el = svdF.Ut()[2], er = svdF.Vt()[2];
    el /= el(0, 2).length();
    er /= er(0, 2).length();

  // Rotate image coordinates.
    Matrix<double>	Tl(3, 3), Trt(3, 3);
    Tl[0][0]  = Tl[1][1] = el[0];
    Tl[0][1]  = -el[1];
    Tl[1][0]  =  el[1];
    Tl[2][2]  =  1.0;
    Trt[0][0] = Trt[1][1] = er[0];
    Trt[1][0] = -er[1];
    Trt[0][1] =  er[1];
    Trt[2][2] =  1.0;
    Matrix<double>	F = Trt * F0 * Tl;
  //#define DEBUG
#ifdef DEBUG
    cerr << "=== BEGIN: get_flengths ===" << endl;
    cerr << " --- rotated F ---\n" << F;
#endif

  // Compute focal lengths.
    if (commonFocalLengths)
    {
	kl = (F[1][2]*F[1][2] - F[2][1]*F[2][1])
	   / (F[0][1]*F[0][1] - F[1][0]*F[1][0]);
	if (kl <= 0.0)
	    throw std::domain_error("TU::get_flengths: cannot compute common focal lengths.");
	kl = kr = sqrt(kl);
    }
    else
    {
	kl = -(F[1][1]*F[2][1] + F[1][0]*F[2][0]) / (F[1][2]*F[2][2]);
	if (kl <= 0.0)
	    throw std::domain_error("TU::get_flengths: cannot compute focal length of the left camera.");
	kl = 1.0 / sqrt(kl);
	kr = -(F[1][1]*F[1][2] + F[0][1]*F[0][2]) / (F[2][1]*F[2][2]);
	if (kr <= 0.0)
	    throw std::domain_error("TU::get_flengths: cannot compute focal length of the right camera.");
	kr = 1.0 / sqrt(kr);
    }
#ifdef DEBUG
    cerr << " --- focal lengths ---\n"
	 << "  kl = " << kl << ", kr = " << kr << endl;
    cerr << "=== END:   get_flengths ===" << endl;
#endif
}
 
}
