/*
 *  $Id: get_principal.cc,v 1.2 2002-07-25 02:38:00 ueshiba Exp $
 */
#include "TU/Vector++.h"
#include <stdexcept>

namespace TU
{
Matrix<double>
get_principal(const Matrix<double>& F0,
	      const Matrix<double>& ldata, const Matrix<double>& rdata)
{
#ifdef DEBUG
    cerr << "=== BEGIN: get_principal ===" << endl;
#endif
  // Normalize the given fundamental matrix.
    double		scale = 0.0;
    Vector<double>	centroid(ldata.ncol());
    for (int i = 0; i < ldata.nrow(); ++i)
    {
	scale += ldata[i](0, 2).square();
	scale += rdata[i](0, 2).square();
	centroid += ldata[i];
	centroid += rdata[i];
    }
    scale = sqrt(scale / (2 * (ldata.nrow() + rdata.nrow())));
    centroid /= (ldata.nrow() + rdata.nrow());
    Matrix<double>	Tt(3, 3);
    Tt[0][0] = Tt[1][1] = scale;
    Tt[2] = centroid;
#ifdef DEBUG
    cerr << " --- Tt ---\n" << Tt;
#endif
    Matrix<double>	F = Tt * F0 * Tt.trns();
    
  // Compute G matrix and epipoles.
    Matrix<double>		G = F(0, 0, 3, 2) * F(0, 0, 2, 2).trns()
				  * F(0, 0, 2, 3);
    Matrix<double>		E(3, 3);
    SVDecomposition<double>	svdF(F);
    E[0] = svdF.Ut()[2];
    E[1] = svdF.Vt()[2];
    E[2] = E[0] ^ E[1];

  // Compute cross vectors.
    double	f[4], g[4], h[5];
    f[0] = E[2] * F * E[1];
    f[1] = E[0] * F * E[2];
    f[2] = E[0] * F * E[1];
    f[3] = E[2] * F * E[2];
    g[0] = E[2] * G * E[1];
    g[1] = E[0] * G * E[2];
    g[2] = E[0] * G * E[1];
    g[3] = E[2] * G * E[2];
    h[0] = f[0]*g[1] - f[1]*g[0];	// h01
    h[1] = f[1]*g[2] - f[2]*g[1];	// h12
    h[2] = f[2]*g[3] - f[3]*g[2];	// h23
    h[3] = f[3]*g[0] - f[0]*g[3];	// h30
    h[4] = f[2]*g[0] - f[0]*g[2];	// h20

  // Compute two solutions for the principal point.
    double	sqr = (h[0] + h[2]) * (h[0] + h[2]) - 4 * h[1] * h[3];
#ifdef DEBUG
    cerr << "  sqr = " << sqr << endl;
#endif
    if (sqr < 0.0)
      	throw std::domain_error("TU::get_principal: cannot compute!!");
    sqr = sqrt(sqr);

  // Compute two solutions.
    Vector<double>	principal_p(3), principal_n(3);
    if (h[0] + h[2] > 0)
    {
	principal_p[0] = (h[0] + h[2] + sqr) / (2 * h[1]);
	principal_n[0] = h[3] / (h[1] * principal_p[0]);
    }
    else
    {
	principal_n[0] = (h[0] + h[2] - sqr) / (2 * h[1]);
	principal_p[0] = h[3] / (h[1] * principal_n[0]);
    }
    principal_p[1] = (h[1] * principal_p[0] - h[2]) / h[4];
    principal_n[1] = (h[1] * principal_n[0] - h[2]) / h[4];
    principal_p[2] = principal_n[2] = 1.0;
#ifdef DEBUG
    cerr << " --- E ---\n" << E;
    cerr << "  f = " << f << "  g = " << g;
    cerr << "  f1*b + f2*a + f3*a*b + f4 = "
	 << f[0]*principal_p[1] + f[1]*principal_p[0] +
	    f[2]*principal_p[0]*principal_p[1] + f[3]
	 << endl;
    cerr << "  g1*b + g2*a + g3*a*b + g4 = "
	 << g[0]*principal_p[1] + g[1]*principal_p[0] +
	    g[2]*principal_p[0]*principal_p[1] + g[3]
	 << endl;
#endif
    principal_p *= E;
    principal_p *= Tt;
    principal_p /= principal_p[2];		// (u0, v0, 1)
    principal_n *= E;
    principal_n *= Tt;
    principal_n /= principal_n[2];		// (u0, v0, 1)

#ifdef DEBUG
    cerr << " --- two solutions for principal points ---\n"
	 << "  u0_p = " << principal_p << "  u0_n = " << principal_n;
#endif

  // Select solution closer to the centroid and construct the transformation.
    if (principal_p(0, 2).sqdist(centroid(0, 2)) < 
	principal_n(0, 2).sqdist(centroid(0, 2)))
	Tt[2] = principal_p;
    else
	Tt[2] = principal_n;
    Tt[0][0] = Tt[1][1] = 1.0;

#ifdef DEBUG
    cerr << "=== END:   get_principal ===" << endl;
#endif
    return Tt.trns();
}
 
}
