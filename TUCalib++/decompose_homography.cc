/*
 *  $Id: decompose_homography.cc,v 1.2 2002-07-25 02:38:00 ueshiba Exp $
 */
#include "TU/Vector++.h"
#include <stdexcept>

namespace TU
{
void
decompose_homography(const Matrix<double>& H,
		     const Matrix<double>& ldata,
		     const Matrix<double>& rdata,
		     Matrix<double>& Rt, Vector<double>& t,
		     Vector<double>& normal)
{
    if (H.nrow() != 3 || H.ncol() != 3)
	throw std::invalid_argument("TU::decompose_homography: homography matrix H must be 3x3!!");
    
  // Compute eigen decomposition.
    Vector<double>	lambda(3);
    Matrix<double>	Ut = (H.trns() * H).eigen(lambda);
    Vector<double>	sigma(3);
    sigma[0] = sqrt(lambda[0]);
    sigma[1] = sqrt(lambda[1]);
    sigma[2] = sqrt(lambda[2]);

  // Compute two solutions for translation.
    Vector<double>	tt[2];
    tt[0].resize(3);
    tt[0][0]  = -sigma[2] * sqrt(lambda[0] - lambda[1]);
    tt[0][1]  =  0.0;
    tt[0][2]  =  sigma[0] * sqrt(lambda[1] - lambda[2]);
    tt[0]    /=  sigma[1] * sqrt(lambda[0] - lambda[2]);
    tt[1]     =  tt[0];
    tt[1][2] *= -1.0;
    tt[0]    *=  Ut;
    tt[1]    *=  Ut;

  // Compute two solutions for plane normal and distance.
    Vector<double>	n[2];
    n[0].resize(3);
    n[0][0]  =  sqrt(lambda[0]-lambda[1]);
    n[0][1]  =  0.0;
    n[0][2]  =  sqrt(lambda[1]-lambda[2]);
    n[0]    *=  sqrt((sigma[0]-sigma[2]) / (sigma[0]+sigma[2])) / sigma[1];
    n[1]     =  n[0];
    n[1][2] *= -1.0;
    n[0]    *=  Ut;
    n[1]    *=  Ut;

  // Compute two solutions for rotation.
    Matrix<double>	RRt[2];
    for (int i = 0; i < 2; ++i)
    {
	RRt[i] = H * (Matrix<double>::I(3) + tt[i]%n[i] / (1.0-n[i]*tt[i]))
	       / sigma[1];
#ifdef DEBUG
	cerr << "==========\n--- Rt * R ---\n" << RRt[i] * RRt[i].trns()
	     << "  det(Rt) = " << RRt[i].det()
	     << "\n--- H ---\n" << H / sigma[1]
	     << "--- Rt*(I - t%(n/d) ---\n" << RRt[i] - (RRt[i] * tt[i])%n[i];
	cerr << "--- Rt ---\n" << RRt[i];
#endif
    }

  // Check positivity of depths.
    int		bad[2];
    for (int i = 0; i < 2; ++i)
    {
      // Choose sign so that the depth of the first left point to be positive.
	if (n[i] * ldata[0] < 0.0)
	{
	    tt[i] *= -1.0;
	    n[i] *= -1.0;
	}
	
      // Check depths.
	bad[i] = 0;
	for (int j = 0; j < ldata.nrow(); ++j)
	    if (n[i] * ldata[j] < 0.0 || n[i] * (rdata[j] * RRt[i]) < 0.0)
		bad[i] = 1;
    }

  // Choose solution from two candidates.
    if (bad[0])
    {
	if (bad[1])
	{
	    throw std::domain_error("TU::decompose_interimage_homography(): No solutions!");
	}
	else		// Only 2nd solution is good.
	{
	    Rt = RRt[1];
	    t = tt[1];
	    normal = n[1];
	}
    }
    else if (bad[1])	// Only 1st solution is good.
    {
	Rt = RRt[0];
	t = tt[0];
	normal = n[0];
    }
    else		// Both solutions are good.
    {
	if (RRt[0][2][2] < RRt[1][2][2])
	{		// Choose solution with similar camera directions.
	    Rt = RRt[0];
	    t = tt[0];
	    normal = n[0];
	}
	else
	{
	    Rt = RRt[1];
	    t = tt[1];
	    normal = n[1];
	}
    }
}
 
}
