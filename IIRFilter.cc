/*
 *  $Id: IIRFilter.cc,v 1.2 2006-10-23 06:39:47 ueshiba Exp $
 */
#include <math.h>
#include "TU/IIRFilter++.h"
#include "TU/Minimize++.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static void
coefficients4(float a0, float b0, float omega0, float alpha0,
	      float a1, float b1, float omega1, float alpha1,
	      float c[8])
{
    const float	c0 = cosf(omega0), s0 = sinf(omega0), e0 = expf(-alpha0),
		c1 = cosf(omega1), s1 = sinf(omega1), e1 = expf(-alpha1);
    c[0] =  e0*e1*(e1*(b0*s0 - a0*c0) + e0*(b1*s1 - a1*c1));	// i(n-3)
    c[1] =  a0*e1*e1 + a1*e0*e0
	 -  2.0*e0*e1*((b0*s0 - a0*c0)*c1 + (b1*s1 - a1*c1)*c0);// i(n-2)
    c[2] =  e0*(b0*s0 - (a0 + 2.0*a1)*c0)
	 +  e1*(b1*s1 - (a1 + 2.0*a0)*c1);			// i(n-1)
    c[3] =  a0 + a1;						// i(n)
    c[4] = -e0*e0*e1*e1;					// o(n-4)
    c[5] =  2.0*e0*e1*(e1*c0 + e0*c1);				// o(n-3)
    c[6] = -e0*e0 - e1*e1 - 4.0*e0*e1*c0*c1;			// o(n-2)
    c[7] =  2.0*(e0*c0 + e1*c1);				// o(n-1)
}

/************************************************************************
*  class DericheConvoler						*
************************************************************************/
//! このCanny-Deriche核の初期化を行う
/*!
  \param alpha	フィルタサイズを表す正数．小さいほど広がりが大きい．
  \return	このCanny-Deriche核自身.
*/
DericheConvolver&
DericheConvolver::initialize(float alpha)
{
    const float	e  = expf(-alpha), beta = sinhf(alpha);
    _c0[0] =  (alpha - 1.0) * e;		// i(n-1)
    _c0[1] =  1.0;				// i(n)
    _c0[2] = -e * e;				// oF(n-2)
    _c0[3] =  2.0 * e;				// oF(n-1)

    _c1[0] = -1.0;				// i(n-1)
    _c1[1] =  0.0;				// i(n)
    _c1[2] = -e * e;				// oF(n-2)
    _c1[3] =  2.0 * e;				// oF(n-1)

    _c2[0] =  (1.0 + beta) * e;			// i(n-1)
    _c2[1] = -1.0;				// i(n)
    _c2[2] = -e * e;				// oF(n-2)
    _c2[3] =  2.0 * e;				// oF(n-1)

    return *this;
}
    
/************************************************************************
*  class GaussianConvoler::Params					*
************************************************************************/
void
GaussianConvolver::Params::set(double aa, double bb, double tt, double aaa)
{
    a	  = aa;
    b	  = bb;
    theta = tt;
    alpha = aaa;
}

GaussianConvolver::Params&
GaussianConvolver::Params::operator -=(const Vector<double>& p)
{
    a	  -= p[0];
    b	  -= p[1];
    theta -= p[2];
    alpha -= p[3];

    return *this;
}

/************************************************************************
*  class GaussianConvolver::EvenConstraint				*
************************************************************************/
Vector<double>
GaussianConvolver::EvenConstraint::operator ()(const AT& params) const
{
    Vector<T>	val(1);
    const T	as0 = params[0].alpha/_sigma, ts0 = params[0].theta/_sigma,
		as1 = params[1].alpha/_sigma, ts1 = params[1].theta/_sigma;
    val[0] = (params[0].a*sinh(as0) + params[0].b* sin(ts0)) *
	     (cosh(as1) - cos(ts1))
	   + (params[1].a*sinh(as1) + params[1].b* sin(ts1)) *
	     (cosh(as0) - cos(ts0));

    return val;
}

Matrix<double>
GaussianConvolver::EvenConstraint::jacobian(const AT& params) const
{
    T	c0  = cos(params[0].theta/_sigma),  s0  = sin(params[0].theta/_sigma),
	ch0 = cosh(params[0].alpha/_sigma), sh0 = sinh(params[0].alpha/_sigma),
	c1  = cos(params[1].theta/_sigma),  s1  = sin(params[1].theta/_sigma),
	ch1 = cosh(params[1].alpha/_sigma), sh1 = sinh(params[1].alpha/_sigma);
	
    Matrix<T>	val(1, 8);
    val[0][0] = sh0*(ch1 - c1);
    val[0][1] = s0 *(ch1 - c1);
    val[0][2] = (params[0].b*c0 *(ch1 - c1) +
		 s0 *(params[1].a*sh1 + params[1].b*s1)) / _sigma;
    val[0][3] = (params[0].a*ch0*(ch1 - c1) +
		 sh0*(params[1].a*sh1 + params[1].b*s1)) / _sigma;
    val[0][4] = sh1*(ch0 - c0);
    val[0][5] = s1 *(ch0 - c0);
    val[0][6] = (params[1].b*c1 *(ch0 - c0) +
		 s1 *(params[0].a*sh0 + params[0].b*s0)) / _sigma;
    val[0][7] = (params[1].a*ch1*(ch0 - c0) +
		 sh1*(params[0].a*sh0 + params[0].b*s0)) / _sigma;

    return val;
}

/************************************************************************
*  class GaussianConvolver::CostFunction				*
************************************************************************/
Vector<double>
GaussianConvolver::CostFunction::operator ()(const AT& params) const
{
    Vector<T>	val(_ndivisions+1);
    for (int k = 0; k < val.dim(); ++k)
    {
	T	f = 0.0, x = k*_range/_ndivisions;
	for (int i = 0; i < params.dim(); ++i)
	{
	    const Params&	p = params[i];
	    f += (p.a*cos(x*p.theta) + p.b*sin(x*p.theta))*exp(-x*p.alpha);
	}
	val[k] = f - (x*x - 1.0)*exp(-x*x/2.0);
    }
    
    return val;
}
    
Matrix<double>
GaussianConvolver::CostFunction::jacobian(const AT& params) const
{
    Matrix<T>	val(_ndivisions+1, 4*params.dim());
    for (int k = 0; k < val.nrow(); ++k)
    {
	Vector<T>&	row = val[k];
	T		x = k*_range/_ndivisions;
	
	for (int i = 0; i < params.dim(); ++i)
	{
	    const Params&	p = params[i];
	    const T		c = cos(x*p.theta), s = sin(x*p.theta),
				e = exp(-x*p.alpha);
	    row[4*i]   = c * e;
	    row[4*i+1] = s * e;
	    row[4*i+2] = (-p.a * s + p.b * c) * e;
	    row[4*i+3] = -x * (p.a * c + p.b * s) * e;
	}
    }
    
    return val;
}

void
GaussianConvolver::CostFunction::update(AT& params, const Vector<T>& dp) const
{
    for (int i = 0; i < params.dim(); ++i)
	params[i] -= dp(4*i, 4);
}

/************************************************************************
*  class GaussianConvolver						*
************************************************************************/
//! このGauss核の初期化を行う
/*!
  \param sigma	フィルタサイズを表す正数．大きいほど広がりが大きい．
  \return	このGauss核自身.
*/
GaussianConvolver&
GaussianConvolver::initialize(float sigma)
{
    coefficients4( 1.80579,   7.42555, 0.676413/sigma, 2.10032/sigma,
		  -0.805838, -1.50785, 1.90174/sigma,  2.15811/sigma, _c0);
    
    coefficients4(-0.628422, -4.68837,  0.666686/sigma, 1.54201/sigma,
		   0.628422,  0.980129, 2.08425/sigma,  1.52152/sigma, _c1);

  /*    coefficients4(-1.27844,   3.24717, 0.752205/sigma, 1.18524/sigma,
	0.278487, -1.54294, 2.21984/sigma,  1.27214/sigma, _c2);*/
    CostFunction::AT	params(CostFunction::D);
    params[0].set(-1.3, 3.6, 0.75, 1.2);
    params[1].set(0.32, -1.7, 2.2, 1.3);
    CostFunction	err(100, 5.0);
    if (sigma < 0.55)
	throw std::runtime_error("GaussianConvolver::initialize(): sigma must be greater than 0.55");
    EvenConstraint	g(sigma);
    minimizeSquare(err, g, params, 1000, 1.0e-6);
    coefficients4(params[0].a, params[0].b,
		  params[0].theta/sigma, params[0].alpha/sigma,
		  params[1].a, params[1].b,
		  params[1].theta/sigma, params[1].alpha/sigma, _c2);

    return *this;
}

template class BilateralIIRFilter<2u>;

template class BilateralIIRFilter2<2u>;
template class BilateralIIRFilter2<2u>&
BilateralIIRFilter2<2u>::convolve(const Image<u_char>& in, Image<u_char>& out);
template BilateralIIRFilter2<2u>&
BilateralIIRFilter2<2u>::convolve(const Image<u_char>& in, Image<float>& out);
template BilateralIIRFilter2<2u>&
BilateralIIRFilter2<2u>::convolve(const Image<float>& in, Image<float>& out);

template class BilateralIIRFilter<4u>;

template class BilateralIIRFilter2<4u>;
template class BilateralIIRFilter2<4u>&
BilateralIIRFilter2<4u>::convolve(const Image<u_char>& in, Image<u_char>& out);
template BilateralIIRFilter2<4u>&
BilateralIIRFilter2<4u>::convolve(const Image<u_char>& in, Image<float>& out);
template BilateralIIRFilter2<4u>&
BilateralIIRFilter2<4u>::convolve(const Image<float>& in, Image<float>& out);

}

#if defined(__GNUG__) || defined(__INTEL_COMPILER)
#  include "TU/Array++.cc"
#  include "TU/IIRFilter++.cc"
#endif
