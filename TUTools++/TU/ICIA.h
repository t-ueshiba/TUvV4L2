/*
 *  $Id: ICIA.h,v 1.3 2012-08-16 06:52:44 ueshiba Exp $
 */
#ifndef __TU_ICIA_H
#define __TU_ICIA_H

#include "TU/Geometry++.h"
#include "TU/Image++.h"
#include "TU/DericheConvolver.h"
#include <iomanip>

namespace TU
{
/************************************************************************
*  class ICIA<MAP, T>							*
************************************************************************/
template <class MAP, class T>
class ICIA
{
  public:
    typedef typename MAP::element_type				element_type;
    
  public:
    ICIA(const Image<T>& imageSrc,
	 element_type intensityThresh=15.0, float alpha=1.0)		;

    element_type	operator ()(const Image<T>& imageDst,
				    MAP& f, bool newton=false,
				    size_t niter_max=100,
				    element_type tol=1.0e-4)		;
    element_type	operator ()(const Image<T>& imageDst,
				    int u0, int v0, size_t w, size_t h,
				    MAP& f, bool newton=false,
				    size_t niter_max=100,
				    element_type tol=1.0e-4)		;

  private:
    typedef typename MAP::param_type				 param_type;
    typedef Matrix<element_type,
		   Buf<element_type, MAP::DOF * MAP::DOF>,
		   Buf<Vector<element_type>, MAP::DOF> >	matrix_type;

    element_type	sqrerr(const Image<T>& imageDst,
			       int u0, int v0, size_t w, size_t h,
			       const MAP& f, param_type& g)	   const;
    matrix_type		moment(int u0, int v0, size_t w, size_t h) const;

    const Image<T>&	_imageSrc;
    Array2<Array<param_type> >	_grad;
    Array2<Array<matrix_type> >	_M;
    const element_type		_intensityThresh;
};

template <class MAP, class T>
ICIA<MAP, T>::ICIA(const Image<T>& imageSrc,
		   element_type intensityThresh, float alpha)
    :_imageSrc(imageSrc), _grad(_imageSrc.height(), _imageSrc.width()),
     _M(_grad.nrow(), _grad.ncol()), _intensityThresh(intensityThresh)
{
    typedef typename MAP::jacobian_type	jacobian_type;

  // 位置に関する原画像の輝度勾配を求める．
    Image<float>		edgeH(_imageSrc.width(), _imageSrc.height()),
				edgeV(_imageSrc.width(), _imageSrc.height());
    DericheConvolver2<float>	convolver(alpha);
    convolver.diffH(_imageSrc.begin(), _imageSrc.end(), edgeH.begin());
    convolver.diffV(_imageSrc.begin(), _imageSrc.end(), edgeV.begin());

  // 変換パラメータに関する原画像の輝度勾配とモーメント行列を求める．
    for (size_t v = 0; v < _grad.nrow(); ++v)
    {
	const float	*eH = edgeH[v].data(), *eV = edgeV[v].data();
	param_type	*grad = _grad[v].data();
	matrix_type	*M    = _M[v].data();
	matrix_type	val(MAP::DOF, MAP::DOF);
	
	if (v == 0)
	    for (size_t u = 0; u < _grad.ncol(); ++u)
	    {
		jacobian_type	J = MAP::jacobian0(u, v);
		((*grad = J[0]) *= *eH++) += (J[1] *= *eV++);
		*M = (val += *grad % *grad);
		++grad;
		++M;
	    }
	else
	{
	    const matrix_type	*Mp = _M[v-1].data();
	    for (size_t u = 0; u < _grad.ncol(); ++u)
	    {
		jacobian_type	J = MAP::jacobian0(u, v);
		((*grad = J[0]) *= *eH++) += (J[1] *= *eV++);
		(*M = (val += *grad % *grad)) += *Mp++;
		++grad;
		++M;
	    }
	}
    }
#ifdef ICIA_DEBUG
    using namespace	std;

    cout << 'M' << 2 << endl;
    _imageSrc.saveHeader(cout, ImageBase::RGB_24);
    _imageSrc.saveHeader(cout, ImageBase::U_CHAR);
#endif
}
    
template <class MAP, class T> inline typename ICIA<MAP, T>::element_type
ICIA<MAP, T>::operator ()(const Image<T>& imageDst, MAP& f,
			  bool newton, size_t niter_max, element_type tol)
{
    return operator ()(imageDst, 0, 0, _imageSrc.width(), _imageSrc.height(),
		       f, newton, niter_max, tol);
}
    
template <class MAP, class T> typename ICIA<MAP, T>::element_type
ICIA<MAP, T>::operator ()(const Image<T>& imageDst,
			  int u0, int v0, size_t w, size_t h, MAP& f,
			  bool newton, size_t niter_max, element_type tol)
{
    using namespace	std;

    if (newton)
    {
	const matrix_type	Minv = moment(u0, v0, w, h).inv();
	element_type		sqr = 0.0;
	for (size_t n = 0; n < niter_max; ++n)
	{
	    param_type		g;
	    element_type	sqr_new = sqrerr(imageDst, u0, v0, w, h, f, g);
#ifdef _DEBUG
	    cerr << "[" << setw(2) << n << "] sqr = " << sqr_new << endl;
#endif
	    f.compose(Minv*g);
	    if (fabs(sqr - sqr_new) <= tol*(sqr_new + sqr + 1.0e-7))
		return sqr_new;
	    sqr = sqr_new;
	}
    }
    else
    {
	matrix_type	M = moment(u0, v0, w, h);
	param_type	diagM(M.size());
	for (size_t i = 0; i < diagM.size(); ++i)
	    diagM[i] = M[i][i];
	
	param_type	g;
	element_type	sqr = sqrerr(imageDst, u0, v0, w, h, f, g);
#ifdef _DEBUG
	cerr << "     sqr = " << sqr << endl;
#endif
	element_type	lambda = 1.0e-4;
	for (size_t n = 0; n < niter_max; ++n)
	{
	  // L-M反復
	    for (;;)
	    {
		for (size_t i = 0; i < M.size(); ++i)
		    M[i][i] = (1.0 + lambda) * diagM[i];
		param_type	dtheta(g);
		dtheta.solve(M);
		MAP		f_new(f);
		f_new.compose(dtheta);
		param_type	g_new;
		element_type	sqr_new = sqrerr(imageDst, u0, v0, w, h,
						 f_new, g_new);
#ifdef _DEBUG	    
		cerr << "[" << setw(2) << n << "] sqr = " << sqr_new
		     << ",\tlambda = " << lambda << endl;
#endif
	      // 二乗誤差が減少するか調べる．
		if (sqr_new <= sqr)
		{
		    f = f_new;

		  // 収束判定
		    if (fabs(sqr - sqr_new) <= tol*(sqr_new + sqr + 1.0e-7))
			return sqr_new;
		
		    g   = g_new;
		    sqr = sqr_new;
		    lambda *= 0.1;		// L-M反復のパラメータを減らす．
		    break;
		}
		else
		    lambda *= 10.0;		// L-M反復のパラメータを増やす．
	    }
	}
    }

    throw runtime_error("ICIA::operator (): maximum iteration limit exceeded!");

    return -1.0;
}
    
template <class MAP, class T> typename ICIA<MAP, T>::element_type
ICIA<MAP, T>::sqrerr(const Image<T>& imageDst,
		     int u0, int v0, size_t w, size_t h,
		     const MAP& f, param_type& g) const
{
    using namespace			std;
    typedef typename MAP::point_type	point_type;
#ifdef ICIA_DEBUG
    Image<RGB>		rgbImage(_imageSrc.width(), _imageSrc.height());
    Image<T>	composedImage(_imageSrc.width(), _imageSrc.height());
#endif
    g.resize(MAP::DOF);

    element_type	sqr = 0.0;
    size_t		npoints = 0;
    for (size_t v = v0; v < v0 + h; ++v)
    {
	const T*		imageSrc = &_imageSrc[v][u0];
	const param_type*	grad	 = &_grad[v][u0];
		
	for (size_t u = u0; u < u0 + w; ++u)
	{
	    const point_type&	p = f(u, v);
	  //const point_type&	p = f(Point2i(u, v));

	    if (0 <= p[0] && p[0] < imageDst.width()  - 1 &&
		0 <= p[1] && p[1] < imageDst.height() - 1)
	    {
		element_type	val = imageDst.at(p);
		if (val > 0.5 && *imageSrc > 0.5)
		{
		    element_type	dI = val - *imageSrc;
		    if (dI > _intensityThresh)
			dI = _intensityThresh;
		    else if (dI < -_intensityThresh)
			dI = -_intensityThresh;
#ifdef ICIA_DEBUG
		    if (dI > 0.0)
			rgbImage[v][u] = RGB(0, 255*dI/_intensityThresh, 0);
		    else
			rgbImage[v][u] = RGB(-255*dI/_intensityThresh, 0, 0);
		    composedImage[v][u] = T((val + *imageSrc) / 2);
#endif
		    g   += dI * *grad;
		    sqr += dI * dI;
		    ++npoints;
		}
#ifdef ICIA_DEBUG
		else
		    rgbImage[v][u] = RGB(0, 0, 255);
#endif
	    }
	    ++imageSrc;
	    ++grad;
	}
    }
#ifdef ICIA_DEBUG
    rgbImage.saveData(cout, ImageBase::RGB_24);
    composedImage.saveData(cout, ImageBase::U_CHAR);
#endif
    if (npoints < MAP::DOF)
	throw runtime_error("ICIA::sqrerr(): not enough points!");
    
    return sqr / npoints;
}
    
template <class MAP, class T> typename ICIA<MAP, T>::matrix_type
ICIA<MAP, T>::moment(int u0, int v0, size_t w, size_t h) const
{
    --u0;
    --v0;

    const int	u1 = std::min(u0+int(w), int(_M.ncol())-1),
		v1 = std::min(v0+int(h), int(_M.nrow())-1);
    matrix_type	val;

    if (u0 < int(_M.ncol()) && v0 < int(_M.nrow()) && u1 >= 0 && v1 >= 0)
    {
	val = _M[v1][u1];
	if (u0 >= 0)
	{
	    val -= _M[v1][u0];
	    if (v0 >= 0)
		(val += _M[v0][u0]) -= _M[v0][u1];
	}
	else if (v0 >= 0)
	    val -= _M[v0][u1];
    }

    return val;
}
    
}
#endif	// !__TU_ICIA_H
