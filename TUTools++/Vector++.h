/*
 *  平成9年 電子技術総合研究所 植芝俊夫 著作権所有
 *
 *  著作者による許可なしにこのプログラムの第三者への開示、複製、改変、
 *  使用等その他の著作人格権を侵害する行為を禁止します。
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *
 *  Copyright 1996
 *  Toshio UESHIBA, Electrotechnical Laboratory
 *
 *  All rights reserved.
 *  Any changing, copying or giving information about source programs of
 *  any part of this software and/or documentation without permission of the
 *  authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damage in use of this program.
 */

/*
 *  $Id: Vector++.h,v 1.7 2003-03-17 00:22:30 ueshiba Exp $
 */
#ifndef __TUVectorPP_h
#define __TUVectorPP_h

#include <math.h>
#ifdef WIN32
#  define M_PI	3.14159265358979323846
#endif
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class Rotation							*
************************************************************************/
class Rotation
{
  public:
    Rotation(int p, int q, double x, double y)		;
    Rotation(int p, int q, double theta)		;
    
    int		p()				const	{return _p;}
    int		q()				const	{return _q;}
    double	cos()				const	{return _c;}
    double	sin()				const	{return _s;}
    
  private:
    const int	_p, _q;				// rotation axis
    double	_c, _s;				// cos & sin
};

/************************************************************************
*  class Vector<T>							*
************************************************************************/
template <class T>	class Matrix;
template <class T>
class Vector : public Array<T>
{
  public:
    explicit Vector(u_int d=0)			:Array<T>(d)	{*this = 0.0;}
    Vector(T* p, u_int d)			:Array<T>(p, d)		{}
    Vector(const Vector& v, u_int i, u_int d)	:Array<T>(v, i, d)	{}
    Vector(const Vector& v)			:Array<T>(v)		{}
    Vector&	operator =(const Vector& v)	{Array<T>::operator =(v);
						 return *this;}

    const Vector	operator ()(u_int, u_int)	const	;
    Vector		operator ()(u_int, u_int)		;

    Vector&	operator  =(double c)		{Array<T>::operator  =(c);
						 return *this;}
    Vector&	operator *=(double c)		{Array<T>::operator *=(c);
						 return *this;}
    Vector&	operator /=(double c)		{Array<T>::operator /=(c);
						 return *this;}
    Vector&	operator +=(const Vector& v)	{Array<T>::operator +=(v);
						 return *this;}
    Vector&	operator -=(const Vector& v)	{Array<T>::operator -=(v);
						 return *this;}
    Vector&	operator ^=(const Vector&)	;
    Vector&	operator *=(const Matrix<T>& m) {return *this = *this * m;}
    Vector	operator  -()		const	{Vector r(*this);
						 r *= -1; return r;}
    double	square()		const	{return *this * *this;}
    double	length()		const	{return sqrt(square());}
    double	sqdist(const Vector& v) const	{return (*this - v).square();}
    double	dist(const Vector& v)	const	{return sqrt(sqdist(v));}
    Vector&	normalize()			{return *this /= length();}
    Vector	normal()		const	;
    Vector&	solve(const Matrix<T>&)		;
    Matrix<T>	skew()			const	;

    void	resize(u_int d)		{Array<T>::resize(d); *this = 0.0;}
    void	resize(T* p, u_int d)	{Array<T>::resize(p, d);}
};

template <class T> inline std::istream&
operator >>(std::istream& in, Vector<T>& v)
{
    return in >> (Array<T>&)v;
}

template <class T> inline std::ostream&
operator <<(std::ostream& out, const Vector<T>& v)
{
    return out << (const Array<T>&)v;
}

/************************************************************************
*  class Matrix<T>							*
************************************************************************/
const double	CNDNUM = 1.0e4;		// default condition number for pinv

template <class T>
class Matrix : public Array2<Vector<T> >
{
  public:
    explicit Matrix(u_int r=0, u_int c=0)
	:Array2<Vector<T> >(r, c)				{*this = 0.0;}
    Matrix(T* p, u_int r, u_int c) :Array2<Vector<T> >(p, r, c)	{}
    Matrix(const Matrix& m, u_int i, u_int j, u_int r, u_int c)
	:Array2<Vector<T> >(m, i, j, r, c)			{}
    Matrix(const Matrix& m)	:Array2<Vector<T> >(m)		{}
    Matrix&	operator =(const Matrix& m)
			{Array2<Vector<T> >::operator =(m); return *this;}

    const Matrix	operator ()(u_int, u_int, u_int, u_int)	const	;
    Matrix		operator ()(u_int, u_int, u_int, u_int)		;

    Matrix&	diag(double)			;
    Matrix&	rot(double, int)		;
    Matrix&	operator  =(double c)		{Array2<Vector<T> >::
						 operator  =(c); return *this;}
    Matrix&	operator *=(double c)		{Array2<Vector<T> >::
						 operator *=(c); return *this;}
    Matrix&	operator /=(double c)		{Array2<Vector<T> >::
						 operator /=(c); return *this;}
    Matrix&	operator +=(const Matrix& m)	{Array2<Vector<T> >::
						 operator +=(m); return *this;}
    Matrix&	operator -=(const Matrix& m)	{Array2<Vector<T> >::
						 operator -=(m); return *this;}
    Matrix&	operator *=(const Matrix& m)	{return *this = *this * m;}
    Matrix&	operator ^=(const Vector<T>&)	;
    Matrix	operator  -()			const	{Matrix r(*this);
							 r *= -1; return r;}
    Matrix	trns()				const	;
    Matrix	inv()				const	;
    Matrix&	solve(const Matrix<T>&)			;
    T		det()				const	;
    T		det(u_int, u_int)		const	;
    T		trace()				const	;
    Matrix	adj()				const	;
    Matrix	pinv(double)			const	;
    Matrix	eigen(Vector<T>&)		const	;
    Matrix	geigen(const Matrix<T>&,
		       Vector<T>&)		const	;
    Matrix	cholesky()			const	;
    Matrix&	normalize()				;
    Matrix&	rotate_from_left(const Rotation&)	;
    Matrix&	rotate_from_right(const Rotation&)	;
    double	square()			const	;
    double	length()		const	{return sqrt(square());}
    Matrix&	symmetrize()				;
    Matrix&	antisymmetrize()			;
    void	rot2angle(double& theta_x,
			  double& theta_y,
			  double& theta_z)	const	;
    Vector<T>	rot2axis(double& c, double& s)	const	;
    Vector<T>	rot2axis()			const	;
    
    static Matrix	I(u_int d)	{return Matrix<T>(d, d).diag(1.0);}
    static Matrix	Rt(const Vector<T>& n, double c, double s)	;
    static Matrix	Rt(const Vector<T>& axis)			;

    void	resize(u_int r, u_int c)
			{Array2<Vector<T> >::resize(r, c); *this = 0.0;}
    void	resize(T* p, u_int r, u_int c)
			{Array2<Vector<T> >::resize(p, r, c);}
};

template <class T> inline std::istream&
operator >>(std::istream& in, Matrix<T>& m)
{
    return in >> (Array2<Vector<T> >&)m;
}

template <class T> inline std::ostream&
operator <<(std::ostream& out, const Matrix<T>& m)
{
    return out << (const Array2<Vector<T> >&)m;
}

/************************************************************************
*  numerical operators							*
************************************************************************/
template <class T> inline Vector<T>
operator +(const Vector<T>& a, const Vector<T>& b)
    {Vector<T> r(a); r += b; return r;}

template <class T> inline Vector<T>
operator -(const Vector<T>& a, const Vector<T>& b)
    {Vector<T> r(a); r -= b; return r;}

template <class T> inline Vector<T>
operator *(double c, const Vector<T>& a)
    {Vector<T> r(a); r *= c; return r;}

template <class T> inline Vector<T>
operator *(const Vector<T>& a, double c)
    {Vector<T> r(a); r *= c; return r;}

template <class T> inline Vector<T>
operator /(const Vector<T>& a, double c)
    {Vector<T> r(a); r /= c; return r;}

template <class T> inline Matrix<T>
operator +(const Matrix<T>& a, const Matrix<T>& b)
    {Matrix<T> r(a); r += b; return r;}

template <class T> inline Matrix<T>
operator -(const Matrix<T>& a, const Matrix<T>& b)
    {Matrix<T> r(a); r -= b; return r;}

template <class T> inline Matrix<T>
operator *(double c, const Matrix<T>& a)
    {Matrix<T> r(a); r *= c; return r;}

template <class T> inline Matrix<T>
operator *(const Matrix<T>& a, double c)
    {Matrix<T> r(a); r *= c; return r;}

template <class T> inline Matrix<T>
operator /(const Matrix<T>& a, double c)
    {Matrix<T> r(a); r /= c; return r;}

template <class T> extern double
operator *(const Vector<T>&, const Vector<T>&)	;

template <class T> inline Vector<T>
operator ^(const Vector<T>& v, const Vector<T>& w)
    {Vector<T> r(v); r ^= w; return r;}

template <class T> extern Vector<T>
operator *(const Vector<T>&, const Matrix<T>&)	;

template <class T> extern Matrix<T>
operator %(const Vector<T>&, const Vector<T>&)	;

template <class T> extern Matrix<T>
operator ^(const Vector<T>&, const Matrix<T>&)	;

template <class T> extern Matrix<T>
operator *(const Matrix<T>&, const Matrix<T>&)	;

template <class T> extern Vector<T>
operator *(const Matrix<T>&, const Vector<T>&)	;

template <class T> inline Matrix<T>
operator ^(const Matrix<T>& m, const Vector<T>& v)
    {Matrix<T> r(m); r ^= v; return r;}

/************************************************************************
*  class LUDecomposition<T>						*
************************************************************************/
template <class T>
class LUDecomposition : private Array2<Vector<T> >
{
  public:
    LUDecomposition(const Matrix<T>&)			;

    void	substitute(Vector<T>&)		const	;
    T		det()				const	{return _det;}
    
  private:
    Array<int>	_index;
    T		_det;
};

/************************************************************************
*  class Householder<T>							*
************************************************************************/
template <class T>	class QRDecomposition;
template <class T>	class TriDiagonal;
template <class T>	class BiDiagonal;

template <class T>
class Householder : public Matrix<T>
{
  private:
    Householder(u_int dd, u_int d)
	:Matrix<T>(dd, dd), _d(d), _sigma(nrow())		{}
    Householder(const Matrix<T>&, u_int)			;

    void		apply_from_left(Matrix<T>&, int)	;
    void		apply_from_right(Matrix<T>&, int)	;
    void		apply_from_both(Matrix<T>&, int)	;
    void		make_transformation()			;
    const Vector<T>&	sigma()			const	{return _sigma;}
    Vector<T>&		sigma()				{return _sigma;}
    bool		sigma_is_zero(int, T)	const	;

  private:
    const u_int		_d;		// deviation from diagonal element
    Vector<T>		_sigma;

    friend class	QRDecomposition<T>;
    friend class	TriDiagonal<T>;
    friend class	BiDiagonal<T>;
};

/************************************************************************
*  class QRDecomposition<T>						*
************************************************************************/
template <class T>
class QRDecomposition : private Matrix<T>
{
  public:
    QRDecomposition(const Matrix<T>&)			;

    Matrix<T>::dim;
    const Matrix<T>&	Rt()			const	{return *this;}
    const Matrix<T>&	Qt()			const	{return _Qt;}
    
  private:
    Householder<T>	_Qt;			// rotation matrix
};

/************************************************************************
*  class TriDiagonal<T>							*
************************************************************************/
template <class T>
class TriDiagonal
{
  public:
    TriDiagonal(const Matrix<T>&)			;

    u_int		dim()			const	{return _Ut.nrow();}
    const Matrix<T>&	Ut()			const	{return _Ut;}
    const Vector<T>&	diagonal()		const	{return _diagonal;}
    const Vector<T>&	off_diagonal()		const	{return _Ut.sigma();}
    void		diagonalize()			;
    
  private:
    enum		{NITER_MAX = 30};

    bool		off_diagonal_is_zero(int)		const	;
    void		initialize_rotation(int, int,
					    double&, double&)	const	;
    
    Householder<T>	_Ut;
    Vector<T>		_diagonal;
    Vector<T>&		_off_diagonal;
};

/************************************************************************
*  class BiDiagonal<T>							*
************************************************************************/
template <class T>
class BiDiagonal
{
  public:
    BiDiagonal(const Matrix<T>&)		;

    u_int		nrow()		const	{return _Vt.nrow();}
    u_int		ncol()		const	{return _Ut.ncol();}
    const Matrix<T>&	Ut()		const	{return _Ut;}
    const Matrix<T>&	Vt()		const	{return _Vt;}
    const Vector<T>&	diagonal()	const	{return _Dt.sigma();}
    const Vector<T>&	off_diagonal()	const	{return _Et.sigma();}
    void		diagonalize()		;

  private:
    enum		{NITER_MAX = 30};
    
    bool		diagonal_is_zero(int)			const	;
    bool		off_diagonal_is_zero(int)		const	;
    void		initialize_rotation(int, int,
					    double&, double&)	const	;

    Householder<T>	_Dt;
    Householder<T>	_Et;
    Vector<T>&		_diagonal;
    Vector<T>&		_off_diagonal;
    T			_anorm;
    const Matrix<T>&	_Ut;
    const Matrix<T>&	_Vt;
};

/************************************************************************
*  class SVDecomposition<T>						*
************************************************************************/
template <class T>
class SVDecomposition : private BiDiagonal<T>
{
  public:
    SVDecomposition(const Matrix<T>& a)
	:BiDiagonal<T>(a)			{diagonalize();}

    BiDiagonal<T>::nrow;
    BiDiagonal<T>::ncol;
    BiDiagonal<T>::Ut;
    BiDiagonal<T>::Vt;
    BiDiagonal<T>::diagonal;
    
    const T&	operator [](int i)	const	{return diagonal()[i];}
};

/************************************************************************
*  class Minimization1<S>						*
************************************************************************/
static const double	DEFAULT_TOL = 3.0e-8;

template <class S>
class Minimization1
{
  private:
    enum		{DEFAULT_NITER_MAX = 100};

  public:
    Minimization1(S tol = DEFAULT_TOL, int niter_max = DEFAULT_NITER_MAX)
	:_tol(tol), _niter_max(niter_max)				{}
    
    virtual S		operator ()(const S&)			const	= 0;
    S			minimize(S&, S)				const	;

  private:
    const S		_tol;
    const int		_niter_max;
};

/************************************************************************
*  class Minimization<S, T>						*
************************************************************************/
template <class S, class T>
class Minimization
{
  private:
    class LineFunction : public Minimization1<S>
    {
      public:
	LineFunction(const Minimization<S, T>& func,
		     const T& x, const Vector<S>& h,
		     S tol, int niter_max)
	  :Minimization1<S>(tol, niter_max),
	   _func(func), _x(x), _h(h)		{}
    
	S	operator ()(const S& d)	const	{return _func(_func.proceed
							      (_x, d * _h));}
    
      private:
	const Minimization<S, T>&	_func;
	const T&			_x;
	const Vector<S>&		_h;
    };

  private:
    enum		{DEFAULT_NITER_MAX = 1000};
		 
  public:
    Minimization(S tol = DEFAULT_TOL, int niter_max = DEFAULT_NITER_MAX,
		 int pr = 0)
      :_tol(tol), _niter_max(niter_max), _print(pr)			{}
    
    virtual S		operator ()(const T&)			const	= 0;
    virtual Vector<S>	ngrad(const T& x)			const	= 0;
    virtual T		proceed(const T&, const Vector<S>&)	const	= 0;
    S			minimize(T&)					;
    S			steepest_descent(T&)				;
    S			line_minimize(T&, const Vector<S>&)	const	;

  protected:
    virtual void	update(const T&)				;
    virtual void	print(int, S, const T&)			const	;
    
  private:
    int			near_enough(S, S)			const	;
 
    const S		_tol;
    const int		_niter_max;
    const int		_print;
};

template <class S, class T> inline int
Minimization<S, T>::near_enough(S a, S b) const
{
#define EPS	1.0e-10
    return 2.0 * fabs(a - b) <= _tol * (fabs(a) + fabs(b) + EPS);
}
 
}

#endif	/* !__TUVectorPP_h	*/
