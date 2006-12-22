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
 *  $Id: Vector++.h,v 1.14 2006-12-22 00:05:55 ueshiba Exp $
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
//! 2次元超平面内での回転を表すクラス
/*!
  具体的には
  \f[
    \TUvec{R}{}(p, q; \theta) \equiv
    \begin{array}{r@{}l}
      & \begin{array}{ccccccccccc}
        & & \makebox[4.0em]{} & p & & & \makebox[3.8em]{} & q & & &
      \end{array} \\
      \begin{array}{l}
        \\ \\ \\ \raisebox{1.5ex}{$p$} \\ \\ \\ \\ \raisebox{1.5ex}{$q$} \\ \\ \\
      \end{array} &
      \TUbeginarray{ccccccccccc}
	1 \\
	& \ddots \\
	& & 1 \\
	& & & \cos\theta & & & & -\sin\theta \\
	& & & & 1 \\
	& & & & & \ddots \\
	& & & & & & 1 \\
	& & & \sin\theta & & & & \cos\theta \\
	& & & & & & & & 1\\
	& & & & & & & & & \ddots \\
	& & & & & & & & & & 1
      \TUendarray
    \end{array}
  \f]
  なる回転行列で表される．
*/
class Rotation
{
  public:
    Rotation(int p, int q, double x, double y)		;
    Rotation(int p, int q, double theta)		;

  //! p軸を返す
  /*!
    \return	p軸のindex.
  */
    int		p()				const	{return _p;}

  //! q軸を返す
  /*!
    \return	q軸のindex.
  */
    int		q()				const	{return _q;}

  //! 回転角生成ベクトルの長さを返す
  /*!
    \return	回転角生成ベクトル(x, y)に対して
		\f$\sqrt{x^2 + y^2}\f$を返す.
  */
    double	length()			const	{return _l;}

  //! 回転角のcos値を返す
  /*!
    \return	回転角のcos値.
  */
    double	cos()				const	{return _c;}

  //! 回転角のsin値を返す
  /*!
    \return	回転角のsin値.
  */
    double	sin()				const	{return _s;}
    
  private:
    const int	_p, _q;				// rotation axis
    double	_l;				// length of (x, y)
    double	_c, _s;				// cos & sin
};

/************************************************************************
*  class Vector<T>							*
************************************************************************/
template <class T>	class Matrix;

//! T型の要素を持つベクトルを表すクラス
template <class T>
class Vector : public Array<T>
{
  public:
  //! 指定された次元のベクトルを生成し，全要素を0で初期化する
  /*!
    \param d	ベクトルの次元.
  */
    explicit Vector(u_int d=0)			:Array<T>(d)	{*this = 0.0;}

  //! 外部記憶領域と次元を指定してベクトルを生成する
  /*!
    \param p	外部記憶領域へのポインタ.
    \param d	ベクトルの次元.
  */
    Vector(T* p, u_int d)			:Array<T>(p, d)		{}

  //! 与えられたベクトルと記憶領域を共有する部分ベクトルを生成する
  /*!
    \param v	元のベクトル.
    \param i	部分ベクトルの第0要素を指定するindex.
    \param d	部分ベクトルの次元.
  */
    Vector(const Vector& v, u_int i, u_int d)	:Array<T>(v, i, d)	{}

  //! コピーコンストラクタ
  /*!
    \param v	コピー元ベクトル.
  */
    Vector(const Vector& v)			:Array<T>(v)		{}
    
  //! コピー演算子
  /*!
    \param v	コピー元ベクトル.
    \return	このベクトル.
  */
    Vector&	operator =(const Vector& v)	{Array<T>::operator =(v);
						 return *this;}

    const Vector	operator ()(u_int, u_int)	const	;
    Vector		operator ()(u_int, u_int)		;

    using	Array<T>::dim;

  //! このベクトルの全ての要素に同一の数値を代入する
  /*!
    \param c	代入する数値.
    \return	このベクトル.
  */
    Vector&	operator  =(double c)		{Array<T>::operator  =(c);
						 return *this;}

  //! このベクトルに指定された数値を掛ける
  /*!
    \param c	掛ける数値.
    \return	このベクトル，すなわち\f$\TUvec{u}{}\leftarrow c\TUvec{u}{}\f$.
  */
    Vector&	operator *=(double c)		{Array<T>::operator *=(c);
						 return *this;}

  //! このベクトルを指定された数値で割る
  /*!
    \param c	割る数値.
    \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \frac{\TUvec{u}{}}{c}\f$.
  */
    Vector&	operator /=(double c)		{Array<T>::operator /=(c);
						 return *this;}

  //! このベクトルに他のベクトルを足す
  /*!
    \param v	足すベクトル.
    \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \TUvec{u}{} + \TUvec{v}{}\f$.
  */
    Vector&	operator +=(const Vector& v)	{Array<T>::operator +=(v);
						 return *this;}

  //! このベクトルから他のベクトルを引く
  /*!
    \param v	引くベクトル.
    \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \TUvec{u}{} - \TUvec{v}{}\f$.
  */
    Vector&	operator -=(const Vector& v)	{Array<T>::operator -=(v);
						 return *this;}

    Vector&	operator ^=(const Vector&)	;

  //! このベクトルの右から行列を掛ける
  /*!
    \param m	掛ける行列.
    \return	このベクトル，すなわち
		\f$\TUtvec{u}{} \leftarrow \TUtvec{u}{}\TUvec{M}{}\f$.
  */
    Vector&	operator *=(const Matrix<T>& m) {return *this = *this * m;}

  //! このベクトルの符号を反転したベクトルを返す
  /*!
    \return	符号を反転したベクトル，すなわち\f$-\TUvec{u}{}\f$.
  */
    Vector	operator  -()		const	{Vector r(*this);
						 r *= -1; return r;}

  //! このベクトルの長さの2乗を返す
  /*!
    \return	ベクトルの長さの2乗，すなわち\f$\TUnorm{\TUvec{u}{}}^2\f$.
  */
    double	square()		const	{return *this * *this;}

  //! このベクトルの長さを返す
  /*!
    \return	ベクトルの長さ，すなわち\f$\TUnorm{\TUvec{u}{}}\f$.
  */
    double	length()		const	{return sqrt(square());}

  //! このベクトルと他のベクトルの差の長さの2乗を返す
  /*!
    \param v	比較対象となるベクトル.
    \return	ベクトル間の差の2乗，すなわち
		\f$\TUnorm{\TUvec{u}{} - \TUvec{v}{}}^2\f$.
  */
    double	sqdist(const Vector& v) const	{return (*this - v).square();}

  //! このベクトルと他のベクトルの差の長さを返す
  /*!
    \param v	比較対象となるベクトル.
    \return	ベクトル間の差，すなわち
		\f$\TUnorm{\TUvec{u}{} - \TUvec{v}{}}\f$.
  */
    double	dist(const Vector& v)	const	{return sqrt(sqdist(v));}

  //! このベクトルの長さを1に正規化する
  /*!
    \return	このベクトル，すなわち
		\f$
		  \TUvec{u}{}\leftarrow\frac{\TUvec{u}{}}{\TUnorm{\TUvec{u}{}}}
		\f$.
  */
    Vector&	normalize()			{return *this /= length();}

    Vector	normal()		const	;
    Vector&	solve(const Matrix<T>&)		;
    Matrix<T>	skew()			const	;

  //! ベクトルの次元を変更し，0に初期化する
  /*!
    \param d	新しい次元.
  */
    void	resize(u_int d)		{Array<T>::resize(d); *this = 0.0;}

  //! ベクトルの内部記憶領域と次元を変更する
  /*!
    \param p	新しい内部記憶領域へのポインタ.
    \param d	新しい次元.
  */
    void	resize(T* p, u_int d)	{Array<T>::resize(p, d);}
};

//! 入力ストリームからベクトルを読み込む(ASCII)
/*!
  \param in	入力ストリーム.
  \param a	ベクトルの読み込み先.
  \return	inで指定した入力ストリーム.
*/
template <class T> inline std::istream&
operator >>(std::istream& in, Vector<T>& v)
{
    return in >> (Array<T>&)v;
}

//! 出力ストリームへベクトルを書き出す(ASCII)
/*!
  \param out	出力ストリーム.
  \param a	書き出すベクトル.
  \return	outで指定した出力ストリーム.
*/
template <class T> inline std::ostream&
operator <<(std::ostream& out, const Vector<T>& v)
{
    return out << (const Array<T>&)v;
}

/************************************************************************
*  class Matrix<T>							*
************************************************************************/
//! T型の要素を持つ行列を表すクラス
/*!
  各行がT型の要素を持つベクトル#TU::Vector<T>になっている．
*/
template <class T>
class Matrix : public Array2<Vector<T> >
{
  public:
  //! 指定されたサイズの行列を生成し，全要素を0で初期化する
  /*!
    \param r	行列の行数.
    \param c	行列の列数.
  */
    explicit Matrix(u_int r=0, u_int c=0)
	:Array2<Vector<T> >(r, c)				{*this = 0.0;}

  //! 外部記憶領域とサイズを指定して行列を生成する
  /*!
    \param p	外部記憶領域へのポインタ.
    \param r	行列の行数.
    \param c	行列の列数.
  */
    Matrix(T* p, u_int r, u_int c) :Array2<Vector<T> >(p, r, c)	{}

  //! 与えられた行列と記憶領域を共有する部分行列を生成する
  /*!
    \param m	元の行列.
    \param i	部分行列の第0行を指定するindex.
    \param j	部分行列の第0列を指定するindex.
    \param r	部分行列の行数.
    \param c	部分行列の列数.
  */
    Matrix(const Matrix& m, u_int i, u_int j, u_int r, u_int c)
	:Array2<Vector<T> >(m, i, j, r, c)			{}

  //! コピーコンストラクタ
  /*!
    \param m	コピー元行列.
  */
    Matrix(const Matrix& m)	:Array2<Vector<T> >(m)		{}

  //! コピー演算子
  /*!
    \param m	コピー元行列.
    \return	この行列.
  */
    Matrix&	operator =(const Matrix& m)
			{Array2<Vector<T> >::operator =(m); return *this;}

    const Matrix	operator ()(u_int, u_int, u_int, u_int)	const	;
    Matrix		operator ()(u_int, u_int, u_int, u_int)		;

    using	Array2<Vector<T> >::dim;
    using	Array2<Vector<T> >::nrow;
    using	Array2<Vector<T> >::ncol;
    
    Matrix&	diag(double)			;
    Matrix&	rot(double, int)		;

  //! この行列の全ての要素に同一の数値を代入する
  /*!
    \param c	代入する数値.
    \return	この行列.
  */
    Matrix&	operator  =(double c)		{Array2<Vector<T> >::
						 operator  =(c); return *this;}
  //! この行列に指定された数値を掛ける
  /*!
    \param c	掛ける数値.
    \return	この行列，すなわち\f$\TUvec{A}{}\leftarrow c\TUvec{A}{}\f$.
  */
    Matrix&	operator *=(double c)		{Array2<Vector<T> >::
						 operator *=(c); return *this;}

  //! この行列を指定された数値で割る
  /*!
    \param c	割る数値.
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \frac{\TUvec{A}{}}{c}\f$.
  */
    Matrix&	operator /=(double c)		{Array2<Vector<T> >::
						 operator /=(c); return *this;}

  //! この行列に他の行列を足す
  /*!
    \param m	足す行列.
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \TUvec{A}{} + \TUvec{M}{}\f$.
  */
    Matrix&	operator +=(const Matrix& m)	{Array2<Vector<T> >::
						 operator +=(m); return *this;}

  //! この行列から他の行列を引く
  /*!
    \param m	引く行列.
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \TUvec{A}{} - \TUvec{M}{}\f$.
  */
    Matrix&	operator -=(const Matrix& m)	{Array2<Vector<T> >::
						 operator -=(m); return *this;}

  //! この行列に他の行列を掛ける
  /*!
    \param m	掛ける行列.
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \TUvec{A}{}\TUvec{M}{}\f$.
  */
    Matrix&	operator *=(const Matrix& m)	{return *this = *this * m;}

    Matrix&	operator ^=(const Vector<T>&)	;

  //! この行列の符号を反転した行列を返す
  /*!
    \return	符号を反転した行列，すなわち\f$-\TUvec{A}{}\f$.
  */
    Matrix	operator  -()			const	{Matrix r(*this);
							 r *= -1; return r;}

    Matrix	trns()				const	;
    Matrix	inv()				const	;
    Matrix&	solve(const Matrix<T>&)			;
    T		det()				const	;
    T		det(u_int, u_int)		const	;
    T		trace()				const	;
    Matrix	adj()				const	;
    Matrix	pinv(double cndnum=1.0e5)	const	;
    Matrix	eigen(Vector<T>&)		const	;
    Matrix	geigen(const Matrix<T>&,
		       Vector<T>&)		const	;
    Matrix	cholesky()			const	;
    Matrix&	normalize()				;
    Matrix&	rotate_from_left(const Rotation&)	;
    Matrix&	rotate_from_right(const Rotation&)	;
    double	square()			const	;

  //! この行列の2乗ノルムを返す
  /*!
    \return	行列の2乗ノルム，すなわち\f$\TUnorm{\TUvec{A}{}}\f$.
  */
    double	length()		const	{return sqrt(square());}

    Matrix&	symmetrize()				;
    Matrix&	antisymmetrize()			;
    void	rot2angle(double& theta_x,
			  double& theta_y,
			  double& theta_z)	const	;
    Vector<T>	rot2axis(double& c, double& s)	const	;
    Vector<T>	rot2axis()			const	;

  //! 単位正方行列を生成する
  /*!
    \param d	単位正方行列の次元.
    \return	単位正方行列.
  */
    static Matrix	I(u_int d)	{return Matrix<T>(d, d).diag(1.0);}

    static Matrix	Rt(const Vector<T>& n, double c, double s)	;
    static Matrix	Rt(const Vector<T>& axis)			;

  //! 行列のサイズを変更し，0に初期化する
  /*!
    \param r	新しい行数.
    \param c	新しい列数.
  */
    void	resize(u_int r, u_int c)
			{Array2<Vector<T> >::resize(r, c); *this = 0.0;}

  //! 行列の内部記憶領域とサイズを変更する
  /*!
    \param p	新しい内部記憶領域へのポインタ.
    \param r	新しい行数.
    \param c	新しい列数.
  */
    void	resize(T* p, u_int r, u_int c)
			{Array2<Vector<T> >::resize(p, r, c);}
};

//! 入力ストリームから行列を読み込む(ASCII)
/*!
  \param in	入力ストリーム.
  \param a	行列の読み込み先.
  \return	inで指定した入力ストリーム.
*/
template <class T> inline std::istream&
operator >>(std::istream& in, Matrix<T>& m)
{
    return in >> (Array2<Vector<T> >&)m;
}

//! 出力ストリームへ行列を書き出す(ASCII)
/*!
  \param out	出力ストリーム.
  \param a	書き出す行列.
  \return	outで指定した出力ストリーム.
*/
template <class T> inline std::ostream&
operator <<(std::ostream& out, const Matrix<T>& m)
{
    return out << (const Array2<Vector<T> >&)m;
}

/************************************************************************
*  numerical operators							*
************************************************************************/
//! 2つのベクトルの足し算
/*!
  \param a	第1引数.
  \param b	第2引数.
  \return	結果を格納したベクトル，すなわち\f$\TUvec{a}{}+\TUvec{b}{}\f$.
*/
template <class T> inline Vector<T>
operator +(const Vector<T>& a, const Vector<T>& b)
    {Vector<T> r(a); r += b; return r;}

//! 2つのベクトルの引き算
/*!
  \param a	第1引数.
  \param b	第2引数.
  \return	結果を格納したベクトル，すなわち\f$\TUvec{a}{}-\TUvec{b}{}\f$.
*/
template <class T> inline Vector<T>
operator -(const Vector<T>& a, const Vector<T>& b)
    {Vector<T> r(a); r -= b; return r;}

//! ベクトルに定数を掛ける
/*!
  \param c	掛ける定数.
  \param a	ベクトル.
  \return	結果を格納したベクトル，すなわち\f$c\TUvec{a}{}\f$.
*/
template <class T> inline Vector<T>
operator *(double c, const Vector<T>& a)
    {Vector<T> r(a); r *= c; return r;}

//! ベクトルに定数を掛ける
/*!
  \param a	ベクトル.
  \param c	掛ける定数.
  \return	結果を格納したベクトル，すなわち\f$c\TUvec{a}{}\f$.
*/
template <class T> inline Vector<T>
operator *(const Vector<T>& a, double c)
    {Vector<T> r(a); r *= c; return r;}

//! ベクトルの各要素を定数で割る
/*!
  \param a	ベクトル.
  \param c	割る定数.
  \return	結果を格納したベクトル，すなわち\f$\frac{1}{c}\TUvec{a}{}\f$.
*/
template <class T> inline Vector<T>
operator /(const Vector<T>& a, double c)
    {Vector<T> r(a); r /= c; return r;}

//! 2つの行列の足し算
/*!
  \param a	第1引数.
  \param b	第2引数.
  \return	結果を格納した行列，すなわち\f$\TUvec{A}{}+\TUvec{B}{}\f$.
*/
template <class T> inline Matrix<T>
operator +(const Matrix<T>& a, const Matrix<T>& b)
    {Matrix<T> r(a); r += b; return r;}

//! 2つの行列の引き算
/*!
  \param a	第1引数.
  \param b	第2引数.
  \return	結果を格納した行列，すなわち\f$\TUvec{A}{}-\TUvec{B}{}\f$.
*/
template <class T> inline Matrix<T>
operator -(const Matrix<T>& a, const Matrix<T>& b)
    {Matrix<T> r(a); r -= b; return r;}

//! 行列に定数を掛ける
/*!
  \param c	掛ける定数.
  \param a	行列.
  \return	結果を格納した行列，すなわち\f$c\TUvec{A}{}\f$.
*/
template <class T> inline Matrix<T>
operator *(double c, const Matrix<T>& a)
    {Matrix<T> r(a); r *= c; return r;}

//! 行列に定数を掛ける
/*!
  \param a	行列.
  \param c	掛ける定数.
  \return	結果を格納した行列，すなわち\f$c\TUvec{A}{}\f$.
*/
template <class T> inline Matrix<T>
operator *(const Matrix<T>& a, double c)
    {Matrix<T> r(a); r *= c; return r;}

//! 行列の各要素を定数で割る
/*!
  \param a	行列.
  \param c	割る定数.
  \return	結果を格納した行列，すなわち\f$\frac{1}{c}\TUvec{A}{}\f$.
*/
template <class T> inline Matrix<T>
operator /(const Matrix<T>& a, double c)
    {Matrix<T> r(a); r /= c; return r;}

template <class T> extern double
operator *(const Vector<T>&, const Vector<T>&)	;

//! 2つの3次元ベクトルのベクトル積
/*!
  \param v	第1引数.
  \param w	第2引数.
  \return	ベクトル積，すなわち\f$\TUvec{v}{}\times\TUvec{w}{}\f$.
*/
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

//! ?x3行列の各行と3次元ベクトルのベクトル積
/*!
  \param m	?x3行列.
  \param v	3次元ベクトル.
  \return	結果の行列，すなわち\f$(\TUtvec{M}{}\times\TUvec{v}{})^\top\f$.
*/
template <class T> inline Matrix<T>
operator ^(const Matrix<T>& m, const Vector<T>& v)
    {Matrix<T> r(m); r ^= v; return r;}

/************************************************************************
*  class LUDecomposition<T>						*
************************************************************************/
//! 正方行列のLU分解を表すクラス
template <class T>
class LUDecomposition : private Array2<Vector<T> >
{
  public:
    LUDecomposition(const Matrix<T>&)			;

    void	substitute(Vector<T>&)		const	;

  //! もとの正方行列の行列式を返す
  /*!
    \return	もとの正方行列の行列式.
  */
    T		det()				const	{return _det;}
    
  private:
    using	Array2<Vector<T> >::nrow;
    using	Array2<Vector<T> >::ncol;
    
    Array<int>	_index;
    T		_det;
};

/************************************************************************
*  class Householder<T>							*
************************************************************************/
template <class T>	class QRDecomposition;
template <class T>	class TriDiagonal;
template <class T>	class BiDiagonal;

//! Householder変換を表すクラス
template <class T>
class Householder : public Matrix<T>
{
  private:
    Householder(u_int dd, u_int d)
	:Matrix<T>(dd, dd), _d(d), _sigma(Matrix<T>::nrow())	{}
    Householder(const Matrix<T>&, u_int)			;

    using		Matrix<T>::dim;
    using		Matrix<T>::nrow;
    using		Matrix<T>::ncol;
    
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
//! 一般行列のQR分解を表すクラス
/*!
  与えられた行列\f$\TUvec{A}{} \in \TUspace{R}{m\times n}\f$に対して
  \f$\TUvec{A}{} = \TUtvec{R}{}\TUtvec{Q}{}\f$なる下半三角行列
  \f$\TUtvec{R}{} \in \TUspace{R}{m\times n}\f$と回転行列
  \f$\TUtvec{Q}{} \in \TUspace{R}{n\times n}\f$を求める
  （\f$\TUvec{A}{}\f$の各行を\f$\TUtvec{Q}{}\f$の行の線型結合で表現す
  る）．
 */
template <class T>
class QRDecomposition : private Matrix<T>
{
  public:
    QRDecomposition(const Matrix<T>&)			;

    using		Matrix<T>::dim;

  //! QR分解の下半三角行列を返す
  /*!
    \return	下半三角行列\f$\TUtvec{R}{}\f$.
  */
    const Matrix<T>&	Rt()			const	{return *this;}

  //! QR分解の回転行列を返す
  /*!
    \return	回転行列\f$\TUtvec{Q}{}\f$.
  */
    const Matrix<T>&	Qt()			const	{return _Qt;}
    
  private:
    using		Matrix<T>::nrow;
    using		Matrix<T>::ncol;
    
    Householder<T>	_Qt;			// rotation matrix
};

/************************************************************************
*  class TriDiagonal<T>							*
************************************************************************/
//! 対称行列の3重対角化を表すクラス
/*!
  与えられた対称行列\f$\TUvec{A}{} \in \TUspace{R}{d\times d}\f$に対し
  て\f$\TUtvec{U}{}\TUvec{A}{}\TUvec{U}{}\f$が3重対角行列となるような回
  転行列\f$\TUtvec{U}{} \in \TUspace{R}{d\times d}\f$を求める．
 */
template <class T>
class TriDiagonal
{
  public:
    TriDiagonal(const Matrix<T>&)			;

  //! 3重対角化される対称行列の次元(= 行数 = 列数)を返す
  /*!
    \return	対称行列の次元.
  */
    u_int		dim()			const	{return _Ut.nrow();}

  //! 3重対角化を行う回転行列を返す
  /*!
    \return	回転行列.
  */
    const Matrix<T>&	Ut()			const	{return _Ut;}

  //! 3重対角行列の対角成分を返す
  /*!
    \return	対角成分.
  */
    const Vector<T>&	diagonal()		const	{return _diagonal;}

  //! 3重対角行列の非対角成分を返す
  /*!
    \return	非対角成分.
  */
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
//! 一般行列の2重対角化を表すクラス
/*!
  与えられた一般行列\f$\TUvec{A}{} \in \TUspace{R}{m\times n}\f$に対し
  て\f$\TUtvec{V}{}\TUvec{A}{}\TUvec{U}{}\f$が2重対角行列となるような2
  つの回転行列\f$\TUtvec{U}{} \in \TUspace{R}{n\times n}\f$,
  \f$\TUtvec{V}{} \in \TUspace{R}{m\times m}\f$を求める．\f$m \le n\f$
  の場合は下半三角な2重対角行列に，\f$m > n\f$の場合は上半三角な2重対角
  行列になる．
 */
template <class T>
class BiDiagonal
{
  public:
    BiDiagonal(const Matrix<T>&)		;

  //! 2重対角化される行列の行数を返す
  /*!
    \return	行列の行数.
  */
    u_int		nrow()		const	{return _Vt.nrow();}

  //! 2重対角化される行列の列数を返す
  /*!
    \return	行列の列数.
  */
    u_int		ncol()		const	{return _Ut.nrow();}

  //! 2重対角化を行うために右から掛ける回転行列の転置を返す
  /*!
    \return	右から掛ける回転行列の転置.
  */
    const Matrix<T>&	Ut()		const	{return _Ut;}

  //! 2重対角化を行うために左から掛ける回転行列を返す
  /*!
    \return	左から掛ける回転行列.
  */
    const Matrix<T>&	Vt()		const	{return _Vt;}

  //! 2重対角行列の対角成分を返す
  /*!
    \return	対角成分.
  */
    const Vector<T>&	diagonal()	const	{return _Dt.sigma();}

  //! 2重対角行列の非対角成分を返す
  /*!
    \return	非対角成分.
  */
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
//! 一般行列の特異値分解を表すクラス
/*!
  与えられた一般行列\f$\TUvec{A}{} \in \TUspace{R}{m\times n}\f$に対し
  て\f$\TUtvec{V}{}\TUvec{A}{}\TUvec{U}{}\f$が対角行列となるような2つの
  回転行列\f$\TUtvec{U}{} \in \TUspace{R}{n\times n}\f$,
  \f$\TUtvec{V}{} \in \TUspace{R}{m\times m}\f$を求める．
 */
template <class T>
class SVDecomposition : private BiDiagonal<T>
{
  public:
  //! 与えられた一般行列の特異値分解を求める
  /*!
    \param a	特異値分解する一般行列.
  */
    SVDecomposition(const Matrix<T>& a)
	:BiDiagonal<T>(a)			{BiDiagonal<T>::diagonalize();}

    using	BiDiagonal<T>::nrow;
    using	BiDiagonal<T>::ncol;
    using	BiDiagonal<T>::Ut;
    using	BiDiagonal<T>::Vt;
    using	BiDiagonal<T>::diagonal;

  //! 特異値を求める
  /*!
    \param i	絶対値の大きい順に並んだ特異値の1つを指定するindex.
    \return	指定されたindexに対応する特異値.
  */
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
    return 2.0*fabs(a - b) <= _tol*(fabs(a) + fabs(b) + EPS);
}
 
}

#endif	/* !__TUVectorPP_h	*/
