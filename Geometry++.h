/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: Geometry++.h,v 1.26 2008-09-08 08:06:14 ueshiba Exp $
 */
#ifndef __TUGeometryPP_h
#define __TUGeometryPP_h

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class Normalize							*
************************************************************************/
//! 点の非同次座標の正規化変換を行うクラス
/*!
  \f$\TUud{x}{}=[\TUtvec{x}{}, 1]^\top~
  (\TUvec{x}{} \in \TUspace{R}{d})\f$に対して，以下のような平行移動と
  スケーリングを行う:
  \f[
	\TUud{y}{} =
	\TUbeginarray{c} s^{-1}(\TUvec{x}{} - \TUvec{c}{}) \\ 1	\TUendarray =
	\TUbeginarray{ccc}
	  s^{-1} \TUvec{I}{d} & -s^{-1}\TUvec{c}{} \\ \TUtvec{0}{d} & 1
	\TUendarray
	\TUbeginarray{c} \TUvec{x}{} \\ 1 \TUendarray =
	\TUvec{T}{}\TUud{x}{}
  \f]
  \f$s\f$と\f$\TUvec{c}{}\f$は，振幅の2乗平均値が空間の次元\f$d\f$に,
  重心が原点になるよう決定される．
*/
class Normalize
{
  public:
  //! 空間の次元を指定して正規化変換オブジェクトを生成する．
  /*!
    恒等変換として初期化される．
    \param d	空間の次元
  */
    Normalize(u_int d=2) :_npoints(0), _scale(1.0), _centroid(d)	{}

    template <class Iterator>
    Normalize(Iterator first, Iterator last)				;
    
    template <class Iterator>
    void		update(Iterator first, Iterator last)		;

    u_int		spaceDim()				const	;
    template <class T2, class B2>
    Vector<double>	operator ()(const Vector<T2, B2>& x)	const	;
    template <class T2, class B2>
    Vector<double>	normalizeP(const Vector<T2, B2>& x)	const	;
    
    Matrix<double>		T()				const	;
    Matrix<double>		Tt()				const	;
    Matrix<double>		Tinv()				const	;
    Matrix<double>		Ttinv()				const	;
    double			scale()				const	;
    const Vector<double>&	centroid()			const	;
    
  private:
    u_int		_npoints;	//!< これまでに与えた点の総数
    double		_scale;		//!< これまでに与えた点の振幅のRMS値
    Vector<double>	_centroid;	//!< これまでに与えた点群の重心
};

//! 与えられた点群の非同次座標から正規化変換オブジェクトを生成する．
/*!
  振幅の2乗平均値が#spaceDim(), 重心が原点になるような正規化変換が計算される．
  \param first	点群の先頭を示す反復子
  \param last	点群の末尾を示す反復子
*/
template <class Iterator> inline
Normalize::Normalize(Iterator first, Iterator last)
    :_npoints(0), _scale(1.0), _centroid()
{
    update(first, last);
}
    
//! 新たに点群を追加してその非同次座標から現在の正規化変換を更新する．
/*!
  振幅の2乗平均値が#spaceDim(), 重心が原点になるような正規化変換が計算される．
  \param first			点群の先頭を示す反復子
  \param last			点群の末尾を示す反復子
  \throw std::invalid_argument	これまでに与えられた点の総数が0の場合に送出
*/
template <class Iterator> void
Normalize::update(Iterator first, Iterator last)
{
    if (_npoints == 0)
    {
	if (first == last)
	    throw std::invalid_argument("Normalize::update(): 0-length input data!!");
	_centroid.resize(first->dim());
    }
    _scale = _npoints * (spaceDim() * _scale * _scale + _centroid * _centroid);
    _centroid *= _npoints;
    while (first != last)
    {
	_scale += first->square();
	_centroid += *first++;
	++_npoints;
    }
    if (_npoints == 0)
	throw std::invalid_argument("Normalize::update(): no input data accumulated!!");
    _centroid /= _npoints;
    _scale = sqrt((_scale / _npoints - _centroid * _centroid) / spaceDim());
}

//! この正規化変換が適用される空間の次元を返す．
/*! 
  \return	空間の次元(同次座標のベクトルとしての次元は#spaceDim()+1)
*/
inline u_int
Normalize::spaceDim() const
{
    return _centroid.dim();
}
    
//! 与えられた点に正規化変換を適用してその非同次座標を返す．
/*!
  \param x	点の非同次座標（#spaceDim()次元）
  \return	正規化された点の非同次座標（#spaceDim()次元）
*/
template <class T2, class B2> inline Vector<double>
Normalize::operator ()(const Vector<T2, B2>& x) const
{
    return (Vector<double>(x) -= _centroid) /= _scale;
}

//! 与えられた点に正規化変換を適用してその同次座標を返す．
/*!
  \param x	点の非同次座標（#spaceDim()次元）
  \return	正規化された点の同次座標（#spaceDim()+1次元）
*/
template <class T2, class B2> inline Vector<double>
Normalize::normalizeP(const Vector<T2, B2>& x) const
{
    return (*this)(x).homogenize();
}

//! 正規化変換のスケーリング定数を返す．
/*!
  \return	スケーリング定数（与えられた点列の振幅の2乗平均値）
*/
inline double
Normalize::scale() const
{
    return _scale;
}

//! 正規化変換の平行移動成分を返す．
/*!
  \return	平行移動成分（与えられた点列の重心）
*/
inline const Vector<double>&
Normalize::centroid() const
{
    return _centroid;
}

/************************************************************************
*  class Point2<T>							*
************************************************************************/
template <class T>
class Point2 : public Vector<T, FixedSizedBuf<T, 2> >
{
  private:
    typedef Vector<T, FixedSizedBuf<T, 2> >	array_type;
    
  public:
    Point2(T u=0, T v=0)						;
    template <class T2, class B2>
    Point2(const Vector<T2, B2>& v) :array_type(v)			{}
    template <class T2, class B2>
    Point2&	operator =(const Vector<T2, B2>& v)
		{
		    array_type::operator =(v);
		    return *this;
		}
    Point2	neighbor(int)					const	;
    Point2&	move(int)						;
    int		adj(const Point2&)				const	;
    int		dir(const Point2&)				const	;
    int		angle(const Point2&, const Point2&)		const	;
};

template <class T> inline
Point2<T>::Point2(T u, T v)
    :array_type()
{
    (*this)[0] = u;
    (*this)[1] = v;
}

template <class T> inline Point2<T>
Point2<T>::neighbor(int dir) const
{
    return Point2(*this).move(dir);
}

template <class T> Point2<T>&
Point2<T>::move(int dir)
{
    switch (dir % 8)
    {
      case 0:
	++(*this)[0];
	break;
      case 1:
      case -7:
	++(*this)[0];
	++(*this)[1];
	break;
      case 2:
      case -6:
	++(*this)[1];
	break;
      case 3:
      case -5:
	--(*this)[0];
	++(*this)[1];
	break;
      case 4:
      case -4:
	--(*this)[0];
	break;
      case 5:
      case -3:
	--(*this)[0];
	--(*this)[1];
	break;
      case 6:
      case -2:
	--(*this)[1];
	break;
      case 7:
      case -1:
	++(*this)[0];
	--(*this)[1];
	break;
    }
    return *this;
}

template <class T> int
Point2<T>::adj(const Point2<T>& p) const
{
    const int	du = int(p[0] - (*this)[0]), dv = int(p[1] - (*this)[1]);

    if (du == 0 && dv == 0)
        return -1;
    switch (du)
    {
      case -1:
      case 0:
      case 1:
        switch (dv)
        {
          case -1:
          case 0:
          case 1:
            return 1;
          default:
            return 0;
        }
        break;
    }
    return 0;
}

template <class T> int
Point2<T>::dir(const Point2<T>& p) const
{
    const int	du = int(p[0] - (*this)[0]), dv = int(p[1] - (*this)[1]);

    if (du == 0 && dv == 0)
        return 4;
    if (dv >= 0)
        if (du > dv)
            return 0;
        else if (du > 0)
            return 1;
        else if (du > -dv)
            return 2;
        else if (dv > 0)
            return 3;
        else
            return -4;
    else
        if (du >= -dv)
            return -1;
        else if (du >= 0)
            return -2;
        else if (du >= dv)
            return -3;
        else
            return -4;
}

template <class T> int
Point2<T>::angle(const Point2<T>& pp, const Point2<T>& pn) const
{
    int dp = pp.dir(*this), ang = dir(pn);
    
    if (dp == 4 || ang == 4)
        return 4;
    else if ((ang -= dp) > 3)
        return ang - 8;
    else if (ang < -4)
        return ang + 8;
    else
        return ang;
}

typedef Point2<short>	Point2s;
typedef Point2<int>	Point2i;
typedef Point2<float>	Point2f;
typedef Point2<double>	Point2d;

/************************************************************************
*  class Point3<T>							*
************************************************************************/
template <class T>
class Point3 : public Vector<T, FixedSizedBuf<T, 3> >
{
  private:
    typedef Vector<T, FixedSizedBuf<T, 3> >	array_type;
    
  public:
    Point3(T x=0, T y=0, T z=0)						;
    template <class T2, class B2>
    Point3(const Vector<T2, B2>& v) :array_type(v)			{}
    template <class T2, class B2>
    Point3&	operator =(const Vector<T2, B2>& v)
		{
		    array_type::operator =(v);
		    return *this;
		}
};

template <class T> inline
Point3<T>::Point3(T x, T y, T z)
    :array_type()
{
    (*this)[0] = x;
    (*this)[1] = y;
    (*this)[2] = z;
}

typedef Point3<short>	Point3s;
typedef Point3<int>	Point3i;
typedef Point3<float>	Point3f;
typedef Point3<double>	Point3d;

/************************************************************************
*  class HyperPlane							*
************************************************************************/
//! d次元射影空間中の超平面を表現するクラス
/*!
  d次元射影空間の点\f$\TUud{x}{} \in \TUspace{R}{d+1}\f$に対して
  \f$\TUtud{p}{}\TUud{x}{} = 0,~\TUud{p}{} \in \TUspace{R}{d+1}\f$
  によって表される．
*/
template <class T, class B=Buf<T> >
class HyperPlane : public Vector<T, B>
{
  public:
    HyperPlane(u_int d=2)						;

  //! 同次座標ベクトルを指定して超平面オブジェクトを生成する．
  /*!
    \param p	(d+1)次元ベクトル（dは超平面が存在する射影空間の次元）
  */
    template <class T2, class B2>
    HyperPlane(const Vector<T2, B2>& p)	:Vector<T, B>(p)		{}

    template <class Iterator>
    HyperPlane(Iterator first, Iterator last)				;

  //! 超平面オブジェクトの同次座標ベクトルを指定する．
  /*!
    \param v	(d+1)次元ベクトル（dは超平面が存在する射影空間の次元）
    \return	この超平面オブジェクト
  */
    template <class T2, class B2>
    HyperPlane&	operator =(const Vector<T2, B2>& v)
				{Vector<T, B>::operator =(v); return *this;}

    template <class Iterator>
    void	fit(Iterator first, Iterator last)			;

  //! この超平面が存在する射影空間の次元を返す．
  /*! 
    \return	射影空間の次元(同次座標のベクトルとしての次元は#spaceDim()+1)
  */
    u_int	spaceDim()		const	{return Vector<T, B>::dim()-1;}

  //! 超平面を求めるために必要な点の最小個数を返す．
  /*!
    現在設定されている射影空間の次元をもとに計算される．
    \return	必要な点の最小個数すなわち入力空間の次元#spaceDim()
  */
    u_int	ndataMin()		const	{return spaceDim();}

    template <class T2, class B2> inline T
    sqdist(const Vector<T2, B2>& x)		const	;
    template <class T2, class B2> inline double
    dist(const Vector<T2, B2>& x)		const	;
};

//! 空間の次元を指定して超平面オブジェクトを生成する．
/*!
  無限遠超平面([0, 0,..., 0, 1])に初期化される．
  \param d	この超平面が存在する射影空間の次元
*/
template <class T, class B> inline
HyperPlane<T, B>::HyperPlane(u_int d)
    :Vector<T, B>(d + 1)
{
    (*this)[d] = 1;
}
    
//! 与えられた点列の非同次座標に当てはめられた超平面オブジェクトを生成する．
/*!
  \param first			点列の先頭を示す反復子
  \param last			点列の末尾を示す反復子
  \throw std::invalid_argument	点の数が#ndataMin()に満たない場合に送出
*/
template <class T, class B> template <class Iterator> inline
HyperPlane<T, B>::HyperPlane(Iterator first, Iterator last)
{
    fit(first, last);
}

//! 与えられた点列の非同次座標に超平面を当てはめる．
/*!
  \param first			点列の先頭を示す反復子
  \param last			点列の末尾を示す反復子
  \throw std::invalid_argument	点の数が#ndataMin()に満たない場合に送出
*/
template <class T, class B> template <class Iterator> void
HyperPlane<T, B>::fit(Iterator first, Iterator last)
{
  // 点列の正規化
    const Normalize	normalize(first, last);

  // 充分な個数の点があるか？
    const u_int		ndata = std::distance(first, last),
			d     = normalize.spaceDim();
    if (ndata < d)	// Vector<T, B>のサイズが未定なのでndataMin()は無効
	throw std::invalid_argument("Hyperplane::initialize(): not enough input data!!");

  // データ行列の計算
    Matrix<T>	A(d, d);
    while (first != last)
    {
	const Vector<T>&	x = normalize(*first++);
	A += x % x;
    }

  // データ行列の最小固有値に対応する固有ベクトルから法線ベクトルを計算し，
  // さらに点列の重心より原点からの距離を計算する．
    Vector<T>		eval;
    const Matrix<T>&	Ut = A.eigen(eval);
    Vector<T, B>::resize(d+1);
    (*this)(0, d) = Ut[Ut.nrow()-1];
    (*this)[d] = -((*this)(0, d)*normalize.centroid());
    if ((*this)[d] > 0.0)
	Vector<T, B>::operator *=(-1.0);
}

//! 与えられた点と超平面の距離の2乗を返す．
/*!
  \param x	点の非同次座標（#spaceDim()次元）または同次座標
		（#spaceDim()+1次元）
  \return	点と超平面の距離の2乗
*/
template <class T, class B> template <class T2, class B2> inline T
HyperPlane<T, B>::sqdist(const Vector<T2, B2>& x) const
{
    const double	d = dist(x);
    return d*d;
}

//! 与えられた点と超平面の距離を返す．
/*!
  \param x			点の非同次座標（#spaceDim()次元）または
				同次座標（#spaceDim()+1次元）
  \return			点と超平面の距離（非負）
  \throw std::invalid_argument	点のベクトルとしての次元が#spaceDim()，
				#spaceDim()+1のいずれでもない場合，もしくは
				この点が無限遠点である場合に送出．
*/
template <class T, class B> template <class T2, class B2> double
HyperPlane<T, B>::dist(const Vector<T2, B2>& x) const
{
    const Vector<T2>&	p = (*this)(0, spaceDim());
    if (x.dim() == spaceDim())
	return (p * x + (*this)[spaceDim()]) / p.length();
    else if (x.dim() == spaceDim() + 1)
    {
	if (x[spaceDim()] == 0.0)
	    throw std::invalid_argument("HyperPlane::dist(): point at infinitiy!!");
	return (*this * x) / (p.length() * x[spaceDim()]);
    }
    else
	throw std::invalid_argument("HyperPlane::dist(): dimension mismatch!!");

    return 0;
}

typedef HyperPlane<float,  FixedSizedBuf<float,  3> >	LineP2f;
typedef HyperPlane<double, FixedSizedBuf<double, 3> >	LineP2d;
typedef HyperPlane<float,  FixedSizedBuf<float,  4> >	PlaneP3f;
typedef HyperPlane<double, FixedSizedBuf<double, 4> >	PlaneP3d;

}
#endif	/* !__TUGeometryPP_h */
