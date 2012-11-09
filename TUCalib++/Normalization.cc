/*
 *  $Id$
 */
#include "TU/Calib++.h"

namespace TU
{
/************************************************************************
*  class Normalization							*
************************************************************************/
//! 与えられた点群の同次座標から正規化変換を計算する
/*!
  振幅の平均値が1, 重心が原点になるような正規化変換が計算される．
  \param points	点群の同次座標．点の個数をN, 点が属する空間の次元をd
		とすると，N\f$\times\f$(d+1)行列として与えられる．
*/
void
Normalization::initialize(const Matrix<double>& points)
{
    _scale = 0.0;
    _centroid.resize(points.ncol()-1);
    for (int i = 0; i < points.nrow(); ++i)
    {
	_scale += points[i](0, dim()).square();
	_centroid += points[i](0, dim());
    }
    _scale = sqrt(_scale / (points.nrow() * dim()));
    _centroid /= points.nrow();
}

//! 正規化変換行列を返す
/*!
  \return	変換行列:
		\f$
		\TUvec{T}{} = 
		\TUbeginarray{ccc}
		 s^{-1} \TUvec{I}{d} & -s^{-1}\TUvec{c}{} \\ \TUtvec{0}{d} & 1
		\TUendarray
		\f$
*/
Matrix<double>
Normalization::T() const
{
    Matrix<double>	TT(dim()+1, dim()+1);
    for (int i = 0; i < dim(); ++i)
    {
	TT[i][i] = 1.0 / _scale;
	TT[i][dim()] = -_centroid[i] / _scale;
    }
    TT[dim()][dim()] = 1.0;

    return TT;
}

//! 正規化変換の転置行列を返す
/*!
  \return	変換の転置行列:
		\f$
		\TUtvec{T}{} = 
		\TUbeginarray{ccc}
		 s^{-1} \TUvec{I}{d} & \TUtvec{0}{d} \\ -s^{-1}\TUtvec{c}{} & 1
		\TUendarray
		\f$
*/
Matrix<double>
Normalization::Tt() const
{
    Matrix<double>	TTt(dim()+1, dim()+1);
    for (int i = 0; i < dim(); ++i)
    {
	TTt[i][i] = 1.0 / _scale;
	TTt[dim()][i] = -_centroid[i] / _scale;
    }
    TTt[dim()][dim()] = 1.0;

    return TTt;
}

//! 正規化変換の逆行列を返す
/*!
  \return	変換の逆行列:
		\f$
		\TUinv{T}{} = 
		\TUbeginarray{ccc}
		 s \TUvec{I}{d} & \TUvec{c}{} \\ \TUtvec{0}{d} & 1
		\TUendarray
		\f$
*/
Matrix<double>
Normalization::Tinv() const
{
    Matrix<double>	TTinv(dim()+1, dim()+1);
    for (int i = 0; i < dim(); ++i)
    {
	TTinv[i][i] = _scale;
	TTinv[i][dim()] = _centroid[i];
    }
    TTinv[dim()][dim()] = 1.0;

    return TTinv;
}

//! 正規化変換の逆行列の転置を返す
/*!
  \return	変換の逆行列の転置:
		\f$
		\TUtinv{T}{} = 
		\TUbeginarray{ccc}
		 s \TUvec{I}{d} & \TUvec{0}{d} \\ \TUtvec{c}{} & 1
		\TUendarray
		\f$
*/
Matrix<double>
Normalization::Ttinv() const
{
    Matrix<double>	TTtinv(dim()+1, dim()+1);
    for (int i = 0; i < dim(); ++i)
    {
	TTtinv[i][i] = _scale;
	TTtinv[dim()][i] = _centroid[i];
    }
    TTtinv[dim()][dim()] = 1.0;

    return TTtinv;
}
 
}
