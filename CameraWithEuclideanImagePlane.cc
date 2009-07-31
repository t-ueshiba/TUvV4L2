/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: CameraWithEuclideanImagePlane.cc,v 1.14 2009-07-31 07:04:44 ueshiba Exp $
 */
#include "TU/Camera.h"

namespace TU
{
/************************************************************************
*  class CameraWithEuclideanImagePlane					*
************************************************************************/
CameraBase&
CameraWithEuclideanImagePlane::setProjection(const Matrix34d& PP)
{
    throw std::runtime_error("CameraWithEuclideanImagePlane::setProjection: Not implemented!!");
    return *this;
}

const CameraBase::Intrinsic&
CameraWithEuclideanImagePlane::intrinsic() const
{
    return _intrinsic;
}

CameraBase::Intrinsic&
CameraWithEuclideanImagePlane::intrinsic()
{
    return _intrinsic;
}

/************************************************************************
*  class CameraWithEuclideanImagePlane::Intrinsic			*
************************************************************************/
//! canonical画像座標系において表現された投影点の画像座標系における位置を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	xの画像座標系における2次元位置，すなわち
		\f$\TUvec{u}{} = k\TUvec{x}{} + \TUvec{u}{0}\f$
*/
Point2d
CameraWithEuclideanImagePlane::Intrinsic::operator ()(const Point2d& x) const
{
    return Point2d(k() * x[0] + _principal[0], k() * x[1] + _principal[1]);
}

//! 内部パラメータに関する投影点の画像座標の1階微分を求める
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	投影点のcanonical画像座標の1階微分を表す2x3ヤコビ行列，すなわち
		\f$
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}} =
		\TUbeginarray{ccc} x & 1 &  \\ y & & 1 \TUendarray
		\f$
*/
Matrix<double>
CameraWithEuclideanImagePlane::Intrinsic::jacobianK(const Point2d& x) const
{
    Matrix<double>	J(2, 3);
    J[0][0] = x[0];
    J[1][0] = x[1];
    J[0][1] = J[1][2] = 1.0;

    return J;
}

//! 画像座標における投影点の2次元位置をcanonical画像座標系に直す．
/*!
  \param u	画像座標系における投影点の2次元位置
  \return	canonical画像座標系におけるuの2次元位置，すなわち
		\f$\TUvec{x}{} = k^{-1}(\TUvec{u}{} - \TUvec{u}{0})\f$
*/
Point2d
CameraWithEuclideanImagePlane::Intrinsic::xcFromU(const Point2d& u) const
{
    return Point2d((u[0] - _principal[0]) / k(), (u[1] - _principal[1]) / k());
}

//! 内部パラメータ行列を返す．
/*!
  \return	3x3内部パラメータ行列，すなわち
		\f$
		\TUvec{K}{} =
		\TUbeginarray{ccc} k & & u_0 \\ & k & v_0 \\ & & 1 \TUendarray
		\f$
*/
Matrix33d
CameraWithEuclideanImagePlane::Intrinsic::K() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = k();
    mat[0][2] = _principal[0];
    mat[1][2] = _principal[1];
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータ行列の転置を返す．
/*!
  \return	3x3内部パラメータ行列の転置，すなわち
		\f$
		\TUtvec{K}{} =
		\TUbeginarray{ccc} k & & \\ & k & \\ u_0 & v_0 & 1 \TUendarray
		\f$
*/
Matrix33d
CameraWithEuclideanImagePlane::Intrinsic::Kt() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = k();
    mat[2][0] = _principal[0];
    mat[2][1] = _principal[1];
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータ行列の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の逆行列，すなわち
		\f$
		\TUinv{K}{} =
		\TUbeginarray{ccc}
		k^{-1} & & -k^{-1}u_0 \\ & k^{-1} & -k^{-1}v_0 \\ & & 1
		\TUendarray
		\f$
*/
Matrix33d
CameraWithEuclideanImagePlane::Intrinsic::Kinv() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = 1.0 / k();
    mat[0][2] = -_principal[0] / k();
    mat[1][2] = -_principal[1] / k();
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータ行列の転置の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の転置の逆行列，すなわち
		\f$
		\TUtinv{K}{} =
		\TUbeginarray{ccc}
		k^{-1} & & \\ & k^{-1} & \\ -k^{-1}u_0 & -k^{-1}v_0 & 1
		\TUendarray
		\f$
*/
Matrix33d
CameraWithEuclideanImagePlane::Intrinsic::Ktinv() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = 1.0 / k();
    mat[2][0] = -_principal[0] / k();
    mat[2][1] = -_principal[1] / k();
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータの自由度を返す．
/*!
  \return	内部パラメータの自由度，すなわち3
*/
u_int
CameraWithEuclideanImagePlane::Intrinsic::dof() const
{
    return 3;
}

//! 画像主点を返す．
/*!
  \return	画像主点
*/
Point2d
CameraWithEuclideanImagePlane::Intrinsic::principal() const
{
    return _principal;
}

//! 画像主点を設定する．
/*!
  \param u0	画像主点の横座標
  \param v0	画像主点の縦座標
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraWithEuclideanImagePlane::Intrinsic::setPrincipal(double u0, double v0)
{
    _principal[0] = u0;
    _principal[1] = v0;
    return *this;
}

//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す#dof()次元ベクトル
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraWithEuclideanImagePlane::Intrinsic::update(const Vector<double>& dp)
{
    CameraWithFocalLength::Intrinsic::update(dp(0, 1));
    _principal[0] -= dp[1];
    _principal[1] -= dp[2];
    return *this;
}

//! 入力ストリームからカメラの内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
std::istream&
CameraWithEuclideanImagePlane::Intrinsic::get(std::istream& in)
{
    CameraWithFocalLength::Intrinsic::get(in);
    return in >> _principal;
}

//! 出力ストリームにカメラの内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
std::ostream&
CameraWithEuclideanImagePlane::Intrinsic::put(std::ostream& out) const
{
    CameraWithFocalLength::Intrinsic::put(out);
    std::cerr << "Principal point:";
    return out << _principal;
}
 
}
