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
 *  $Id: CameraWithFocalLength.cc,v 1.12 2008-10-05 23:25:17 ueshiba Exp $
 */
#include "TU/Camera.h"

namespace TU
{
/************************************************************************
*  class CameraWithFocalLength						*
************************************************************************/
CameraBase&
CameraWithFocalLength::setProjection(const Matrix34d& PP)
{
    Matrix33d			Nt = PP(0, 0, 3, 3);
    Nt[0] *= 0.5;
    Nt[1] *= 0.5;
    SVDecomposition<double>	svd(Nt);
    if (svd[2] < 0.0)
	throw std::invalid_argument("CameraWithFocalLength::setProjection: cannot extract camera rotation due to nevative singular value!");
    const Matrix<double>&	RRt = svd.Vt().trns() * svd.Ut();
    const double		p = RRt[0]*Nt[0] + RRt[1]*Nt[1],
				q = RRt[2]*Nt[2];
    Vector3d			tc;
    tc[0] = -PP[0][3] / p;
    tc[1] = -PP[1][3] / p;
    tc[2] = -PP[2][3] / q;
    setFocalLength(p / q).setRotation(RRt).setTranslation(tc * RRt);

    return *this;
}

const CameraBase::Intrinsic&
CameraWithFocalLength::intrinsic() const
{
    return _intrinsic;
}

CameraBase::Intrinsic&
CameraWithFocalLength::intrinsic()
{
    return _intrinsic;
}

/************************************************************************
*  class CameraWithFocalLength::Intrinsic				*
************************************************************************/
//! canonical画像座標系において表現された投影点の画像座標系における位置を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	xの画像座標系における2次元位置，すなわち
		\f$\TUvec{u}{} = k\TUvec{x}{}\f$
*/
Point2d
CameraWithFocalLength::Intrinsic::operator ()(const Point2d& x) const
{
    return Point2d(_k * x[0], _k * x[1]);
}

//! 内部パラメータに関する投影点の画像座標の1階微分を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	投影点のcanonical画像座標の1階微分を表す2x1ヤコビ行列，すなわち
		\f$
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}} = \TUvec{x}{}
		\f$
*/
Matrix<double>
CameraWithFocalLength::Intrinsic::jacobianK(const Point2d& x) const
{
    Matrix<double>	J(2, 1);
    J[0][0] = x[0];
    J[1][0] = x[1];

    return J;
}

//! canonical画像座標に関する投影点の画像座標の1階微分を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	投影点のcanonical画像座標の1階微分を表す2x2ヤコビ行列，すなわち
		\f$
		\TUdisppartial{\TUvec{u}{}}{\TUvec{x}{}} =
		k\TUvec{I}{2\times 2}
		\f$
*/
Matrix22d
CameraWithFocalLength::Intrinsic::jacobianXC(const Point2d& x) const
{
    Matrix22d	J;
    return J.diag(_k);
}
    
//! 画像座標における投影点の2次元位置をcanonical画像座標系に直す．
/*!
  \param u	画像座標系における投影点の2次元位置
  \return	canonical画像座標系におけるuの2次元位置，すなわち
		\f$\TUvec{x}{} = k^{-1}\TUvec{u}{}\f$
*/
Point2d
CameraWithFocalLength::Intrinsic::xcFromU(const Point2d& u) const
{
    return Point2d(u[0] / _k, u[1] / _k);
}

//! 内部パラメータ行列を返す．
/*!
  \return	3x3内部パラメータ行列，すなわち
		\f$
		\TUvec{K}{} =
		\TUbeginarray{ccc} k & & \\ & k & \\ & & 1 \TUendarray
		\f$
*/
Matrix33d
CameraWithFocalLength::Intrinsic::K() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = _k;
    mat[2][2] = 1.0;

    return mat;
}
    
//! 内部パラメータ行列の転置を返す．
/*!
  \return	3x3内部パラメータ行列の転置，すなわち
		\f$
		\TUtvec{K}{} =
		\TUbeginarray{ccc} k & & \\ & k & \\ & & 1 \TUendarray
		\f$
*/
Matrix33d
CameraWithFocalLength::Intrinsic::Kt() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = _k;
    mat[2][2] = 1.0;

    return mat;
}
    
//! 内部パラメータ行列の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の逆行列，すなわち
		\f$
		\TUinv{K}{} =
		\TUbeginarray{ccc} k^{-1} & & \\ & k^{-1} & \\ & & 1 \TUendarray
		\f$
*/
Matrix33d
CameraWithFocalLength::Intrinsic::Kinv() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = 1.0 / _k;
    mat[2][2] = 1.0;

    return mat;
}
    
//! 内部パラメータ行列の転置の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の転置の逆行列，すなわち
		\f$
		\TUtinv{K}{} =
		\TUbeginarray{ccc} k^{-1} & & \\ & k^{-1} & \\ & & 1 \TUendarray
		\f$
*/
Matrix33d
CameraWithFocalLength::Intrinsic::Ktinv() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = 1.0 / _k;
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータの自由度を返す．
/*!
  \return	内部パラメータの自由度，すなわち1
*/
u_int
CameraWithFocalLength::Intrinsic::dof() const
{
    return 1;
}

//! 焦点距離を返す．
/*!
  \return	焦点距離
*/
double
CameraWithFocalLength::Intrinsic::k() const
{
    return _k;
}

//! 焦点距離を設定する．
/*!
  \param k	焦点距離
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraWithFocalLength::Intrinsic::setFocalLength(double k)
{
    _k = k;
    return *this;
}    

//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す#dof()次元ベクトル
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraWithFocalLength::Intrinsic::update(const Vector<double>& dp)
{
    _k -= dp[0];
    return *this;
}

//! 入力ストリームからカメラの内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
std::istream&
CameraWithFocalLength::Intrinsic::get(std::istream& in)
{
    CanonicalCamera::Intrinsic::get(in);
    return in >> _k;
}

//! 出力ストリームにカメラの内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
std::ostream&
CameraWithFocalLength::Intrinsic::put(std::ostream& out) const
{
    using namespace	std;
    
    CanonicalCamera::Intrinsic::put(out);
    cerr << "Focal length:    ";
    return out << _k << endl;
}
 
}
