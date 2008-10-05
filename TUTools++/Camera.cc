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
 *  $Id: Camera.cc,v 1.12 2008-10-05 23:25:16 ueshiba Exp $
 */
#include "TU/Camera.h"

namespace TU
{
/************************************************************************
*  class Camera								*
************************************************************************/
CameraBase&
Camera::setProjection(const Matrix34d& PP)
{
    Matrix33d	KK;		// camera intrinsic parameters.
    KK[0]    = PP[2](0, 3);
    KK[1]    = PP[1](0, 3);
    KK[2]    = PP[0](0, 3);
    QRDecomposition<double>	qr(KK);
    KK[0][0] =  qr.Rt()[2][2];
    KK[0][1] =  qr.Rt()[2][1];
    KK[0][2] = -qr.Rt()[2][0];
    KK[1][0] =  0.0;
    KK[1][1] =  qr.Rt()[1][1];
    KK[1][2] = -qr.Rt()[1][0];
    KK[2][0] =  0.0;
    KK[2][1] =  0.0;
    KK[2][2] = -qr.Rt()[0][0];

    Matrix33d	RRt;		// camera rotation.
    RRt[0]   =  qr.Qt()[2];
    RRt[1]   =  qr.Qt()[1];
    RRt[2]   = -qr.Qt()[0];

    Vector3d	tt;		// camera translation.
    tt[0]    = -PP[0][3];
    tt[1]    = -PP[1][3];
    tt[2]    = -PP[2][3];

  // Negate sign of PP so that KK has positive determinant.
    if (KK[0][0] * KK[1][1] * KK[2][2] < 0.0)
    {
	KK *= -1.0;
	tt *= -1.0;
    }
    
    if (KK[0][0] < 0.0)
    {
	KK[0][0] *= -1.0;
	RRt[0] *= -1.0;
    }
    if (KK[1][1] < 0.0)
    {
	KK[0][1] *= -1.0;
	KK[1][1] *= -1.0;
	RRt[1] *= -1.0;
    }
    if (KK[2][2] < 0.0)
    {
	KK[0][2] *= -1.0;
	KK[1][2] *= -1.0;
	KK[2][2] *= -1.0;
	RRt[2] *= -1.0;
    }
    tt = (KK.inv() * tt) * RRt;

    setIntrinsic(KK).setTranslation(tt).setRotation(RRt);

    return *this;
}

const CameraBase::Intrinsic&
Camera::intrinsic() const
{
    return _intrinsic;
}

CameraBase::Intrinsic&
Camera::intrinsic()
{
    return _intrinsic;
}

/************************************************************************
*  class Camera::Intrinsic						*
************************************************************************/
//! canonical座標系において表現された投影点の画像座標系における位置を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	xの画像座標系における位置，すなわち
		\f$
		\TUbeginarray{c} \TUvec{u}{} \\ 1 \TUendarray =
		\TUvec{K}{}
		\TUbeginarray{c} \TUvec{x}{} \\ 1 \TUendarray
		\f$
*/
Point2d
Camera::Intrinsic::operator ()(const Point2d& x) const
{
    return Point2d(_k00 * x[0] + _k01 * x[1] + principal()[0],
		   k() * x[1] + principal()[1]);
}

//! 内部パラメータに関する投影点の画像座標の1階微分を求める．
/*!
  ただし，アスペクト比aと焦点距離kの積ak, 非直交歪みsと焦点距離kの積skをそれぞれ
  第4, 第5番目の内部パラメータとして扱い，k, u0, v0, ak, skの5パラメータに関する
  1階微分としてヤコビ行列を計算する．
  \param x	canonical画像座標における投影点の2次元位置
  \return	投影点のcanonical画像座標の1階微分を表す2x5ヤコビ行列，すなわち
		\f$
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}} =
		\TUbeginarray{ccccc}
		& 1 & & x & y \\ y & & 1 & &
		\TUendarray
		\f$
*/
Matrix<double>
Camera::Intrinsic::jacobianK(const Point2d& x) const
{
    Matrix<double>	J(2, 5);
    J[1][0] = J[0][4] = x[1];
    J[0][1] = J[1][2] = 1.0;
    J[0][3] = x[0];

    return J;
}

//! canonical画像座標に関する投影点の画像座標の1階微分を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	投影点の画像座標の1階微分を表す2x2ヤコビ行列，すなわち
		\f$
		\TUdisppartial{\TUvec{u}{}}{\TUvec{x}{}} =
		\TUbeginarray{cc} ak & sk \\ & k \TUendarray
		\f$
*/
Matrix22d
Camera::Intrinsic::jacobianXC(const Point2d& x) const
{
    Matrix22d	J;
    J[0][0] = _k00;
    J[0][1] = _k01;
    J[1][1] = k();

    return J;
}

//! 画像座標における投影点の2次元位置をcanonical画像座標系に直す．
/*!
  \param u	画像座標系における投影点の2次元位置
  \return	canonical画像座標系におけるuの2次元位置，すなわち
		\f$
		\TUbeginarray{c} \TUvec{x}{c} \\ 1 \TUendarray =
		\TUinv{K}{}
		\TUbeginarray{c} \TUvec{u}{}  \\ 1 \TUendarray
		\f$
*/
Point2d
Camera::Intrinsic::xcFromU(const Point2d& u) const
{
    return Point2d((u[0] - principal()[0] -
		    (u[1] - principal()[1]) * _k01 / k()) / _k00,
		   (u[1] - principal()[1]) / k());
}

//! 内部パラメータ行列を返す．
/*!
  \return	3x3内部パラメータ行列，すなわち
		\f$
		\TUvec{K}{} =
		\TUbeginarray{ccc}
		ak & sk & u_0 \\ & k & v_0 \\ & & 1
		\TUendarray
		\f$
*/
Matrix33d
Camera::Intrinsic::K() const
{
    Matrix33d	mat;
    mat[0][0] = _k00;
    mat[0][1] = _k01;
    mat[0][2] = principal()[0];
    mat[1][1] = k();
    mat[1][2] = principal()[1];
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータ行列の転置を返す．
/*!
  \return	3x3内部パラメータ行列の転置，すなわち
		\f$
		\TUtvec{K}{} =
		\TUbeginarray{ccc}
		ak & & \\ sk & k & \\ u_0 & v_0 & 1
		\TUendarray
		\f$
*/
Matrix33d
Camera::Intrinsic::Kt() const
{
    Matrix33d	mat;
    mat[0][0] = _k00;
    mat[1][0] = _k01;
    mat[2][0] = principal()[0];
    mat[1][1] = k();
    mat[2][1] = principal()[1];
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータ行列の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の逆行列，すなわち
		\f$
		\TUinv{K}{} =
		\TUbeginarray{ccc}
		a^{-1}k^{-1} & -a^{-1}k^{-1}s &
		-a^{-1}k^{-1}(u_0 - s v_0) \\ & k^{-1} & -k^{-1}v_0 \\ & & 1
		\TUendarray
		\f$
*/
Matrix33d
Camera::Intrinsic::Kinv() const
{
    Matrix33d	mat;
    mat[0][0] = 1.0 / _k00;
    mat[0][1] = -_k01 / (_k00 * k());
    mat[0][2] = -principal()[0] * mat[0][0] - principal()[1] * mat[0][1];
    mat[1][1] = 1.0 / k();
    mat[1][2] = -principal()[1] / k();
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータ行列の転置の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の転置の逆行列，すなわち
		\f$
		\TUtinv{K}{} =
		\TUbeginarray{ccc}
		a^{-1}k^{-1} & & \\ -a^{-1}k^{-1}s & k^{-1} & \\
		-a^{-1}k^{-1}(u_0 - s v_0) & -k^{-1}v_0 & 1
		\TUendarray
		\f$
*/
Matrix33d
Camera::Intrinsic::Ktinv() const
{
    Matrix33d	mat;
    mat[0][0] = 1.0 / _k00;
    mat[1][0] = -_k01 / (_k00 * k());
    mat[2][0] = -principal()[0] * mat[0][0] - principal()[1] * mat[1][0];
    mat[1][1] = 1.0 / k();
    mat[2][1] = -principal()[1] / k();
    mat[2][2] = 1.0;

    return mat;
}

//! 内部パラメータの自由度を返す．
/*!
  \return	内部パラメータの自由度，すなわち5
*/
u_int
Camera::Intrinsic::dof() const
{
    return 5;
}

//! アスペクト比を返す．
/*!
  \return	アスペクト比
*/
double
Camera::Intrinsic::aspect() const
{
    return _k00 / k();
}

//! 非直交歪みを返す．
/*!
  \return	非直交歪み
*/
double
Camera::Intrinsic::skew() const
{
    return _k01 / k();
}

//! 焦点距離を設定する．
/*!
  \param kk	焦点距離
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
Camera::Intrinsic::setFocalLength(double kk)
{
    _k00 *= (kk / k());
    _k01 *= (kk / k());
    return CameraWithFocalLength::Intrinsic::setFocalLength(kk);
}

//! 与えられた内部パラメータ行列から内部パラメータを設定する．
/*!
  \param K	3x3内部パラメータ行列
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
Camera::Intrinsic::setIntrinsic(const Matrix33d& K)
{
    setAspect(K[0][0] / K[1][1])
	.setSkew(K[0][1] / K[1][1])
	.setPrincipal(K[0][2]/K[2][2], K[1][2]/K[2][2])
	.setFocalLength(K[1][1]/K[2][2]);

    return *this;
}

//! アスペクト比を設定する．
/*!
  \param aspect	アスペクト比
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
Camera::Intrinsic::setAspect(double aspect)
{
    _k00 = aspect * k();
    return *this;
}

//! 非直交性歪みを設定する．
/*!
  \param skew	非直交性歪み
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
Camera::Intrinsic::setSkew(double skew)
{
    _k01 = skew * k();
    return *this;
}

//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す#dof()次元ベクトル
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
Camera::Intrinsic::update(const Vector<double>& dp)
{
    CameraWithEuclideanImagePlane::Intrinsic::update(dp(0, 3));
    _k00 -= dp[3];
    _k01 -= dp[4];
    return *this;
}

//! 入力ストリームからカメラの内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
std::istream&
Camera::Intrinsic::get(std::istream& in)
{
    CameraWithEuclideanImagePlane::Intrinsic::get(in);
    in >> _k00 >> _k01;
    _k00 *= k();
    _k01 *= k();

    return in;
}

//! 出力ストリームにカメラの内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
std::ostream&
Camera::Intrinsic::put(std::ostream& out) const
{
    using namespace	std;
    
    CameraWithEuclideanImagePlane::Intrinsic::put(out);
    cerr << "Aspect ratio:    "; out << aspect() << endl;
    cerr << "Skew:            "; out << skew() << endl;

    return out;
}
 
}
