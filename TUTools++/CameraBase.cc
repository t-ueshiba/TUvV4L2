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
 *  $Id: CameraBase.cc,v 1.16 2009-07-31 07:04:44 ueshiba Exp $
 */
#include "TU/Camera.h"

namespace TU
{
/************************************************************************
*  class CameraBase							*
************************************************************************/
CameraBase::~CameraBase()
{
}

//! 3次元ユークリッド空間からcanonical画像平面への投影行列を求める．
/*!
  \return	canonical画像平面への投影行列，すなわち
		\f$
		\TUvec{P}{c} = \TUtvec{R}{}
		\TUbeginarray{cc}
		\TUvec{I}{3\times 3} & -\TUvec{t}{}
		\TUendarray
		\f$
*/
Matrix34d
CameraBase::Pc() const
{
    Matrix34d	PP;
    PP(0, 0, 3, 3) = _Rt;
    PP[0][3] = -(_Rt[0] * _t);
    PP[1][3] = -(_Rt[1] * _t);
    PP[2][3] = -(_Rt[2] * _t);

    return PP;
}

//! カメラパラメータに関する投影点のcanonical画像座標の1階微分を求める．
/*!
  \param X	対象点の3次元位置
  \return	投影点のcanonical画像座標の1階微分を表す2x6ヤコビ行列，すなわち
		\f$
		\TUbeginarray{cc}
		\TUdisppartial{\TUvec{x}{}}{\TUvec{t}{}} &
		\TUdisppartial{\TUvec{x}{}}{\TUvec{\theta}{}}
		\TUendarray
		\f$
*/
Matrix<double>
CameraBase::jacobianPc(const Point3d& X) const
{
    const Vector3d&		dX = X - _t;
    const Vector<double>&	x  = _Rt * dX;
    Matrix<double>		J(2, 6);
    (J[0](0, 3) = (x[0] / x[2] * _Rt[2] - _Rt[0])) /= x[2];
    (J[1](0, 3) = (x[1] / x[2] * _Rt[2] - _Rt[1])) /= x[2];
    J[0](3, 3) = J[0](0, 3) ^ dX;
    J[1](3, 3) = J[1](0, 3) ^ dX;

    return J;
}

//! 点の3次元位置に関する投影点のcanonical画像座標の1階微分を求める．
/*!
  \param x	対象点の3次元位置
  \return	投影点のcanonical画像座標の1階微分を表す2x3ヤコビ行列，すなわち
		\f$\TUdisppartial{\TUvec{x}{}}{\TUvec{X}{}}\f$
*/
Matrix23d
CameraBase::jacobianXc(const Point3d& X) const
{
    const Vector<double>&	x = _Rt * (X - _t);
    Matrix23d			J;
    (J[0] = (_Rt[0] - x[0] / x[2] * _Rt[2])) /= x[2];
    (J[1] = (_Rt[1] - x[1] / x[2] * _Rt[2])) /= x[2];

    return J;
}
 
//! 全カメラパラメータに関する投影点の画像座標の1階微分を求める．
/*!
  \param x	対象点の3次元位置
  \return	投影点の画像座標の1階微分を表す2x(6+#dofIntrinsic())ヤコビ行列，
		すなわち
		\f$
		\TUbeginarray{ccc}
		\TUdisppartial{\TUvec{u}{}}{\TUvec{t}{}} &
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\theta}{}} &
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}}
		\TUendarray =
		\TUbeginarray{cc}
		\TUdisppartial{\TUvec{u}{}}{\TUvec{x}{}}
		\TUbeginarray{cc}
		\TUdisppartial{\TUvec{x}{}}{\TUvec{t}{}} &
		\TUdisppartial{\TUvec{x}{}}{\TUvec{\theta}{}}
		\TUendarray &
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}}
		\TUendarray
		\f$
*/
Matrix<double>
CameraBase::jacobianP(const Point3d& X) const
{
    const Point2d&		x  = xc(X);
    const Matrix<double>&	JK = intrinsic().jacobianK(x);
    Matrix<double>		J(2, 6 + JK.ncol());
    J(0, 0, 2, 6) = intrinsic().jacobianXC(x) * jacobianPc(X);
    J(0, 6, 2, JK.ncol()) = JK;

    return J;
}
	
//! 入力ストリームからカメラの外部／内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
std::istream&
CameraBase::get(std::istream& in)
{
    const double	RAD = M_PI / 180.0;
    Vector3d		axis;
    in >> _t >> axis;
    _Rt = Matrix33d::Rt(RAD*axis);
    return intrinsic().get(in);
}

//! 出力ストリームにカメラの外部／内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
std::ostream&
CameraBase::put(std::ostream& out) const
{
    const double	DEG = 180.0 / M_PI;
    std::cerr << "Position:       ";    out << _t;
    std::cerr << "Rotation(deg.): ";    out << DEG*_Rt.rot2axis();
    return intrinsic().put(out);
}

/************************************************************************
*  class CameraBase::Intrinsic						*
************************************************************************/
CameraBase::Intrinsic::~Intrinsic()
{
}

//! canonical画像座標系において表現された投影点の画像座標系における位置を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	xの画像座標系における2次元位置，すなわち
		\f$\TUvec{u}{} = {\cal K}(\TUvec{x}{})\f$
*/
Point2d
CameraBase::Intrinsic::operator ()(const Point2d& x) const
{
    return x;
}

//! canonical画像座標系において表現された投影点の画像座標系に放射歪曲を付加する．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	放射歪曲付加後の投影点の2次元位置
*/
Point2d
CameraBase::Intrinsic::xd(const Point2d& x) const
{
    return x;
}

//! 内部パラメータに関する投影点の画像座標の1階微分を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	投影点のcanonical画像座標の1階微分を表す2x#dofIntrinsic()
		ヤコビ行列，すなわち
		\f$\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}}\f$
*/
Matrix<double>
CameraBase::Intrinsic::jacobianK(const Point2d& x) const
{
    return Matrix<double>(2, 0);
}

//! canonical画像座標に関する投影点の画像座標の1階微分を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	投影点のcanonical画像座標の1階微分を表す2x2ヤコビ行列，すなわち
		\f$\TUdisppartial{\TUvec{u}{}}{\TUvec{x}{}}\f$
*/
Matrix22d
CameraBase::Intrinsic::jacobianXC(const Point2d& x) const
{
    return Matrix22d::I(2);
}
    
//! 画像座標における投影点の2次元位置をcanonical画像座標系に直す．
/*!
  \param u	画像座標系における投影点の2次元位置
  \return	canonical画像座標系におけるuの2次元位置，すなわち
		\f$\TUvec{x}{} = {\cal K}^{-1}(\TUvec{u}{})\f$
*/
Point2d
CameraBase::Intrinsic::xcFromU(const Point2d& u) const
{
    return u;
}

//! 内部パラメータ行列を返す．
/*!
  \return	3x3内部パラメータ行列
*/
Matrix33d
CameraBase::Intrinsic::K() const
{
    return Matrix33d::I(3);
}
    
//! 内部パラメータ行列の転置を返す．
/*!
  \return	3x3内部パラメータ行列の転置
*/
Matrix33d
CameraBase::Intrinsic::Kt() const
{
    return Matrix33d::I(3);
}
    
//! 内部パラメータ行列の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の逆行列
*/
Matrix33d
CameraBase::Intrinsic::Kinv() const
{
    return Matrix33d::I(3);
}
    
//! 内部パラメータ行列の転置の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の転置の逆行列
*/
Matrix33d
CameraBase::Intrinsic::Ktinv() const
{
    return Matrix33d::I(3);
}

//! 内部パラメータの自由度を返す．
/*!
  \return	内部パラメータの自由度
*/
u_int
CameraBase::Intrinsic::dof() const
{
    return 0;
}

//! 焦点距離を返す．
/*!
  \return	焦点距離
*/
double
CameraBase::Intrinsic::k() const
{
    return 1.0;
}

//! 画像主点を返す．
/*!
  \return	画像主点
*/
Point2d
CameraBase::Intrinsic::principal() const
{
    return Point2d(0.0, 0.0);
}

//! アスペクト比を返す．
/*!
  \return	アスペクト比
*/
double
CameraBase::Intrinsic::aspect() const
{
    return 1.0;
}

//! 非直交歪みを返す．
/*!
  \return	非直交歪み
*/
double
CameraBase::Intrinsic::skew() const
{
    return 0.0;
}

//! 放射歪曲の第1係数を返す．
/*!
  \return	放射歪曲の第1係数
*/
double
CameraBase::Intrinsic::d1() const
{
    return 0.0;
}

//! 放射歪曲の第2係数を返す．
/*!
  \return	放射歪曲の第2係数
*/
double
CameraBase::Intrinsic::d2() const
{
    return 0.0;
}

//! 焦点距離を設定する．
/*!
  \param k	焦点距離
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraBase::Intrinsic::setFocalLength(double k)
{
    return *this;				// Do nothing.
}

//! 画像主点を設定する．
/*!
  \param u0	画像主点の横座標
  \param v0	画像主点の縦座標
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraBase::Intrinsic::setPrincipal(double u0, double v0)
{
    return *this;				// Do nothing.
}

//! アスペクト比を設定する．
/*!
  \param aspect	アスペクト比
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraBase::Intrinsic::setAspect(double aspect)
{
    return *this;				// Do nothing.
}

//! 非直交性歪みを設定する．
/*!
  \param skew	非直交性歪み
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraBase::Intrinsic::setSkew(double skew)
{
    return *this;				// Do nothing.
}

//! 放射歪曲係数以外の内部パラメータを設定する．
/*!
  \param K	3x3内部パラメータ行列
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraBase::Intrinsic::setIntrinsic(const Matrix33d& K)
{
    return *this;				// Do nothing.
}

//! 放射歪曲係数を設定する．
/*!
  \param d1	放射歪曲の第1係数
  \param d2	放射歪曲の第2係数
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraBase::Intrinsic::setDistortion(double d1, double d2)
{
    return *this;				// Do nothing.
}
	
//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す#dof()次元ベクトル
  \return	この内部パラメータ
*/
CameraBase::Intrinsic&
CameraBase::Intrinsic::update(const Vector<double>& dp)
{
    return *this;				// Do nothing.
}

//! 入力ストリームからカメラの内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
std::istream&
CameraBase::Intrinsic::get(std::istream& in)
{
    return in;					// Do nothing.
}

//! 出力ストリームにカメラの内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
std::ostream&
CameraBase::Intrinsic::put(std::ostream& out) const
{
    return out;					// Do nothing.
}
 
}
