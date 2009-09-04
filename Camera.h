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
 *  $Id: Camera.h,v 1.6 2009-09-04 04:01:05 ueshiba Exp $
 */
#ifndef __TUCamera_h
#define __TUCamera_h

#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  class CameraBase							*
************************************************************************/
//! すべての透視投影カメラの基底となるクラス
class __PORT CameraBase
{
  public:
  //! カメラの内部パラメータを表すクラス
    class __PORT Intrinsic
    {
      public:
	virtual ~Intrinsic()						;
	
      // various operations.
	virtual Point2d		operator ()(const Point2d& x)	const	;
	virtual Point2d		xd(const Point2d& x)		const	;
	virtual Matrix<double>	jacobianK(const Point2d& x)	const	;
	virtual Matrix22d	jacobianXC(const Point2d& x)	const	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;

      // calibration matrices.    
	virtual Matrix33d	K()				const	;
	virtual Matrix33d	Kt()				const	;
	virtual Matrix33d	Kinv()				const	;
	virtual Matrix33d	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		k()				const	;
	virtual Point2d		principal()			const	;
	virtual double		aspect()			const	;
	virtual double		skew()				const	;
	virtual double		d1()				const	;
	virtual double		d2()				const	;
	virtual Intrinsic&	setFocalLength(double k)		;
	virtual Intrinsic&	setPrincipal(double u0, double v0)	;
	virtual Intrinsic&	setAspect(double aspect)		;
	virtual Intrinsic&	setSkew(double skew)			;
	virtual Intrinsic&	setIntrinsic(const Matrix33d& K)	;
	virtual Intrinsic&	setDistortion(double d1, double d2)	;

      // parameter updating functions.
	virtual Intrinsic&	update(const Vector<double>& dp)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;
    };
    
  public:
  //! 位置を原点に，姿勢を単位行列にセットして初期化する．
    CameraBase() :_t(), _Rt()	{_Rt[0][0] = _Rt[1][1] = _Rt[2][2] = 1.0;}

  //! 位置と姿勢をセットして初期化する．
  /*!
    \param t	カメラ位置を表す3次元ベクトル
    \param Rt	カメラ姿勢を表す3x3回転行列
  */
    CameraBase(const Point3d& t, const Matrix33d& Rt)	:_t(t), _Rt(Rt)	{}
    virtual ~CameraBase()						;
    
  // various operations in canonical coordinates.
    Point2d		xc(const Point3d& X)			const	;
    Point2d		xcFromU(const Point2d& u)		const	;
    Matrix34d		Pc()					const	;
    Matrix<double>	jacobianPc(const Point3d& X)		const	;
    Matrix23d		jacobianXc(const Point3d& X)		const	;

  // various oeprations in image coordinates.
    Point2d		operator ()(const Point3d& X)		const	;
    Matrix34d		P()					const	;
    Matrix<double>	jacobianP(const Point3d& X)		const	;
    Matrix<double>	jacobianFCC(const Point3d& X)		const	;
    Matrix23d		jacobianX(const Point3d& X)		const	;
    Matrix<double>	jacobianK(const Point3d& X)		const	;
    Matrix22d		jacobianXC(const Point3d& X)		const	;

  //! 投影行列をセットする．
  /*!
    \param P	3x4投影行列
    \return	このカメラ
  */
    virtual CameraBase& setProjection(const Matrix34d& P)		=0;

  // parameter updating functions.
    void		update(const Vector<double>& dp)		;
    void		updateFCC(const Vector<double>& dp)		;
    void		updateIntrinsic(const Vector<double>& dp)	;
    
  // calibration matrices.
  //! 内部パラメータ行列を返す．
  /*!
    \return	3x3内部パラメータ行列
  */
    Matrix33d		K()		const	{return intrinsic().K();}

  //! 内部パラメータ行列の転置を返す．
  /*!
    \return	3x3内部パラメータ行列の転置
  */
    Matrix33d		Kt()		const	{return intrinsic().Kt();}

  //! 内部パラメータ行列の逆行列を返す．
  /*!
    \return	3x3内部パラメータ行列の逆行列
  */
    Matrix33d		Kinv()		const	{return intrinsic().Kinv();}

  //! 内部パラメータ行列の転置の逆行列を返す．
  /*!
    \return	3x3内部パラメータ行列の転置の逆行列
  */
    Matrix33d		Ktinv()		const	{return intrinsic().Ktinv();}

  // extrinsic parameters.
  //! カメラの位置を返す．
  /*!
    \return	カメラの3次元位置
  */
    const Point3d&	t()		const	{return _t;}

  //! カメラの姿勢を返す．
  /*!
    \return	カメラの姿勢を表す3x3回転行列
  */
    const Matrix33d&	Rt()		const	{return _Rt;}
    CameraBase&		setTranslation(const Point3d& t)	;
    CameraBase&		setRotation(const Matrix33d& Rt)	;

  // intrinsic parameters.
  //! 内部パラメータを返す．
  /*!
    \return	内部パラメータ
  */
    virtual const Intrinsic&
			intrinsic()	const	= 0;

  //! 内部パラメータを返す．
  /*!
    \return	内部パラメータ
  */
    virtual Intrinsic&	intrinsic()		= 0;

  //! 内部パラメータの自由度を返す．
  /*!
    \return	内部パラメータの自由度
  */
    u_int		dofIntrinsic()	const	{return intrinsic().dof();}

  //! 焦点距離を返す．
  /*!
    \return	焦点距離
  */
    double		k()		const	{return intrinsic().k();}

  //! 画像主点を返す．
  /*!
    \return	画像主点
  */
    Point2d		principal()	const	{return intrinsic().principal();}

  //! アスペクト比を返す．
  /*!
    \return	アスペクト比
  */
    double		aspect()	const	{return intrinsic().aspect();}

  //! 非直交歪みを返す．
  /*!
    \return	非直交歪み
  */
    double		skew()		const	{return intrinsic().skew();}

  //! 放射歪曲の第1係数を返す．
  /*!
    \return	放射歪曲の第1係数
  */
    double		d1()		const	{return intrinsic().d1();}

  //! 放射歪曲の第2係数を返す．
  /*!
    \return	放射歪曲の第2係数
  */
    double		d2()		const	{return intrinsic().d2();}
    CameraBase&		setFocalLength(double k)		;
    CameraBase&		setPrincipal(double u0, double v0)	;
    CameraBase&		setAspect(double aspect)		;
    CameraBase&		setSkew(double skew)			;
    CameraBase&		setIntrinsic(const Matrix33d& K)	;
    CameraBase&		setDistortion(double d1, double d2)	;
    
  // I/O functions.
    std::istream&	get(std::istream& in)			;
    std::ostream&	put(std::ostream& out)		const	;

  private:
    Point3d		_t;			// camera center.
    Matrix33d		_Rt;			// camera orientation.
};

//! 3次元空間中の点の投影点のcanonical画像座標系における位置を求める．
/*!
  \param X	対象点の3次元位置
  \return	canonical画像座標系におけるxの投影点の位置，すなわち
		\f$
		\TUvec{x}{} = 
		\frac{1}{\TUtvec{r}{z}(\TUvec{X}{} - \TUvec{t}{})}
		\TUbeginarray{c}
		\TUtvec{r}{x}(\TUvec{X}{} - \TUvec{t}{}) \\
		\TUtvec{r}{y}(\TUvec{X}{} - \TUvec{t}{})
		\TUendarray
		\f$
*/
inline Point2d
CameraBase::xc(const Point3d& X) const
{
    const Vector<double>&	x = _Rt * (X - _t);
    return Point2d(x[0] / x[2], x[1] / x[2]);
}

//! 画像座標における投影点の2次元位置をcanonical画像座標系に直す．
/*!
  \param u	画像座標系における投影点の2次元位置
  \return	canonical画像カメラ座標系におけるuの2次元位置，すなわち
		\f$\TUvec{x}{} = {\cal K}^{-1}(\TUvec{u}{})\f$
*/
inline Point2d
CameraBase::xcFromU(const Point2d& u) const
{
    return intrinsic().xcFromU(u);
}

//! 3次元空間中の点の投影点の画像座標系における位置を求める．
/*!
  \param X	対象点の3次元位置
  \return	Xの投影点の画像座標系における位置，すなわち
		\f$\TUvec{u}{} = {\cal K}(\TUvec{x}{}(\TUvec{X}{}))\f$
*/
inline Point2d
CameraBase::operator ()(const Point3d& X) const
{
    return intrinsic()(xc(X));
}

//! 3次元ユークリッド空間から画像平面への投影行列を求める．
/*!
  \return	画像平面への投影行列，すなわち
  \f$
    \TUvec{P}{} = \TUvec{K}{}\TUtvec{R}{}
    \TUbeginarray{cc} \TUvec{I}{3\times 3} & -\TUvec{t}{} \TUendarray
  \f$
*/
inline Matrix34d
CameraBase::P() const
{
    return K() * Pc();
}

//! カメラ位置以外の全カメラパラメータに関する投影点の画像座標の1階微分を求める．
/*!
  \param X	対象点の3次元位置
  \return	投影点の画像座標の1階微分を表す2x(3+#dofIntrinsic())ヤコビ行列，
		すなわち
		\f$
		\TUbeginarray{ccc}
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\theta}{}} &
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}}
		\TUendarray =
		\TUbeginarray{ccc}
		\TUdisppartial{\TUvec{u}{}}{\TUvec{x}{}}
		\TUdisppartial{\TUvec{x}{}}{\TUvec{\theta}{}} &
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}}
		\TUendarray
		\f$
*/
inline Matrix<double>
CameraBase::jacobianFCC(const Point3d& X) const
{
    const Matrix<double>&	J = jacobianP(X);
    return Matrix<double>(J, 0, 3, J.nrow(), J.ncol() - 3);
}

//! 点の3次元位置に関する投影点の画像座標の1階微分を求める．
/*!
  \param X	対象点の3次元位置
  \return	投影点の画像座標の1階微分を表す2x3ヤコビ行列，すなわち
		\f$
		\TUdisppartial{\TUvec{u}{}}{\TUvec{X}{}} =
		\TUdisppartial{\TUvec{u}{}}{\TUvec{x}{}}
		\TUdisppartial{\TUvec{x}{}}{\TUvec{X}{}}
		\f$
*/
inline Matrix23d
CameraBase::jacobianX(const Point3d& X) const
{
    return intrinsic().jacobianXC(xc(X)) * jacobianXc(X);
}

//! 内部パラメータに関する投影点の画像座標の1階微分を求める
/*!
  \param X	対象点の3次元位置
  \return	投影点の画像座標の1階微分を表す2x#dofIntrinsic()ヤコビ行列，
		すなわち
		\f$\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}}\f$
*/
inline Matrix<double>
CameraBase::jacobianK(const Point3d& X) const
{
    return intrinsic().jacobianK(xc(X));
}

//! canonical画像座標に関する投影点の画像座標の1階微分を求める
/*!
  \param X	対象点の3次元位置
  \return	投影点の画像座標の1階微分を表す2x2ヤコビ行列，すなわち
		\f$\TUdisppartial{\TUvec{u}{}}{\TUvec{x}{}}\f$
*/
inline Matrix22d
CameraBase::jacobianXC(const Point3d& X) const
{
    return intrinsic().jacobianXC(xc(X));
}

//! 内部パラメータを指定された量だけ更新する．
/*!
  \f$\Delta\TUvec{p}{} = \Delta\TUvec{\kappa}{}\f$に対して
  \f$\TUvec{\kappa}{} \leftarrow \TUvec{\kappa}{} - \Delta\TUvec{\kappa}{}\f$
  と更新する．
  \param dp	更新量を表す#dofIntrinsic()次元ベクトル
  \return	この内部パラメータ
*/
inline void
CameraBase::updateIntrinsic(const Vector<double>& dp)
{
    intrinsic().update(dp);			// update intrinsic parameters.
}

//! カメラの姿勢と内部パラメータを指定された量だけ更新する．
/*!
  \f$\Delta\TUvec{p}{} = [\Delta\TUtvec{\theta}{},
  ~\Delta\TUtvec{\kappa}{}]^\top\f$に対して
  \f{eqnarray*}
  \TUtvec{R}{} & \leftarrow &
  \TUtvec{R}{}\TUtvec{R}{}(\Delta\TUvec{\theta}{}) \\
  \TUvec{\kappa}{} & \leftarrow & \TUvec{\kappa}{} - \Delta\TUvec{\kappa}{}
  \f}
  と更新する．カメラの位置は更新されない．
  \param dp	更新量を表す3+#dofIntrinsic()次元ベクトル
  \return	この内部パラメータ
*/
inline void
CameraBase::updateFCC(const Vector<double>& dp)
{
    _Rt *= Matrix<double>::Rt(dp(0, 3));	// update rotation.
    updateIntrinsic(dp(3, dp.dim() - 3));	// update intrinsic parameters.
}

//! カメラの外部／内部パラメータを指定された量だけ更新する．
/*!
  \f$\Delta\TUvec{p}{} = [\Delta\TUtvec{t}{},~\Delta\TUtvec{\theta}{},
  ~\Delta\TUtvec{\kappa}{}]^\top\f$に対して
  \f{eqnarray*}
  \TUvec{t}{} & \leftarrow & \TUvec{t}{} - \Delta\TUvec{t}{} \\
  \TUtvec{R}{} & \leftarrow &
  \TUtvec{R}{}\TUtvec{R}{}(\Delta\TUvec{\theta}{}) \\
  \TUvec{\kappa}{} & \leftarrow & \TUvec{\kappa}{} - \Delta\TUvec{\kappa}{}
  \f}
  と更新する．
  \param dp	更新量を表す6+#dofIntrinsic()次元ベクトル
  \return	この内部パラメータ
*/
inline void
CameraBase::update(const Vector<double>& dp)
{
    _t -= dp(0, 3);				// update translation.
    updateFCC(dp(3, dp.dim() - 3));		// update other prameters.
}

//! カメラの位置を設定する．
/*!
  \param t	カメラの3次元位置
  \return	このカメラ
*/
inline CameraBase&
CameraBase::setTranslation(const Point3d& t)
{
    _t = t;
    return *this;
}

//! カメラの姿勢を設定する．
/*!
  \param Rt	カメラの姿勢を表す3x3回転行列
  \return	このカメラ
*/
inline CameraBase&
CameraBase::setRotation(const Matrix33d& Rt)
{
    _Rt = Rt;
    return *this;
}

//! 焦点距離を設定する．
/*!
  \param k	焦点距離
  \return	このカメラ
*/
inline CameraBase&
CameraBase::setFocalLength(double k)
{
    intrinsic().setFocalLength(k);
    return *this;
}

//! 画像主点を設定する．
/*!
  \param u0	画像主点の横座標
  \param v0	画像主点の縦座標
  \return	このカメラ
*/
inline CameraBase&
CameraBase::setPrincipal(double u0, double v0)
{
    intrinsic().setPrincipal(u0, v0);
    return *this;
}

//! アスペクト比を設定する．
/*!
  \param aspect	アスペクト比
  \return	このカメラ
*/
inline CameraBase&
CameraBase::setAspect(double aspect)
{
    intrinsic().setAspect(aspect);
    return *this;
}

//! 非直交性歪みを設定する．
/*!
  \param skew	非直交性歪み
  \return	このカメラ
*/
inline CameraBase&
CameraBase::setSkew(double skew)
{
    intrinsic().setSkew(skew);
    return *this;
}

//! 放射歪曲係数以外の内部パラメータを設定する．
/*!
  \param K	3x3内部パラメータ行列
  \return	このカメラ
*/
inline CameraBase&
CameraBase::setIntrinsic(const Matrix33d& K)
{
    intrinsic().setIntrinsic(K);
    return *this;
}

//! 放射歪曲係数を設定する．
/*!
  \param d1	放射歪曲の第1係数
  \param d2	放射歪曲の第2係数
  \return	このカメラ
*/
inline CameraBase&
CameraBase::setDistortion(double d1, double d2)
{
    intrinsic().setDistortion(d1, d2);
    return *this;
}

//! 入力ストリームからカメラの外部／内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param camera	外部／内部パラメータの読み込み先
  \return	inで指定した入力ストリーム
*/
inline std::istream&
operator >>(std::istream& in, CameraBase& camera)
{
    return camera.get(in);
}

//! 出力ストリームにカメラの外部／内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \param camera	外部／内部パラメータの書き出し元
  \return	outで指定した出力ストリーム
*/
inline std::ostream&
operator <<(std::ostream& out, const CameraBase& camera)
{
    return camera.put(out);
}

//! 入力ストリームからカメラの内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param camera	内部パラメータの読み込み先
  \return	inで指定した入力ストリーム
*/
inline std::istream&
operator >>(std::istream& in, CameraBase::Intrinsic& intrinsic)
{
    return intrinsic.get(in);
}

//! 出力ストリームにカメラの内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \param camera	内部パラメータの書き出し元
  \return	outで指定した出力ストリーム
*/
inline std::ostream&
operator <<(std::ostream& out, const CameraBase::Intrinsic& intrinsic)
{
    return intrinsic.put(out);
}

/************************************************************************
*  class CanonicalCamera						*
************************************************************************/
//! すべての内部パラメータが標準既定値(焦点距離とアスペクト比が1, 非直交歪みと放射歪曲係数が0, 画像主点が原点に一致)となる透視投影カメラを表すクラス
class __PORT CanonicalCamera : public CameraBase
{
  public:
  //! 位置を原点に，姿勢を単位行列にセットして初期化する．
    CanonicalCamera()	:CameraBase(), _intrinsic()	{}

  //! 位置と姿勢をセットして初期化する．
  /*!
    \param t	カメラの3次元位置
    \param Rt	カメラの姿勢を表す3x3回転行列
  */
    CanonicalCamera(const Point3d& t, const Matrix33d& Rt)
	:CameraBase(t, Rt), _intrinsic()		{}

  //! 投影行列をセットして初期化する．
  /*!
    \param P	3x4投影行列
  */
    CanonicalCamera(const Matrix34d& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}
    
    virtual CameraBase&	setProjection(const Matrix34d& P)		;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;

  private:
    Intrinsic	_intrinsic;
};

/************************************************************************
*  class CameraWithFocalLength						*
************************************************************************/
//! 焦点距離以外の内部パラメータが標準既定値となる透視投影カメラを表すクラス
class __PORT CameraWithFocalLength : public CameraBase
{
  public:
  //! 焦点距離のみから成る内部パラメータを表すクラス
    class __PORT Intrinsic : public CanonicalCamera::Intrinsic
    {
      public:
      //! 内部パラメータをセットして初期化する．
      /*!
	\param k	焦点距離
      */
	Intrinsic(double k=1.0)	:_k(k)					{}

      // various operations.
	virtual Point2d		operator ()(const Point2d& x)	const	;
	virtual Matrix<double>	jacobianK(const Point2d& x)	const	;
	virtual Matrix22d	jacobianXC(const Point2d& x)	const	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;

      // calibration matrices.
	virtual Matrix33d	K()				const	;
	virtual Matrix33d	Kt()				const	;
	virtual Matrix33d	Kinv()				const	;
	virtual Matrix33d	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		k()				const	;
	virtual	CameraBase::Intrinsic&
				setFocalLength(double k)		;

      // parameter updating functions.
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;

      private:
	double	_k;
    };
    
  public:
  //! 位置を原点に，姿勢を単位行列に，内部パラメータをデフォルト値にセットして初期化する．
    CameraWithFocalLength()	:CameraBase(), _intrinsic()	{}

  //! 外部／内部パラメータをセットして初期化する．
  /*!
    \param t	カメラの3次元位置
    \param Rt	カメラの姿勢を表す3x3回転行列
    \param k	焦点距離
  */
    CameraWithFocalLength(const Point3d& t,
			  const Matrix33d& Rt, double k=1.0)
	:CameraBase(t, Rt), _intrinsic(k)			{}

  //! 投影行列をセットして初期化する．
  /*!
    \param P	3x4投影行列
  */
    CameraWithFocalLength(const Matrix34d& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&		setProjection(const Matrix34d& P)	;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};

/************************************************************************
*  class CameraWithEuclideanImagePlane					*
************************************************************************/
//! 焦点距離と画像主点以外の内部パラメータが標準既定値となる透視投影カメラを表すクラス
class __PORT CameraWithEuclideanImagePlane : public CameraBase
{
  public:
  //! 焦点距離と画像主点から成る内部パラメータを表すクラス
    class __PORT Intrinsic : public CameraWithFocalLength::Intrinsic
    {
      public:
      //! 内部パラメータをセットして初期化する．
      /*!
	\param k	焦点距離
	\param u0	画像主点の横座標
	\param v0	画像主点の縦座標
      */
	Intrinsic(double k=1.0, double u0=0.0, double v0=0.0)
	    :CameraWithFocalLength::Intrinsic(k), _principal(u0, v0)	{}

      //! #CameraWithFocalLength型カメラの内部パラメータをセットして初期化する．
      /*!
	画像主点は(0, 0)に初期化される．
	\param intrinsic	#CameraWithFocalLengthの内部パラメータ
      */
	Intrinsic(const CameraWithFocalLength::Intrinsic& intrinsic)
	    :CameraWithFocalLength::Intrinsic(intrinsic),
	     _principal(0.0, 0.0)					{}
	
      // various operations.	
	virtual Point2d		operator ()(const Point2d& x)	const	;
	virtual Matrix<double>	jacobianK(const Point2d& x)	const	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;
    
      // calibration matrices.	
	virtual Matrix33d	K()				const	;
	virtual Matrix33d	Kt()				const	;
	virtual Matrix33d	Kinv()				const	;
	virtual Matrix33d	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual Point2d		principal()			const	;
	virtual CameraBase::Intrinsic&
				setPrincipal(double u0, double v0)	;

      // parameter updating functions.
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;

      private:
	Point2d	_principal;
    };
    
  public:
  //! 位置を原点に，姿勢を単位行列に，内部パラメータをデフォルト値にセットして初期化する．
    CameraWithEuclideanImagePlane()	:CameraBase(), _intrinsic()	{}

  //! 外部／内部パラメータをセットして初期化する．
  /*!
    \param t	カメラの3次元位置
    \param Rt	カメラの姿勢を表す3x3回転行列
    \param k	焦点距離
    \param u0	画像主点の横座標
    \param v0	画像主点の縦座標
  */
    CameraWithEuclideanImagePlane(const Point3d&	t,
				  const Matrix33d&	Rt,
				  double		k=1.0,
				  double		u0=0,
				  double		v0=0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0)			{}

  //! 投影行列をセットして初期化する．
  /*!
    \param P	3x4投影行列
  */
    CameraWithEuclideanImagePlane(const Matrix34d& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&	setProjection(const Matrix34d& P)		;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};
    
/************************************************************************
*  class Camera								*
************************************************************************/
//! 放射歪曲係数のみが標準既定値(0)となる透視投影カメラを表すクラス
class __PORT Camera : public CameraBase
{
  public:
  //! 放射歪曲係数意外の全内部パラメータを表すクラス
    class __PORT Intrinsic : public CameraWithEuclideanImagePlane::Intrinsic
    {
      public:
      //! 内部パラメータをセットして初期化する．
      /*!
	\param k	焦点距離
	\param u0	画像主点の横座標
	\param v0	画像主点の縦座標
	\param aspect	アスペクト比
	\param skew	非直交性歪み
      */
	Intrinsic(double k=1.0, double u0=0.0, double v0=0.0,
		  double aspect=1.0, double skew=0.0)
	    :CameraWithEuclideanImagePlane::Intrinsic(k, u0, v0),
	     _k00(aspect * k), _k01(skew * k)				{}

      //! #CameraWithEuclideanImagePlane型カメラの内部パラメータをセットして初期化する．
      /*!
	アスペクト比と非直交歪みは0に初期化される．
	\param intrinsic	#CameraWithEuclideanImagePlaneの内部パラメータ
      */
	Intrinsic(const CameraWithEuclideanImagePlane::Intrinsic& intrinsic)
	    :CameraWithEuclideanImagePlane::Intrinsic(intrinsic),
	     _k00(k()), _k01(0.0)					{}

      //! 内部パラメータをセットして初期化する．
      /*!
	\param K	3x3内部パラメータ行列
      */
	Intrinsic(const Matrix33d& K)
	    :CameraWithEuclideanImagePlane::Intrinsic(),
	     _k00(k()), _k01(0.0)			{setIntrinsic(K);}
	
      // various operations.
	virtual Point2d		operator ()(const Point2d& x)	const	;
	virtual Matrix<double>	jacobianK(const Point2d& x)	const	;
	virtual Matrix22d	jacobianXC(const Point2d& x)	const	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;

      // calibration matrices.
	virtual Matrix33d	K()				const	;
	virtual Matrix33d	Kt()				const	;
	virtual Matrix33d	Kinv()				const	;
	virtual Matrix33d	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		aspect()			const	;
	virtual double		skew()				const	;
	virtual	CameraBase::Intrinsic&
				setFocalLength(double k)		;
	virtual CameraBase::Intrinsic&
				setAspect(double aspect)		;
	virtual CameraBase::Intrinsic&
				setSkew(double skew)			;
	virtual CameraBase::Intrinsic&
				setIntrinsic(const Matrix33d& K)	;

      // parameter updating functions.
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;
    
      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;

      protected:
      //! 焦点距離とアスペクト比の積を返す．
      /*!
	\return		焦点距離kとアスペクト比aの積ak
      */
		double		k00()			const	{return _k00;}

      //! 焦点距離と非直交歪みの積を返す．
      /*!
	\return		焦点距離kと非直交歪みsの積sk
      */
		double		k01()			const	{return _k01;}
	
      private:
	double	_k00, _k01;
    };
    
  public:
  //! 位置を原点に，姿勢を単位行列に，内部パラメータをデフォルト値にセットして初期化する．
    Camera()	:CameraBase(), _intrinsic()			{}

  //! 外部／内部パラメータをセットして初期化する．
  /*!
    \param t	カメラの3次元位置
    \param Rt	カメラの姿勢を表す3x3回転行列
    \param k	焦点距離
    \param u0	画像主点の横座標
    \param v0	画像主点の縦座標
    \param a	アスペクト比
    \param s	非直交歪み
  */
    Camera(const Point3d&	t,
	   const Matrix33d&	Rt,
	   double		k=1.0,
	   double		u0=0,
	   double		v0=0,
	   double		aspect=1.0,
	   double		skew=0.0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0, aspect, skew)	{}

  //! 投影行列をセットして初期化する．
  /*!
    \param P	3x4投影行列
  */
    Camera(const Matrix34d& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&	setProjection(const Matrix34d& P)		;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};

/************************************************************************
*  class CameraWithDistortion						*
************************************************************************/
//! 放射歪曲係数を含む全内部パラメータが可変となる透視投影カメラを表すクラス
class __PORT CameraWithDistortion : public CameraBase
{
  public:
  //! 放射歪曲係数を含む全内部パラメータを表すクラス
    class __PORT Intrinsic : public Camera::Intrinsic
    {
      public:
      //! 内部パラメータをセットして初期化する．
      /*!
	\param k	焦点距離
	\param u0	画像主点の横座標
	\param v0	画像主点の縦座標
	\param aspect	アスペクト比
	\param skew	非直交性歪み
	\param d1	放射歪曲の第1係数
	\param d2	放射歪曲の第2係数
      */
	Intrinsic(double k=1.0, double u0=0.0, double v0=0.0,
		  double aspect=1.0, double skew=0.0,
		  double d1=0.0, double d2=0.0)
	    :Camera::Intrinsic(k, u0, v0, aspect, skew),
	     _d1(d1), _d2(d2)						{}

      //! #Camera型カメラの内部パラメータをセットして初期化する．
      /*!
	2つの放射歪曲係数は0に初期化される．
	\param intrinsic	#Cameraの内部パラメータ
      */
	Intrinsic(const Camera::Intrinsic& intrinsic)
	    :Camera::Intrinsic(intrinsic), _d1(0.0), _d2(0.0)		{}

      //! 内部パラメータをセットして初期化する．
      /*!
	2つの放射歪曲係数は0に初期化される．
	\param K	3x3内部パラメータ行列
      */
	Intrinsic(const Matrix33d& K)
	    :Camera::Intrinsic(), _d1(0.0), _d2(0.0)	{setIntrinsic(K);}
	
      // various operations.
	virtual Point2d		operator ()(const Point2d& x)	const	;
	virtual Point2d		xd(const Point2d& x)		const	;
	virtual Matrix<double>	jacobianK(const Point2d& x)	const	;
	virtual Matrix22d	jacobianXC(const Point2d& x)	const	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		d1()				const	;
	virtual double		d2()				const	;
	virtual CameraBase::Intrinsic&	
				setDistortion(double d1, double d2)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;

      private:
	double	_d1, _d2;
    };
    
  public:
  //! 位置を原点に，姿勢を単位行列に，内部パラメータをデフォルト値にセットして初期化する．
    CameraWithDistortion()	:CameraBase(), _intrinsic()		{}

  //! 外部／内部パラメータをセットして初期化する．
  /*!
    \param t	カメラの3次元位置
    \param Rt	カメラの姿勢を表す3x3回転行列
    \param k	焦点距離
    \param u0	画像主点の横座標
    \param v0	画像主点の縦座標
    \param a	アスペクト比
    \param s	非直交歪み
    \param d1	放射歪曲の第1係数
    \param d2	放射歪曲の第2係数
  */
    CameraWithDistortion(const Point3d&		t,
			 const Matrix33d&	Rt,
			 double			k=1.0,
			 double			u0=0,
			 double			v0=0,
			 double			aspect=1.0,
			 double			skew=0.0,
			 double			d1=0.0,
			 double			d2=0.0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0, aspect, skew, d1, d2)	{}

    CameraWithDistortion(const Matrix34d& P,
			 double d1=0.0, double d2=0.0)			;

    virtual CameraBase&		setProjection(const Matrix34d& P)	;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};
 
//! 投影行列と放射歪曲係数をセットして初期化する．
/*!
  \param P	3x4投影行列
  \param d1	放射歪曲の第1係数
  \param d2	放射歪曲の第2係数
*/
inline
CameraWithDistortion::CameraWithDistortion(const Matrix34d& P,
					   double d1, double d2)
    :CameraBase(), _intrinsic()
{
    setProjection(P);
    setDistortion(d1, d2);
}

}
#endif	/* !__TUCamera_h */
