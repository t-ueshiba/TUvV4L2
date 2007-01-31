/*
 *  $Id: Calib++.cc,v 1.3 2007-01-31 05:42:44 ueshiba Exp $
 */
#include "TU/Calib++.h"

namespace TU
{
/************************************************************************
*  class MeasurementMatrix						*
************************************************************************/
//! 複数の視点から平面パターンを観測してカメラキャリブレーションを行う．
/*!
  ワールド座標系を参照平面に固定し，平面上の特徴点の同次座標を
  \f$[X_j, Y_j, 0, 1]^\top~(j=0,\ldots,P-1)\f$とする．これを内部パラメータ
  を固定して視点iから観測した像を\f$\TUud{u}{ij}=[u_{ij},v_{ij},1]^\top~
  (i=0,\ldots,F-1)\f$とする．これらのデータより観測行列を
  \f[
    \TUtvec{W}{} =
    \TUbeginarray{cccccc}
      X_0     & Y_0     & 1 & \TUtud{u}{00}    & \cdots & \TUtud{u}{0~F-1} \\
      \vdots  & \vdots  & \vdots & \vdots      &	& \vdots	   \\
      X_{P-1} & Y_{P-1} & 1 & \TUtud{u}{P-1~0} & \cdots & \TUtud{u}{P-1~F-1}
    \TUendarray
  \f]
  と構成する．この観測行列に対して本アルゴリズム適用すると，カメラの内部
  パラメータおよび各視点でのカメラの外部パラメータが計算される．
  \param cameras	各視点について，参照平面に固定されたワールド座標系
			から見たカメラの外部パラメータが返される．
  \param doRefinement	trueの場合，非線型最適化によって内部/外部パラメータ
			をrefineする．
  \return		カメラの内部パラメータ．
 */
template <class INTRINSIC> INTRINSIC
MeasurementMatrix::calibrateWithPlanes(Array<CanonicalCamera>& cameras,
				       bool doRefinement)
    const
{
    INTRINSIC	K(initializeCalibrationWithPlanes(cameras));
    if (doRefinement)
	refineCalibrationWithPlanes(K, cameras);

    return K;
}

//! 平面上のパターンを複数の視点から観測した画像の像を用いて，カメラの内部パラメータと外部パラメータをrefineする．
/*!
  観測行列の形式についてはcalibrateWithPlanes()を参照．
  \param K	 カメラの内部パラメータの初期値を与える．その最適推定値が返
		 される．
  \param cameras 各視点について，参照平面に固定されたワールド座標系から見た
		 カメラの外部パラメータの初期値を与える．その最適推定値が
		 返される．
 */
template <class INTRINSIC> void
MeasurementMatrix::refineCalibrationWithPlanes
    (INTRINSIC& K, Array<CanonicalCamera>& cameras) const
{
    if (nframes() < 3)
	throw std::invalid_argument("TU::MeasurementMatrix::refineCalibrationWithPlanes: Two or more frames required!!");

    CostCP		err(*this, K.dof());
    NullConstraint<ET>	g;

    minimizeSquareSparse(err, g, K, cameras, 200);
}

//! カメラの全パラメータと特徴点位置の初期値を非線型最適化によりrefineする．
/*!
  \param cameras	個々のカメラの全パラメータの初期値を与える．その最
			適推定値が返される．
  \param Xt		特徴点のユークリッド座標系における同次座標の初期値を
			与える．その最適推定値が返される．
*/
template <class CAMERA> void
MeasurementMatrix::bundleAdjustment(Array<CAMERA>& cameras,
				    Matrix<ET>& Xt) const
{
    if (nframes() < 2)
	throw std::invalid_argument("TU::MeasurementMatrix::bundleAdjustment: Two or more frames required!!");

    CostBA<CAMERA>			err(*this, cameras[0].dofIntrinsic());
    typename CostBA<CAMERA>::CostCD	g(cameras);
    Matrix<ET>				shape(Xt, 0, 0, Xt.nrow(), 3);

    minimizeSquareSparse(err, g, cameras, shape, 200);
}

//! カメラの外部パラメータ，全てのカメラに共通な焦点距離および特徴点位置の初期値を非線型最適化によりrefineする．
/*!
  \param K		全てのカメラに共通な焦点距離の初期値を与える．その最
			適推定値が返される．
  \param cameras	個々のカメラの外部パラメータの初期値を与える．その最
			適推定値が返される．
  \param Xt		特徴点のユークリッド座標系における同次座標の初期値を
			与える．その最適推定値が返される．
*/
template <class INTRINSIC> void
MeasurementMatrix::bundleAdjustment(INTRINSIC& K,
				    Array<CanonicalCamera>& cameras,
				    Matrix<ET>& Xt) const
{
    if (nframes() < 2)
	throw std::invalid_argument("TU::MeasurementMatrix::bundleAdjustment (with common intrinsic parameters estimation): Two or more frames required!!");

    CostBACI<INTRINSIC>				err(*this);
    typename CostBACI<INTRINSIC>::ATA		params(K, cameras);
    typename CostBACI<INTRINSIC>::CostCD	g(params);
    Matrix<ET>					shape(Xt, 0, 0, Xt.nrow(), 3);

    minimizeSquareSparse(err, g, params, shape, 200);

    K = params.K;
    for (int i = 0; i < cameras.dim(); ++i)
	cameras[i] = params[i];
}

//! カメラの位置が不変との仮定のもとで，個々のカメラの全パラメータおよび特徴点位置の初期値を非線型最適化によりrefineする．
/*!
  \param cameras	個々のカメラの全パラメータの初期値を与える．その最
			適推定値が返される．
  \param Xt		特徴点のユークリッド座標系における同次座標の初期値を
			与える．その最適推定値が返される．
*/
template <class CAMERA> void
MeasurementMatrix::bundleAdjustmentWithFixedCameraCenters
	(Array<CAMERA>& cameras, Matrix<ET>& Xt) const
{
    if (nframes() < 3)
	throw std::invalid_argument("TU::MeasurementMatrix::bundleAdjustmentWithFixedCameraCenters: Three or more frames required!!");

    CostBA<CAMERA>	err(*this, cameras[0].dofIntrinsic(), true);
    NullConstraint<ET>	g;
    Matrix<ET>		shape(Xt, 0, 0, Xt.nrow(), 3);

    minimizeSquareSparse(err, g, cameras, shape, 200);
}

/************************************************************************
*  class MeasurementMatrix::CostBA<CAMERA>				*
************************************************************************/
template <class CAMERA>
MeasurementMatrix::CostBA<CAMERA>::CostBA(const MeasurementMatrix& Wt,
					  u_int dofIntrinsic,
					  bool fixCameraCenter)
    :_Wt(Wt), _fcc(fixCameraCenter), _adim(0), _adims(nframes())
{
    _adims[0] = dofIntrinsic + (_fcc ? 3 : 0);
    _adim += _adims[0];
    for (int i = 1; i < _adims.dim(); ++i)
    {
	_adims[i] = dofIntrinsic + (_fcc ? 3 : 6);
	_adim += _adims[i];
    }
}

template <class CAMERA> Vector<double>
MeasurementMatrix::CostBA<CAMERA>::operator ()(const ATA& p,
					       const ATB& x, int j) const
{
    Vector<ET>	val(2 * nframes());
    for (int i = 0; i < nframes(); ++i)
    {
	const Point2<ET>&	u = p[i](x);
	val[2*i]   = u[0] - _Wt[j][3*i];
	val[2*i+1] = u[1] - _Wt[j][3*i+1];
    }

    return val;
}

template <class CAMERA> BlockMatrix<double>
MeasurementMatrix::CostBA<CAMERA>::jacobianA(const ATA& p,
					     const ATB& x, int j) const
{
    BlockMatrix<ET>	J(nframes());
    J[0] = (_fcc ? p[0].jacobianFCC(x) : p[0].jacobianK(x));
    for (int i = 1; i < nframes(); ++i)
	J[i] = (_fcc ? p[i].jacobianFCC(x) : p[i].jacobianP(x));
    
    return J;
}

template <class CAMERA> Matrix<double>
MeasurementMatrix::CostBA<CAMERA>::jacobianB(const ATA& p,
					     const ATB& x, int j) const
{
    Matrix<ET>	K(2 * nframes(), 3);
    for (int i = 0; i < nframes(); ++i)
	K(2*i, 0, 2, K.ncol()) = p[i].jacobianX(x);

    return K;
}

template <class CAMERA> void
MeasurementMatrix::CostBA<CAMERA>::updateA(ATA& p, const Vector<ET>& dp) const
{
    int	d = 0;
    if (_fcc)
	p[0].updateFCC(dp(d, _adims[0]));
    else
	p[0].updateIntrinsic(dp(d, _adims[0]));
    d += _adims[0];
    for (int i = 1; i < _adims.dim(); ++i)
    {
      // Update camera parameters.
	if (_fcc)
	    p[i].updateFCC(dp(d, _adims[i]));
	else
	    p[i].update(dp(d, _adims[i]));
	d += _adims[i];
    }

    if (!_fcc)
    {
      // Force the distance between 0th and 1st cameras keep constant.
	Vector<ET>	t = p[1].t() - p[0].t();
	t.normalize() *= p[0].t().dist(p[1].t());
	p[1].setTranslation(p[0].t() + t);
    }
}

template <class CAMERA> void
MeasurementMatrix::CostBA<CAMERA>::updateB(ATB& x, const Vector<ET>& dx) const
{
    x -= dx;
}

/************************************************************************
*  class MeasurementMatrix::CostBACI<INTRINSIC>				*
************************************************************************/
template <class INTRINSIC> Vector<double>
MeasurementMatrix::CostBACI<INTRINSIC>::operator ()(const ATA& p,
						    const ATB& x, int j) const
{
    Vector<ET>	val(2 * nframes());
    for (int i = 0; i < nframes(); ++i)
    {
	const Point2<ET>&	u = p.K(p[i].xc(x));
	val[2*i]   = u[0] - _Wt[j][3*i];
	val[2*i+1] = u[1] - _Wt[j][3*i+1];
    }

    return val;
}

template <class INTRINSIC> Matrix<double>
MeasurementMatrix::CostBACI<INTRINSIC>::jacobianA(const ATA& p,
						  const ATB& x, int j) const
{
    Matrix<ET>	J(2 * nframes(), adim());
    J(0, 0, 2, p.K.dof()) = p.K.jacobianK(p[0].xc(x));
    for (int i = 1; i < nframes(); ++i)
    {
	const Point2<ET>& xc = p[i].xc(x);
	J(2*i, 0, 2, p.K.dof()) = p.K.jacobianK(xc);
	J(2*i, p.K.dof() + 6*(i-1), 2, 6)
	    = p.K.jacobianXC(xc) * p[i].jacobianPc(x);
    }
    
    return J;
}

template <class INTRINSIC> Matrix<double>
MeasurementMatrix::CostBACI<INTRINSIC>::jacobianB(const ATA& p,
						  const ATB& x, int j) const
{
    Matrix<ET>	K(2 * nframes(), 3);
    for (int i = 0; i < nframes(); ++i)
	K(2*i, 0, 2, K.ncol())
	    = p.K.jacobianXC(p[i].xc(x)) * p[i].jacobianXc(x);

    return K;
}

template <class INTRINSIC> void
MeasurementMatrix::CostBACI<INTRINSIC>::updateA(ATA& p,
						const Vector<ET>& dp) const
{
    p.K.update(dp(0, p.K.dof()));
    for (int i = 1; i < nframes(); ++i)
	p[i].update(dp(p.K.dof() + 6*(i-1), 6));

  // Force the distance between 0th and 1st cameras keep constant.
    Vector<ET>	t = p[1].t() - p[0].t();
    t.normalize() *= (p[0].t().dist(p[1].t()));
    p[1].setTranslation(p[0].t() + t);
}

template <class INTRINSIC> void
MeasurementMatrix::CostBACI<INTRINSIC>::updateB(ATB& x,
						const Vector<ET>& dx) const
{
    x -= dx;
}
 
}
