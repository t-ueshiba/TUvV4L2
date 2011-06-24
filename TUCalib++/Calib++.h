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
 *  $Id: Calib++.h,v 1.14 2011-06-24 00:32:26 ueshiba Exp $
 */
#ifndef __TUCalibPP_h
#define __TUCalibPP_h

#include "TU/Camera.h"
#include "TU/BlockDiagonalMatrix++.h"
#include "TU/Minimize.h"

/*!
  \mainpage	libTUCalib++ - カメラキャリブレーション用ライブラリ

  libTUCalib++は，カメラの内部パラメータおよび外部パラメータを求める
  ために必用な様々なアルゴリズムを実装したキャリブレーションライブラリ
  である．
*/

namespace TU
{
/************************************************************************
*  class Normalization						*
************************************************************************/
/*!
  同次座標の正規化変換を表すクラス．\f$\TUud{x}{}=[\TUtvec{x}{}, 1]^\top~
  (\TUvec{x}{} \in \TUspace{R}{d})\f$に対して，以下のような平行移動と
  スケーリングを行う変換を表現する:
  \f[
	\TUud{y}{} =
	\TUbeginarray{c} s^{-1}(\TUvec{x}{} - \TUvec{c}{}) \\ 1	\TUendarray =
	\TUbeginarray{ccc}
	  s^{-1} \TUvec{I}{d} & -s^{-1}\TUvec{c}{} \\ \TUtvec{0}{d} & 1
	\TUendarray
	\TUbeginarray{c} \TUvec{x}{} \\ 1 \TUendarray =
	\TUvec{T}{}\TUud{x}{}
  \f]
*/
class Normalization
{
  public:
  //! 正規化変換オブジェクトを初期化する．
  /*!
    initialize()を行わないと使用できない．
  */
    Normalization()	:_scale(1.0), _centroid()	{}

  //! 与えられた点群の同次座標から計算した正規化変換で初期化する．
  /*!
    具体的にどのような変換が計算されるかはinitialize()を参照．
    \param points	点群の同次座標．点の個数をN, 点が属する空間の次元をd
			とすると，N\f$\times\f$(d+1)行列として与えられる．
  */
    Normalization(const Matrix<double>& points)
	:_scale(0.0), _centroid(points.ncol()-1)	{initialize(points);}

    void		initialize(const Matrix<double>& points)	;

    u_int		dim()					  const	;
    Matrix<double>	T()					  const	;
    Matrix<double>	Tt()					  const	;
    Matrix<double>	Tinv()					  const	;
    Matrix<double>	Ttinv()					  const	;

  private:
    double		_scale;
    Vector<double>	_centroid;
};

//! この正規化変換が適用される空間の次元を返す．
/*! 
  \return	空間の次元(返される値をdとすると，同時座標のベクトル
		としての次元はd+1)．
*/
inline u_int
Normalization::dim() const
{
    return _centroid.dim();
}

/************************************************************************
*  class MeasurementMatrix						*
************************************************************************/
/*!
  複数の視点から観測された複数の特徴点の像から構成される観測行列を表すクラス．
  視点iから観測したj番目の点の像を\f$\TUud{u}{ij} = [u_{ij}, v_{ij}, 1]^\top
  ~(i=0,\ldots,F-1; j=0,\ldots,P-1)\f$とする(フレーム数F, 特徴点数P)と，
  \f[
	\TUtvec{W}{} =
	\TUbeginarray{ccc}
	\TUtud{u}{00} & \cdots & \TUtud{u}{F-1~1} \\ \vdots & & \vdots \\
	\TUtud{u}{0~P-1} & \cdots & \TUtud{u}{F-1~P-1}
	\TUendarray
  \f]
  と定義される．
*/
class MeasurementMatrix : public Matrix<double>
{
  public:
    typedef double	ET;		//!< ベクトル, 行列の要素の型

  //! 観測行列を\f$0\times 0\f$行列(フレーム数と特徴点数が共に0)として初期化．
    MeasurementMatrix()	:Matrix<ET>()			{}
    MeasurementMatrix(const MeasurementMatrix& Wt,
		      const Array<u_int>&      index)	;

  //! 観測行列に含まれるフレーム数を返す．
  /*!
    \return	フレーム数(観測行列の列数の1/3)．
  */
    u_int		nframes()		const	{return ncol()/3;}

  //! 観測行列に含まれる特徴点数を返す．
  /*!
    \return	特徴点数(観測行列の行数)．
  */
    u_int		npoints()		const	{return nrow();}

    Vector<ET>		centroid()		const	;
    const Matrix<ET>	frame(u_int i)		const	;
    Matrix<ET>		frame(u_int i)			;

    
    Matrix<ET>	affineFundamental(u_int frame0=0, u_int frame1=1) const	;
    Matrix<ET>	fundamental(u_int frame0=0, u_int frame1=1)	const	;
    Matrix<ET>	affinity(u_int frame0=0, u_int frame1=1)	const	;
    Matrix<ET>	homography(u_int frame0=0, u_int frame1=1,
			   bool doRefinement=true)		const	;
    Matrix<ET>	rotation(u_int frame0=0, u_int frame1=1)	const	;

    template <class INTRINSIC> INTRINSIC
		calibrateWithPlanes(Array<CanonicalCamera>& cameras,
				    bool doRefinement=true)	const	;
    void	affineFactorization(Matrix<ET>& P,
				    Matrix<ET>& Xt)		const	;
    static void	affineToMetric(Matrix<ET>& P,
			       Matrix<ET>& Xt)				;
    void	projectiveFactorization(Matrix<ET>& P,
					Matrix<ET>& Xt)		const	;
    void	projectiveToMetric(Matrix<ET>& P,
				   Matrix<ET>& Xt)		const	;
    void	projectiveToMetricWithFocalLengthsEstimation
		    (Matrix<ET>& P, Matrix<ET>& Xt)		const	;
    void	projectiveToMetricWithCommonFocalLengthEstimation
		    (Matrix<ET>& P, Matrix<ET>& Xt)		const	;

    template <class INTRINSIC>
    void	refineCalibrationWithPlanes
				(INTRINSIC& K,
				 Array<CanonicalCamera>& cameras) const	;
    template <class CAMERA>
    void	bundleAdjustment(Array<CAMERA>& cameras,
				 Matrix<ET>& Xt)		  const	;
    template <class INTRINSIC>
    void	bundleAdjustment(INTRINSIC& K,
				 Array<CanonicalCamera>& cameras,
				 Matrix<ET>& Xt)		  const	;
    template <class CAMERA>
    void	bundleAdjustmentWithFixedCameraCenters
				(Array<CAMERA>& cameras,
				 Matrix<ET>& Xt)		  const	;
    Matrix<ET>	reconstruction(const Matrix<ET>& P,
			       bool inhomogeneous=false)	  const	;
    ET		assessFundamental(const Matrix<ET>& F,
				  u_int frame0=0, u_int frame1=1) const	;
    ET		assessHomography(const Matrix<ET>& H,
				 u_int frame0=0, u_int frame1=1)  const	;
    ET		assessError(const Matrix<ET>& P)		  const	;
    
    friend std::istream&
		operator >>(std::istream& in, MeasurementMatrix& Wt)	;

  private:
    std::istream&
		get(std::istream& in, int j, u_int nfrms)		;
    Matrix<ET>	initializeCalibrationWithPlanes
			    (Array<CanonicalCamera>& cameras)	  const	;
    void	initializeFocalLengthsEstimation(Matrix<ET>& P,
						 Matrix<ET>& Xt)  const	;

    class CostH		// cost function for homography estimation.
    {
      public:
      	typedef ET		value_type;
	typedef Vector<ET>	AT;

      public:
	CostH(const MeasurementMatrix& Wt, u_int frame0, u_int frame1)
	    :_Wt(Wt), _frame0(frame0), _frame1(frame1)			{}

	Vector<ET>	operator ()(const AT& h)		const	;
	Matrix<ET>	jacobian(const AT& h)			const	;
	void		update(AT& h, const Vector<ET>& dh)	const	;
	u_int		npoints()		const	{return _Wt.npoints();}
	
      private:
	const MeasurementMatrix&	_Wt;
	const u_int			_frame0, _frame1;
    };
    
    class CostPF	// cost function for projective factorization.
    {
      private:
	enum		{DEFAULT_NITER_MAX = 50};
    
      public:
	CostPF(const Matrix<ET>& Wt0, u_int niter_max=DEFAULT_NITER_MAX);
    
	ET			minimize(Vector<ET>& mu)		;
    
	u_int			nframes()	const	{return _Wt0.ncol()/3;}
	u_int			npoints()	const	{return _Wt0.nrow();}
	u_int			N()		const	{return _Wt0.ncol();}
	const Vector<ET>&	s()		const	{return _s;}
	const Matrix<ET>&	Ut()		const	{return _Ut;}
	const Matrix<ET>&	Vt()		const	{return _Vt;}
    
      private:
	int			frame_index(u_int p)		const	;
	int			point_index(u_int p)		const	;
	ET			operator ()(const Vector<ET>& mu)	;
	Matrix<ET>		S(int m)			const	;
	Matrix<ET>		T(int m)			const	;
	void			update(const Vector<ET>& mu)		;
	void			print(int i, ET val,
				      const Vector<ET>& mu)	const	;
    
	const u_int		_niter_max;
	const Matrix<ET>&	_Wt0;	// initial measurement matrix: W
	Vector<ET>		_s;	// singular values of W
	Matrix<ET>		_Ut;	// right basis of SVD
	Matrix<ET>		_Vt;	// left basis of SVD
    };

    class CostPM	// cost function for projective to metric conversion.
    {
      public:
      	typedef ET		value_type;
	typedef Vector<ET>	AT;

      public:
	CostPM(const Matrix<ET>& A, const Vector<ET>& b) :_AA(A), _b(b) {}
	
	Vector<ET>	operator ()(const AT& p)	const	;
	Matrix<ET>	jacobian(const AT& p)		const	;
	void		update(AT& p,
			       const Vector<ET>& dp)	const	{p -= dp;}

      private:
	const Matrix<ET>&	_AA;
	const Vector<ET>&	_b;
    };

    class CostCP	// cost function for refining calibration
    {			//   using multiple planar patterns.
      public:
      	typedef ET			value_type;
	typedef Matrix<ET>		jacobian_type;
	typedef CameraBase::Intrinsic	ATA;
	typedef CanonicalCamera		ATB;

      public:
	CostCP(const MeasurementMatrix& Wt, u_int adim)
	    :_Wt(Wt), _adim(adim)					{}

	Vector<ET>	operator ()(const ATA& K,
				    const ATB& camera, int i)	const	;
	Matrix<ET>	jacobianA(const ATA& K,
				  const ATB& camera, int i)	const	;
	Matrix<ET>	jacobianB(const ATA& K,
				  const ATB& camera, int i)	const	;
	void		updateA(ATA& K,
				const Vector<ET>& dK)		const	;
	void		updateB(ATB& camera,
				const Vector<ET>& dcamera)	const	;

	u_int		npoints()	const	{return _Wt.npoints();}
	u_int		adim()		const	{return _adim;}
	u_int		adims()		const	{return adim();}
	
      private:
	const MeasurementMatrix&	_Wt;	// measurement matrix.
	const u_int			_adim;
    };

    template <class CAMERA>
    class CostBA	// cost function for bundle adjustment.
    {
      public:
      	typedef ET			value_type;
	typedef BlockDiagonalMatrix<ET>	jacobian_type;
	typedef Array<CAMERA>		ATA;
	typedef Vector<ET>		ATB;

	class CostCD	// cost function for keeping distance between 0th
	{		//   and 1st cameras constant.
	  public:
	    CostCD(const ATA& p)  :_sqdist01(p[0].t().sqdist(p[1].t()))	{}
	
	    Vector<ET>	operator ()(const ATA& p) const
			{
			    Vector<ET>	val(1);
			    val[0] = p[0].t().sqdist(p[1].t()) - _sqdist01;
			    return val;
			}
	    Matrix<ET>	jacobian(const ATA& p) const
			{
			    Matrix<ET>	L(1, p[0].dofIntrinsic() +
					  (6+p[0].dofIntrinsic())*(p.dim()-1));
			    (L[0](p[0].dofIntrinsic(), 3) = p[1].t()-p[0].t())
				*= 2.0;
			    return L;
			}

	  private:
	    const ET	_sqdist01;
	};

      public:
	CostBA(const MeasurementMatrix& Wt,
	       u_int dofIntrinsic, bool fixCameraCenter=false)		;

	Vector<ET>	operator ()(const ATA& p,
				    const ATB& x, int j)	const	;
	BlockDiagonalMatrix<ET>
			jacobianA(const ATA& p,
				  const ATB& x, int j)		const	;
	Matrix<ET>	jacobianB(const ATA& p,
				  const ATB& x, int j)		const	;
	void		updateA(ATA& p,	const Vector<ET>& dp)	const	;
	void		updateB(ATB& x, const Vector<ET>& dx)	const	;

	u_int			nframes()	const	{return _Wt.nframes();}
	u_int			adim()		const	{return _adim;}
	const Array<u_int>&	adims()		const	{return _adims;}
    
      private:
	const MeasurementMatrix&	_Wt;	// measurement matrix.
	const bool			_fcc;	// fix camera center or not.
	u_int				_adim;	// dimension of parameters A.
	Array<u_int>			_adims;	// dimensions of block jacobian.
    };

    template <class INTRINSIC>
    class CostBACI	// cost function for bundle adjustment
    {			//   with common intrinsic parameters estimation.
      public:
	typedef ET			value_type;
	typedef Matrix<ET>		jacobian_type;
	
	struct ATA : public Array<CanonicalCamera>
	{
	    ATA(const INTRINSIC& KK,
		const Array<CanonicalCamera>& cc)
		:Array<CanonicalCamera>(cc), K(KK)			{}
	    
	    INTRINSIC			K;	// common intrinsic parameters.
	};
	typedef Vector<ET>		ATB;
    
	class CostCD	// cost function for keeping distance between 0th
	{		//   and 1st cameras constant.
	  public:
	    CostCD(const ATA& p) :_sqdist01(p[0].t().sqdist(p[1].t()))	{}
	
	    Vector<ET>	operator ()(const ATA& p) const
			{
			    Vector<ET>	val(1);
			    val[0] = p[0].t().sqdist(p[1].t()) - _sqdist01;
			    return val;
			}
	    Matrix<ET>	jacobian(const ATA& p) const
			{
			    Matrix<ET>	L(1, p.K.dof() + 6*(p.dim()-1));
			    (L[0](p.K.dof(), 3) = (p[1].t()-p[0].t()))
				*= 2.0;
			    return L;
			}

	  private:
	    const ET	_sqdist01;
	};

      public:
	CostBACI(const MeasurementMatrix& Wt)	:_Wt(Wt)		{}

	Vector<ET>	operator ()(const ATA& p,
				    const ATB& x, int j)	const	;
	Matrix<ET>	jacobianA(const ATA& p,
				  const ATB& x, int j)		const	;
	Matrix<ET>	jacobianB(const ATA& p,
				  const ATB& x, int j)		const	;
	void		updateA(ATA& p, const Vector<ET>& dp)	const	;
	void		updateB(ATB& x, const Vector<ET>& dx)	const	;

	u_int		nframes()	const	{return _Wt.nframes();}
	u_int		adim()		const	{return 1 + 6*(nframes() - 1);}
	u_int		adims()		const	{return adim();}
    
      private:
	const MeasurementMatrix&	_Wt;	// measurement matrix.
    };
};
 
//! 観測行列をストリームから読み込む
/*!
  \param in	入力ストリーム．
  \param Wt	読み込まれた観測行列．
  \return	inで与えた入力ストリーム．
*/
inline std::istream&
operator >>(std::istream& in, MeasurementMatrix& Wt)
{
    return Wt.get(in, 0, 0);
}

std::ostream&	operator <<(std::ostream& out, const MeasurementMatrix& Wt);

/************************************************************************
*  Calibration functions						*
************************************************************************/
void		get_flengths(const Matrix<double>& F,
			     double& kl, double& kr,
			     u_int commonFocalLengths=0);
Matrix<double>	get_principal(const Matrix<double>& F,
			      const Matrix<double>& ldata,
			      const Matrix<double>& rdata);
void		decompose_essential(const Matrix<double>& E,
				    const Matrix<double>& ldata,
				    const Matrix<double>& rdata,
				    Matrix<double>& Rt, Vector<double>& t);
void		decompose_homography(const Matrix<double>& H,
				     const Matrix<double>& ldata,
				     const Matrix<double>& rdata,
				     Matrix<double>& Rt,
				     Vector<double>& t,
				     Vector<double>& normal);
Matrix<double>	projection(const Matrix<double>& data);

/************************************************************************
*  Registration functions						*
************************************************************************/
double	euclidean_registration(const Matrix<double>& data0,
			       const Matrix<double>& data1,
			       Matrix<double>& Rt, Vector<double>& t);
double	similarity_registration(const Matrix<double>& data0,
				const Matrix<double>& data1,
				Matrix<double>& Rt, Vector<double>& t,
				double& s);
double	affine_registration(const Matrix<double>& data0,
			    const Matrix<double>& data1,
			    Matrix<double>& At, Vector<double>& b);

/************************************************************************
*  Assessment functions							*
************************************************************************/
void	assess_projection(const Matrix<double>& data, const Matrix<double>& P);

/************************************************************************
*  File I/O functions							*
************************************************************************/
Matrix<double>	get_Cdata(const char* filename, const char* suffix,
			  u_int ncol, u_int nrow_min);
Matrix<double>	get_full_Cdata(const char* filename,
			       const char* suffix, u_int nrow_min);
Matrix<double>	get_matrix(const char* filename, const char* suffix);
void		put_matrix(const char* filename, const char* suffix,
			   const Matrix<double>& m);
Matrix<double>	get_HomogeneousMatrix(const char* filename,
				      const char* suffix);
void		put_InHomogeneousMatrix(const char* filename,
					const char* suffix,
					const Matrix<double>& m);
}
#endif	// !__TUCalibPP_h
