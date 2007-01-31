/*
 *  $Id: MeasurementMatrix.cc,v 1.11 2007-01-31 05:42:44 ueshiba Exp $
 */
#include "TU/Calib++.h"
#include <iomanip>
#include <stdexcept>

namespace TU
{
/************************************************************************
*  Static functions							*
************************************************************************/
inline u_int	min(u_int a, u_int b)	{return (a < b ? a : b);}
inline int	sign(double x)		{return (x > 0 ? 1 : x < 0 ? -1 : 0);}

static Vector<double>
extpro(const Vector<double>& a, const Vector<double>& b)
{
    const u_int		d = a.dim();
    Vector<double>	v(d * (d+1) / 2);
    int			n = 0;
    for (int i = 0; i < d; ++i)
    {
	v[n++] = a[i]*b[i];
	for (int j = i+1; j < d; ++j)
	    v[n++] = a[i]*b[j] + a[j]*b[i];
    }
    return v;
}

/************************************************************************
*  Global functions							*
************************************************************************/
//! 観測行列をストリームに書き出す．
/*!
  \param out	出力ストリーム．
  \param Wt	観測行列．
  \return	outで与えた出力ストリーム．
*/
std::ostream&
operator <<(std::ostream& out, const MeasurementMatrix& Wt)
{
    using namespace	std;
	
    for (int j = 0; j < Wt.npoints(); ++j)
    {
	for (int i = 0; i < Wt.nframes(); ++i)
	{
	    out << ' ' << Wt[j][3*i  ]/Wt[j][3*i+2]
		<< ' ' << Wt[j][3*i+1]/Wt[j][3*i+2];
	}
	out << endl;
    }
    return out << endl;
}

/************************************************************************
*  class MeasurementMatrix						*
************************************************************************/
//! 既存の観測行列から一部のフレームを取り出して部分観測行列を生成する．
/*!
  \param Wt	既存の観測行列
  \param index	取り出すフレームのindexを格納した配列．index[0] = 0,
		index[1] = 1であればフレーム0, 1がWtから取り出される．
*/
MeasurementMatrix::MeasurementMatrix(const MeasurementMatrix& Wt,
				     const Array<u_int>&      index)
    :Matrix<double>(Wt.npoints(), min(3*index.dim(), Wt.ncol()))
{
    for (int i = 0; i < nframes(); ++i)
    {
	if (index[i] >= Wt.nframes())
	    throw std::invalid_argument("TU::MeasurementMatrix::MeasurementMatrix: Cannot find specified frame in given measurement matrix!!");
	(*this)(0, 3*i, npoints(), 3) = Wt(0, 3*index[i], Wt.npoints(), 3);
    }
}

//! 特徴点像の重心を計算する．
/*!
  \return	観測行列に含まれる特徴点の重心．各フレームにおける特徴点の
		重心の同時座標をフレーム順に並べた3F次元ベクトルとして返さ
		れる．
*/
Vector<double>
MeasurementMatrix::centroid() const
{
    Vector<ET>	c(ncol());
    for (int j = 0; j < npoints(); ++j)
	c += (*this)[j];
    return c /= npoints();
}

//! 指定されたフレームの特徴点像を取り出す．
/*!
  \param i	フレームのindex．
  \return	i番目のフレームに観測された特徴点を表す npoints() \f$\times\f$
		3行列．
*/
const Matrix<double>
MeasurementMatrix::frame(u_int i) const
{
    return (*this)(0, 3*i, npoints(), 3);
}

//! 指定されたフレームの特徴点像を取り出す．
/*!
  \param i	フレームのindex．
  \return	i番目のフレームに観測された特徴点を表す npoints() \f$\times\f$
		3行列．
*/
Matrix<double>
MeasurementMatrix::frame(u_int i)
{
    return (*this)(0, 3*i, npoints(), 3);
}

//! 2枚のフレーム間のアフィン基礎行列を計算する．
/*!
  \param frame0	一方のフレームのindex.
  \param frame1	もう一方のフレームのindex.
  \return	frame0, frame1間のアフィン基礎行列．frame0=\f$i\f$,
		frame1=\f$i'\f$のとき
		\f[
		\forall j~\TUtud{u}{i'j}\TUvec{F}{A}\TUud{u}{ij} \approx 0
		\f]
		を満たす\f$\TUvec{F}{A}\in\TUspace{R}{3\times3}\f$
		を返す．ただし，\f$\TUvec{F}{A}\f$の左上\f$2\times 2\f$
		部分は0．
*/
Matrix<double>
MeasurementMatrix::affineFundamental(u_int frame0, u_int frame1) const
{
    if (frame0 >= nframes() || frame1 >= nframes())
	throw std::invalid_argument("TU::MeasurementMatrix::affineFundamental: Cannot find specified frames!!");
    if (npoints() < 4)
	throw std::invalid_argument("TU::MeasurementMatrix::affineFundamental: At least 4 points needed!!");

    const Vector<ET>&	c = centroid();
    Matrix<ET>		u(npoints(), 4);
    for (int j = 0; j < npoints(); ++j)
    {
	u[j](0, 2) = (*this)[j](3*frame0, 2) - c(3*frame0, 2);
	u[j](2, 2) = (*this)[j](3*frame1, 2) - c(3*frame1, 2);
    }
    
    Matrix<ET>	A(4, 4);
    for (int j = 0; j < npoints(); ++j)
	A += u[j] % u[j];
    Vector<ET>	evalue;
    Vector<ET>	n = A.eigen(evalue)[3];
#ifdef DEBUG
    std::cerr << "-----------------" << std::endl
	      << "  Eigen values = " << evalue;
#endif
    Matrix<ET>	F(3, 3);
    
    F[2][0] = n[0];
    F[2][1] = n[1];
    F[0][2] = n[2];
    F[1][2] = n[3];
    F[2][2] = -(n[0]*c[3*frame0] + n[1]*c[3*frame0+1] +
		n[2]*c[3*frame1] + n[3]*c[3*frame1+1]);

    return F;
}

//! 2枚のフレーム間の基礎行列を計算する．
/*!
  \param frame0	一方のフレームのindex.
  \param frame1	もう一方のフレームのindex.
  \return	frame0, frame1間の基礎行列．frame0=\f$i\f$, frame1=\f$i'\f$
		のとき
		\f[
		\forall j~\TUtud{u}{i'j}\TUvec{F}{}\TUud{u}{ij} \approx 0
		\f]
		を満たす\f$\TUvec{F}{}\in\TUspace{R}{3\times3}\f$
		を返す．ただし，\f$\det\TUvec{F}{}=0\f$．
*/
Matrix<double>
MeasurementMatrix::fundamental(u_int frame0, u_int frame1) const
{
    if (frame0 >= nframes() || frame1 >= nframes())
	throw std::invalid_argument("TU::MeasurementMatrix::fundamental: Cannot find specified frames!!");
    if (npoints() < 8)
	throw std::invalid_argument("TU::MeasurementMatrix::fundamental: At least 8 points needed!!");

#ifdef DEBUG
    std::cerr << "=== BEGIN: TUMeasurementMatrix::fundamental ==="
	      << std::endl;
#endif
  // Normalize the input measurement matrix.
    Normalization	norm0(frame(frame0)), norm1(frame(frame1));
    Matrix<ET>		Wt(npoints(), 6);
    Wt(0, 0, npoints(), 3) = frame(frame0) * norm0.Tt();
    Wt(0, 3, npoints(), 3) = frame(frame1) * norm1.Tt();
    
    Matrix<ET>	F(3, 3), A(9, 9);
    Vector<ET>	weight(npoints());
    ET		c = 0.0;
    weight = 1.0 / npoints();
    for (;;)
    {
	Matrix<ET>	B(9, 9);
	A = 0.0;
	for (int i = 0; i < npoints(); ++i)
	{
	    Vector<ET>		v(9);
	    const Vector<ET>	&ldata = Wt[i](0, 3), &rdata = Wt[i](3, 3);
	    
	    v(0, 3) = rdata[0] * ldata;
	    v(3, 3) = rdata[1] * ldata;
	    v(6, 3) = rdata[2] * ldata;
	    A += (weight[i] * v) % v;

	    const Matrix<ET>&	tmp = (weight[i] * ldata) % ldata;
	    B(0, 0, 3, 3) += tmp;
	    B(3, 3, 3, 3) += tmp;
	    for (int j = 0; j < 3; ++j)
		for (int k = 0; k < 3; ++k)
		{
		    B[3*j  ][3*k  ] += weight[i] * rdata[j] * rdata[k];
		    B[3*j+1][3*k+1] += weight[i] * rdata[j] * rdata[k];
		}
	}
	A -= c * B;
	Vector<ET>	f((ET*)F, 9);
	Vector<ET>	evalue;
	f = A.eigen(evalue)[8];
#ifdef DEBUG
	std::cerr << " -----------------" << std::endl
		  << "   Weight       = " << weight
		  << "   Eigen values = " << evalue
		  << "   Error        = " << std::sqrt(c) << std::endl;
#endif
	if (fabs(evalue[8]) < 1.0e-12)
	    break;

	c += evalue[8] / (f * B * f);      // Update error.
	
      // Update weights.
      	ET	weight_sum = 0.0;
	for (int i = 0; i < npoints(); ++i)
	    weight_sum += (weight[i] =
			   1.0 / ((F(0, 0, 2, 3) * Wt[i](0, 3)).square() +
				  (Wt[i](3, 3) * F(0, 0, 3, 2)).square()));
	weight /= weight_sum;
    }
#ifdef DEBUG
    SVDecomposition<ET>	svdF(F);
    std::cerr << " Singular values of uncorrected F = " << svdF.diagonal();
#endif
    const Matrix<ET>&	Ainv = A.pinv(1.0e6);
    Vector<ET>		f((ET*)F, 9);
    
    for (;;)
    {
	const Matrix<ET>&	G = F.adj().trns();
	Vector<ET>		g((ET*)G, 9);
	const ET		detF = f * g / 3.0;
#ifdef DEBUG	
	std::cerr << " det(F) = " << detF << std::endl;
#endif
	if (fabs(detF) < 1.0e-12)
	    break;
	
	f -= detF / (g * Ainv * g) * (Ainv * g);
	f.normalize();
    }
#ifdef DEBUG    
    SVDecomposition<ET>	svdF2(F);
    std::cerr << " Singular values of corrected F = " << svdF2.diagonal();
    std::cerr << "=== END:   MeasurementMatrix::fundamental ==="
	      << std::endl;
#endif
    return norm1.Tt() * F * norm0.T();
}

//! 2枚のフレーム間のアフィン変換行列を計算する．
/*!
  \param frame0		一方のフレームのindex.
  \param frame1		もう一方のフレームのindex.
  \return		frame0, frame1間のアフィン変換行列．frame0=\f$i\f$,
			frame1=\f$i'\f$のとき
			\f[
			\forall j~\TUtud{u}{i'j}\approx\TUvec{A}{}\TUud{u}{ij}
			\f]
			を満たす\f$\TUvec{A}{}\in\TUspace{R}{3\times3}\f$を返
			す．ただし，\f$A_{31} = A_{32} = 0,~A_{33} = 1\f$．
*/
Matrix<double>
MeasurementMatrix::affinity(u_int frame0, u_int frame1) const
{
    if (frame0 >= nframes() || frame1 >= nframes())
	throw std::invalid_argument("TU::MeasurementMatrix::affinity: Cannot find specified frames!!");
    if (npoints() < 3)
	throw std::invalid_argument("TU::MeasurementMatrix::affinity: At least 3 points needed!!");

    const Vector<ET>&	c = centroid();
    Matrix<ET>		X(4, 4);
    Vector<ET>		a(4);
    for (int j = 0; j < npoints(); ++j)
    {
	const Vector<ET>&	dx = (*this)[j](3*frame0, 2) - c(3*frame0, 2);
	const Vector<ET>&	du = (*this)[j](3*frame1, 2) - c(3*frame1, 2);
	
	X(0, 0, 2, 2) += dx % dx;
	X(2, 2, 2, 2) += dx % dx;
	a(0, 2) += dx * du[0];
	a(2, 2) += dx * du[1];
    }
    a.solve(X);
    
    Matrix<ET>	A(3, 3);
    A[0][0] = a[0];
    A[0][1] = a[1];
    A[0][2] = c[3*frame1  ] - a(0, 2) * c(3*frame0, 2);
    A[1][0] = a[2];
    A[1][1] = a[3];
    A[1][2] = c[3*frame1+1] - a(2, 2) * c(3*frame0, 2);
    A[2][2] = 1.0;
    
    return A;
}

//! 2枚のフレーム間の射影変換行列を計算する．
/*!
  \param frame0		一方のフレームのindex.
  \param frame1		もう一方のフレームのindex.
  \param doRefinement	trueの場合，非線型最適化によるrefinementを行う．
  \return		frame0, frame1間の射影変換行列．frame0=\f$i\f$,
			frame1=\f$i'\f$のとき
			\f[
			\forall j~\TUtud{u}{i'j}\approx\TUvec{H}{}\TUud{u}{ij}
			\f]
			を満たす\f$\TUvec{H}{}\in\TUspace{R}{3\times3}\f$を返す．
*/
Matrix<double>
MeasurementMatrix::homography(u_int frame0, u_int frame1,
			      bool doRefinement) const
{
    if (frame0 >= nframes() || frame1 >= nframes())
	throw std::invalid_argument("TU::MeasurementMatrix::homography: Cannot find specified frames!!");
    if (npoints() < 4)
	throw std::invalid_argument("TU::MeasurementMatrix::homography: At least 4 points needed!!");

  // Normalize the input measurement matrix.
    Normalization	norm0(frame(frame0)), norm1(frame(frame1));
    Matrix<ET>		Wt(npoints(), 6);
    Wt(0, 0, npoints(), 3) = frame(frame0) * norm0.Tt();
    Wt(0, 3, npoints(), 3) = frame(frame1) * norm1.Tt();

    Matrix<ET>	Q(9, 9);
    for (int j = 0; j < npoints(); ++j)
    {
	const Vector<ET>	&ldata = Wt[j](0, 3), &rdata = Wt[j](3, 3);
	const Matrix<ET>	X = ldata % ldata;
	
	Q(0, 0, 3, 3) += (rdata[1]*rdata[1] + rdata[2]*rdata[2]) * X;
	Q(3, 3, 3, 3) += (rdata[2]*rdata[2] + rdata[0]*rdata[0]) * X;
	Q(6, 6, 3, 3) += (rdata[0]*rdata[0] + rdata[1]*rdata[1]) * X;
	Q(0, 3, 3, 3) = (Q(3, 0, 3, 3) -= rdata[0]*rdata[1] * X);
	Q(0, 6, 3, 3) = (Q(6, 0, 3, 3) -= rdata[2]*rdata[0] * X);
	Q(3, 6, 3, 3) = (Q(6, 3, 3, 3) -= rdata[1]*rdata[2] * X);
    }

    Matrix<ET>	H(3, 3);
    Vector<ET>	h((ET*)H, 9);
    Vector<ET>	evalue(9);
    h = Q.eigen(evalue)[8];
#ifdef DEBUG
    std::cerr << "TU::MeasurementMatrix::homography():: eigen values = "
	      << evalue;
#endif
    H = norm1.Tinv() * H * norm0.T();	// Unnormalize computed homography.

  // Nonlinear refinement based on MLE(optional).
    if (doRefinement)
    {
	CostH					err(*this, frame0, frame1);
	ConstNormConstraint<Vector<ET> >	g(h);
	minimizeSquare(err, g, h);
    }

  // Normalize the computed homography to make determinant unity.
    ET	det = H.det();
    if (det > 0)
	H /= pow(det, 1.0/3.0);
    else
	H /= -pow(-det, 1.0/3.0);
    
    return H;
}

//! 2枚のフレーム間の回転行列を計算する．
/*!
  \param frame0	一方のフレームのindex.
  \param frame1	もう一方のフレームのindex.
  \return	frame0, frame1間の回転行列．frame0=\f$i\f$,
		frame1=\f$i'\f$のとき
		\f[
		\forall j~\TUtud{u}{i'j} \approx \TUtvec{R}{}\TUud{u}{ij}
		\f]
		を満たす\f$\TUtvec{R}{}\in\TUspace{R}{3\times3}\f$を返す．
		ただし，\f$\TUtvec{R}{}\TUvec{R}{} = \TUvec{I}{3}\f$．
*/
Matrix<double>
MeasurementMatrix::rotation(u_int frame0, u_int frame1) const
{
    if (frame0 >= nframes() || frame1 >= nframes())
	throw std::invalid_argument("TU::MeasurementMatrix::rotation: Cannot find specified frames!!");
    if (npoints() < 3)
	throw std::invalid_argument("TU::MeasurementMatrix::rotation: At least 3 points needed!!");

    Matrix<ET>	M(3, 3);                        // Moment matrix.
    for (int j = 0; j < npoints(); ++j)
    {
	const Vector<ET>	&ldata = (*this)[j](3*frame0, 3),
				&rdata = (*this)[j](3*frame1, 3);
	M += ldata.normal() % rdata.normal();
    }
    SVDecomposition<ET>	svd(M);
    
    return svd.Ut().trns() * svd.Vt();
}

//! 観測行列をアフィンカメラ行列と特徴点の3次元アフィン座標に分解する．
/*!
  \param P	すべての視点におけるアフィンカメラ行列を，
		\f[
		  \TUvec{P}{} =
		  \TUbeginarray{c} \TUvec{P}{0} \\ \vdots \\ \TUvec{P}{F-1}
		  \TUendarray
		\f]
		(\f$\TUvec{P}{i}\in\TUspace{R}{3\times 4}\f$は視点iにおける
		アフィンカメラ行列)なる\f$3F\times 4\f$行列として返す．
  \param Xt	すべての特徴点の3次元アフィン空間での位置を，その同次座標を
		第j行とする\f$P\times 4\f$行列として返す．
*/
void
MeasurementMatrix::affineFactorization(Matrix<ET>& P, Matrix<ET>& Xt) const
{
    if (nframes() < 2)
	throw std::invalid_argument("TU::MeasurementMatrix::affineFactorization: Two or more frames required!!");
    
  // Make affine measurement matrix around the centroid.
    const Vector<ET>	c = centroid();
    Matrix<ET>		Wa(2*nframes(), npoints());
    for (int i = 0; i < nframes(); ++i)
	for (int j = 0; j < npoints(); ++j)
	{
	    Wa[2*i  ][j] = (*this)[j][3*i  ] - c[3*i  ];
	    Wa[2*i+1][j] = (*this)[j][3*i+1] - c[3*i+1];
	}

  // Factor the affine measurement matrix.
    SVDecomposition<ET> svd(Wa);
#ifdef DEBUG
    std::cerr << "  singular vaules of Wa: " << svd.diagonal();
#endif

  // Compute affine motion and shape.
    P.resize(3*nframes(), 4);
    Xt.resize( npoints(), 4);
    for (int i = 0; i < nframes(); ++i)
    {
	for (int n = 0; n < 3; ++n)
	{
	    P[3*i  ][n] = svd.Vt()[n][2*i];
	    P[3*i+1][n] = svd.Vt()[n][2*i+1];
	    P[3*i+2][n] = 0.0;
	}
	P[3*i  ][3] = c[3*i  ];
	P[3*i+1][3] = c[3*i+1];
	P[3*i+2][3] = 1.0;
    }
    for (int j = 0; j < npoints(); ++j)
    {
	for (int n = 0; n < 3; ++n)
	    Xt[j][n] = svd.Ut()[n][j] * svd[n];
	Xt[j][3] = 1.0;
    }
}

//! カメラ行列と特徴点の3次元座標をアフィン座標系からユークリッド座標系に直す．
/*!
  引数となるカメラ行列と特徴点座標はaffineFactorization()によって計算できる．
  ただし，そのときの観測行列に含まれる画像上での特徴点像の位置は，内部パラ
  メータの影響をキャンセルしたcanonical座標で表現されていなければならない．
  \param P	すべての視点におけるアフィンカメラ行列を\f$3F\times 4\f$行列
		として与える．これがユークリッド座標系に変換されて返される．
  \param Xt	すべての特徴点のアフィン座標を\f$P\times 4\f$行列として与える．
		これがユークリッド座標に変換されて返される．
*/
void
MeasurementMatrix::affineToMetric(Matrix<ET>& P, Matrix<ET>& Xt)
{
    const u_int	nframes = P.nrow() / 3;

    if (nframes < 3)
	throw std::invalid_argument("TU::MeasurementMatrix::affineToMetric: Three or more frames required!!");
    
  // Construct constraint matrix C.
    Matrix<ET>	C(2*nframes, 6);
    for (int i = 0; i < nframes; ++i)
    {
	const Vector<ET>	&p = P[3*i](0, 3), &q = P[3*i+1](0, 3);
	const ET		one_uu = 1.0 + P[3*i  ][3]*P[3*i  ][3],
				one_vv = 1.0 + P[3*i+1][3]*P[3*i+1][3],
				uv     = P[3*i][3]*P[3*i+1][3];
	
	C[2*i  ][0] =	 p[0]*p[0]/one_uu - q[0]*q[0]/one_vv;
	C[2*i  ][1] =	 p[1]*p[1]/one_uu - q[1]*q[1]/one_vv;
	C[2*i  ][2] =	 p[2]*p[2]/one_uu - q[2]*q[2]/one_vv;
	C[2*i  ][3] = 2*(p[1]*p[2]/one_uu - q[1]*q[2]/one_vv);
	C[2*i  ][4] = 2*(p[2]*p[0]/one_uu - q[2]*q[0]/one_vv);
	C[2*i  ][5] = 2*(p[0]*p[1]/one_uu - q[0]*q[1]/one_vv);
	C[2*i+1][0] = p[0]*q[0] - uv*(p[0]*p[0]/one_uu + q[0]*q[0]/one_vv)/2;
	C[2*i+1][1] = p[1]*q[1] - uv*(p[1]*p[1]/one_uu + q[1]*q[1]/one_vv)/2;
	C[2*i+1][2] = p[2]*q[2] - uv*(p[2]*p[2]/one_uu + q[2]*q[2]/one_vv)/2;
	C[2*i+1][3] = p[1]*q[2] + p[2]*q[1]
		    - uv*(p[1]*p[2]/one_uu + q[1]*q[2]/one_vv);
	C[2*i+1][4] = p[2]*q[0] + p[0]*q[2]
		    - uv*(p[2]*p[0]/one_uu + q[2]*q[0]/one_vv);
	C[2*i+1][5] = p[0]*q[1] + p[1]*q[0]
		    - uv*(p[0]*p[1]/one_uu + q[0]*q[1]/one_vv);
    }

  // Construct B.
    Vector<ET>		evalue;
    const Vector<ET>&	b = (C.trns()*C).eigen(evalue)[5];
#ifdef DEBUG
    std::cerr << "  eigen values of CC: " << evalue;
#endif
    Matrix<ET>	B(3, 3);
    B[0][0] = b[0];
    B[1][1] = b[1];
    B[2][2] = b[2];
    B[1][2] = B[2][1] = b[3];
    B[2][0] = B[0][2] = b[4];
    B[0][1] = B[1][0] = b[5];
    if (B.det() < 0.0)
	B *= -1.0;
    
  // Factor B into A and At s.t. B = A * At.
    Matrix<ET>	At(4, 4);
    At(0, 0, 3, 3) = B.cholesky();
    At[3][3] = 1.0;

  // Transform from affine to Euclidean world.
    P  *= At.trns();
    Xt *= At.inv();

  // Convert Affine projection matrices to Euclidean ones.
    for (int i = 0; i < nframes; ++i)
    {
	const ET	xc = P[3*i][3], yc = P[3*i+1][3];
	Matrix<ET>	Nt(3, 3);
	Nt[0] = P[3*i  ](0, 3);
	Nt[1] = P[3*i+1](0, 3);
	Nt[2] = -xc * Nt[0] - yc * Nt[1];

	SVDecomposition<ET>	svd(Nt);
	P(3*i, 0, 3, 3)  = svd.Vt().trns() * svd.Ut();
	P(3*i, 3, 3, 1) *= ((2.0 + xc*xc + yc*yc) / (svd[0] + svd[1]));
    }

  /* Fix WC to 0th camera
     and set distance between 0th and 1st camera to unity. */
    At(0, 0, 3, 3) = P(0, 0, 3, 3);	// rotation of 0th camera.
    At[3][0] = -P[0][3];
    At[3][1] = -P[1][3];
    At[3][2] = -P[2][3];
    At[3](0, 3) *= P(0, 0, 3, 3);	// t0: translation of 0th camera.
    Vector<ET>	t(3);
    t[0] = -P[3][3];
    t[1] = -P[4][3];
    t[2] = -P[5][3];
    (t *= P(3, 0, 3, 3)) -= At[3](0, 3);	// t1 - t0.
    At[3] /= t.length();
    
    P  *= At.trns();
    Xt *= At.inv();
}

//! 観測行列を透視投影カメラ行列と特徴点の3次元射影座標に分解する．
/*!
  \param P	すべての視点における透視投影カメラ行列を，
		\f[
		  \TUvec{P}{} =
		  \TUbeginarray{c} \TUvec{P}{0} \\ \vdots \\ \TUvec{P}{F-1}
		  \TUendarray
		\f]
		(\f$\TUvec{P}{i}\in\TUspace{R}{3\times 4}\f$は視点iにおける
		透視投影カメラ行列)なる\f$3F\times 4\f$行列として返す．
  \param Xt	すべての特徴点の3次元射影空間での位置を，その同次座標を第j行
		とする\f$P\times 4\f$行列として返す．
*/
void
MeasurementMatrix::projectiveFactorization(Matrix<ET>& P, Matrix<ET>& Xt) const
{
    if (nframes() < 2)
	throw std::invalid_argument("TU::MeasurementMatrix::projectiveFactorization: Two or more frames required!!");
    
  // Normalize the input measurement matrix.
    Array<Normalization>	norm(nframes());
    Matrix<ET>			Wt(npoints(), 3*nframes());
    for (int i = 0; i < nframes(); ++i)
    {
	norm[i].initialize(frame(i));
	Wt(0, 3*i, npoints(), 3) = frame(i) * norm[i].Tt();
    }

  // Minimize cost function.
    MeasurementMatrix::CostPF	f(Wt);
    Vector<ET>	mu((f.nframes() - 1) * (f.npoints() - 1));
    mu = 1.0;
    f.minimize(mu);

  // Decompose measurement matrix into motion(P) and shape(Xt).
    P.resize(3*nframes(), 4);
    Xt.resize( npoints(), 4);
    for (int n = 0; n < 4; ++n)
    {
	for (int i = 0; i < 3*nframes(); ++i)
	    P[i][n] = f.Vt()[n][i];
	for (int j = 0; j < npoints(); ++j)
	    Xt[j][n] = f.s()[n] * f.Ut()[n][j];
    }

  // Unnormalize camera matrices and recover image origins.
    for (int i = 0; i < nframes(); ++i)
	P(3*i, 0, 3, 4) = norm[i].Tinv() * P(3*i, 0, 3, 4);
}

//! カメラ行列と特徴点の3次元座標を射影座標系からユークリッド座標系に直す．
/*!
  引数となるカメラ行列と特徴点座標はprojectiveFactorization()によって計算で
  きる．ただし，そのときの観測行列に含まれる画像上での特徴点像の位置は，内部
  パラメータの影響をキャンセルしたcanonical座標で表現されていなければならない．
  \param P	すべての視点における透視投影カメラ行列を\f$3F\times 4\f$行列
		として与える．これがユークリッド座標系に変換されて返される．
  \param Xt	すべての特徴点の射影座標を\f$P\times 4\f$行列として与える．
		これがユークリッド座標に変換されて返される．
*/
void
MeasurementMatrix::projectiveToMetric(Matrix<ET>& P, Matrix<ET>& Xt) const
{
    if (nframes() < 2)
	throw std::invalid_argument("TU::MeasurementMatrix::projectiveToMetric: Two or more frames required!!");
    
  // Compute the focal point and the inverse projection of 0-th camera.
    SVDecomposition<ET>	svd(P(0, 0, 3, 4));
    const Vector<ET>&		c0 = svd.Ut()[3];
    const Matrix<ET>&		P0tinv = svd.Ut()[0] % svd.Vt()[0] / svd[0]
				       + svd.Ut()[1] % svd.Vt()[1] / svd[1]
				       + svd.Ut()[2] % svd.Vt()[2] / svd[2];

  // Initial guess of the parameter: q.
    Matrix<ET>	A(5*(nframes()-1), 4);
    Vector<ET>	b(5*(nframes()-1));
    for (int i = 0; i < nframes()-1; ++i)
    {
	const Matrix<ET>&	Pi = P(3*(i+1), 0, 3, 4);
	const Matrix<ET>&	Nt = Pi * P0tinv;
	const Vector<ET>&	e  = Pi * c0;
	
	A[5*i  ](0, 3)	= 2*e[0]*Nt[0] - 2*e[2]*Nt[2];
	A[5*i+1](0, 3)	= 2*e[1]*Nt[1] - 2*e[2]*Nt[2];
	A[5*i+2](0, 3)	= e[1]*Nt[0] + e[0]*Nt[1];
	A[5*i+3](0, 3)	= e[2]*Nt[0] + e[0]*Nt[2];
	A[5*i+4](0, 3)	= e[2]*Nt[1] + e[1]*Nt[2];
	A[5*i  ][3]	= e[0]*e[0] - e[2]*e[2];
	A[5*i+1][3]	= e[1]*e[1] - e[2]*e[2];
	A[5*i+2][3]	= e[0]*e[1];
	A[5*i+3][3]	= e[0]*e[2];
	A[5*i+4][3]	= e[1]*e[2];
	b[5*i  ]	= -Nt[0]*Nt[0] + Nt[2]*Nt[2];
	b[5*i+1]	= -Nt[1]*Nt[1] + Nt[2]*Nt[2];
	b[5*i+2]	= -Nt[0]*Nt[1];
	b[5*i+3]	= -Nt[0]*Nt[2];
	b[5*i+4]	= -Nt[1]*Nt[2];
    }
    Vector<ET>	q = b * A;
    q.solve(A.trns() * A);
#ifdef DEBUG	// Check the order of magnitute of last two eigen values.
    std::cerr << "q = " << q;
    std::cerr << "sqrt(fabs(q(0, 3)*q(0, 3) - q[3])) = "
	      << sqrt(fabs(q(0, 3)*q(0, 3) - q[3])) << std::endl;
#endif
    Vector<ET>		p = q(0, 3);
    CostPM		f(A, b);
    NullConstraint<ET>	g;
    minimizeSquare(f, g, p);
    
  // Construct projective transformation H.
    Matrix<ET>	Ht(4, 4);
    Ht(0, 0, 3, 4) = P0tinv.trns() + p % c0;
    Ht[3] = c0;
    
  // Transform from projective to Euclidean world.
    P  *= Ht.trns();
    Xt *= Ht.inv();

  // Determine sign of Ht[3] so that points are in front of the cameras.
    int	sig = 0;
    for (int j = 0; j < npoints(); ++j)
	sig += (Xt[j][3] > 0.0 ? 1 : Xt[j][3] < 0.0 ? -1 : 0);
    if (sig < 0)
    {
	P(0, 3, P.nrow(), 1) *= -1.0;
	Xt(0, 3, Xt.nrow(), 1) *= -1.0;
    }
	
  // Normalize so that 4-th component of Xt is 1.
    for (int j = 0; j < npoints(); ++j)
	Xt[j] /= Xt[j][3];

  // Extract translation and rotation components of camera motion.
    Array<CanonicalCamera>	cameras(nframes());
    for (int i = 0; i < nframes(); ++i)
	cameras[i].setProjection(P(3*i, 0, 3, 4));

    bundleAdjustment(cameras, Xt);

  // Set refined camera parameters.
    for (int i = 0; i < nframes(); ++i)
	P(3*i, 0, 3, 4) = cameras[i].P();
}

//! 個々のカメラの焦点距離を推定しつつ，カメラ行列と特徴点の3次元座標を射影座標系からユークリッド座標系に直す．
/*!
  引数となるカメラ行列と特徴点座標はprojectiveFactorization()によって計算で
  きる．ただし，そのときの観測行列に含まれる画像上での特徴点像の位置は，焦点
  距離以外の内部パラメータの影響をキャンセルした座標で表現されていなければな
  らない．焦点距離は個々のカメラ毎に異なっていてよい．
  \param P	すべての視点における透視投影カメラ行列を\f$3F\times 4\f$行列
		として与える．これがユークリッド座標系に変換されて返される．
  \param Xt	すべての特徴点の射影座標を\f$P\times 4\f$行列として与える．
		これがユークリッド座標に変換されて返される．
*/
void
MeasurementMatrix::projectiveToMetricWithFocalLengthsEstimation
			(Matrix<ET>& P, Matrix<ET>& Xt) const
{
    initializeFocalLengthsEstimation(P, Xt);
    
  // Extract translation and rotation components of camera motion.
    Array<CameraWithFocalLength>	cameras(nframes());
    for (int i = 0; i < nframes(); ++i)
	cameras[i].setProjection(P(3*i, 0, 3, 4));
    
    bundleAdjustment(cameras, Xt);

  // Set refined camera parameters.
    for (int i = 0; i < nframes(); ++i)
	P(3*i, 0, 3, 4) = cameras[i].P();
}

//! すべてのカメラに共通の焦点距離を推定しつつ，カメラ行列と特徴点の3次元座標を射影座標系からユークリッド座標系に直す．
/*!
  引数となるカメラ行列と特徴点座標はprojectiveFactorization()によって計算で
  きる．ただし，そのときの観測行列に含まれる画像上での特徴点像の位置は，焦点
  距離以外の内部パラメータの影響をキャンセルした座標で表現されていなければな
  らない．焦点距離はすべてのカメラで共通でなければならない．
  \param P	すべての視点における透視投影カメラ行列を\f$3F\times 4\f$行列
		として与える．これがユークリッド座標系に変換されて返される．
  \param Xt	すべての特徴点の射影座標を\f$P\times 4\f$行列として与える．
		これがユークリッド座標に変換されて返される．
*/
void
MeasurementMatrix::projectiveToMetricWithCommonFocalLengthEstimation
			(Matrix<ET>& P, Matrix<ET>& Xt) const
{
    initializeFocalLengthsEstimation(P, Xt);
    
  // Extract translation and rotation components of camera motion.
    Array<CanonicalCamera>	cameras(nframes());
    ET				k = 0.0;
    for (int i = 0; i < nframes(); ++i)
    {
	Matrix<ET>	Nt = P(3*i, 0, 3, 3);
	Nt[0] *= 0.5;
	Nt[1] *= 0.5;
	SVDecomposition<ET>	svd(Nt);
	if (svd.diagonal()[2] < 0.0)
	    throw std::domain_error("TU::MeasurementMatrix::projectiveToMetricWithCommonFocalLengthEstimation: cannot extract camera rotation due to nevative singular value!");
	const Matrix<ET>&	Rt = svd.Vt().trns() * svd.Ut();
	ET			p = Rt[0]*Nt[0] + Rt[1]*Nt[1], q = Rt[2]*Nt[2];
	Vector<ET>	t(3);
	t[0] = -P[3*i  ][3] / p;
	t[1] = -P[3*i+1][3] / p;
	t[2] = -P[3*i+2][3] / q;
	t *= Rt;
	cameras[i].setTranslation(t).setRotation(Rt);
	k += p / q;
    }

    CameraWithFocalLength::Intrinsic	K(k / nframes());
    bundleAdjustment(K, cameras, Xt);

  // Set refined camera parameters.
    for (int i = 0; i < nframes(); ++i)
    {
	P(3*i, 0, 3, 4) = cameras[i].P();
	P[3*i  ] *= K.k();
	P[3*i+1] *= K.k();
    }
}

//! 複数の画像上における特徴点の像の位置から，その3次元空間での位置を求める．
/*!
  カメラ行列Pは，どのような空間(射影空間，アフィン空間，ユークリッド空間)
  における表現でもよい．返される特徴点の3次元位置は，Pと同じ空間における表現
  となる．
  \param P		カメラ行列(フレーム数をFとすると\f$3F\times 4\f$行列)
  \param inhomogeneous	返される同次座標の第4要素を1にしたい場合はtrueを与え
			る．
  \return		特徴点の3次元位置を表す同次座標．
*/
Matrix<double>
MeasurementMatrix::reconstruction(const Matrix<ET>& P, bool inhomogeneous) const
{
    if (nframes() < 2)
	throw std::invalid_argument("TU::MeasurementMatrix::reconstruction: Two or more frames required!!");

    Matrix<ET>	Xt(npoints(), 4);
    for (int j = 0; j < npoints(); ++j)
    {
	Matrix<ET>	Q(2*nframes(), 4);
	for (int i = 0; i < nframes(); ++i)
	{
	    Q[2*i  ] = P[3*i  ] - (*this)[j][3*i  ] * P[3*i+2];
	    Q[2*i+1] = P[3*i+1] - (*this)[j][3*i+1] * P[3*i+2];
	}
	Vector<ET>	evalue;
	Xt[j] = (Q.trns() * Q).eigen(evalue)[3];
	if (inhomogeneous)
	    Xt[j] /= Xt[j][3];
    }
    
    return Xt;
}

//! 基礎行列の精度を評価する．
/*!
  精度は，対応する2点\f$\TUvec{u}{}\f$と点\f$\TUdvec{u}{}\f$について
  \f$\TUvec{u}{}\f$とそれに対応するエピポーラ線\f$\TUdtvec{u}{}\TUvec{F}{}\f$
  の距離および\f$\TUdvec{u}{}\f$とそれに対応するエピポーラ線
  \f$\TUvec{F}{}\TUvec{u}{}\f$の距離をすべての特徴点に対して計算し，その
  平均距離として評価する．
  \param F	評価の対象となる基礎行列．
  \param frame0	一方のフレームのindex.
  \param frame1	もう一方のフレームのindex.
  \return	特徴点像とエピポーラ線との距離の平均値．
 */
double
MeasurementMatrix::assessFundamental(const Matrix<ET>& F,
				     u_int frame0, u_int frame1) const
{
    ET	d_sum = 0.0;
    for (int j = 0; j < npoints(); ++j)
    {
	const Vector<ET>	&ldata = (*this)[j](3*frame0, 3),
				&rdata = (*this)[j](3*frame1, 3);
	Vector<ET>		l = F * ldata;
	d_sum += fabs(rdata * l) / l(0, 2).length();
	l = rdata * F;
	d_sum += fabs(l * ldata) / l(0, 2).length();
    }

    return d_sum / (2*npoints());
}

//! 射影変換行列の精度を評価する．
/*!
  精度は，対応する2点\f$\TUvec{u}{}\f$と点\f$\TUdvec{u}{}\f$について
  変換した特徴点像\f$\TUvec{H}{}\TUvec{u}{}\f$と実際の特徴点像
  \f$\TUdvec{u}{}\f$の距離をすべての特徴点に対して計算し，その平均距離として
  評価する．
  \param H	評価の対象となる射影変換行列．
  \param frame0	一方のフレームのindex.
  \param frame1	もう一方のフレームのindex.
  \return	変換した特徴点像と実際の特徴点像との距離の平均値．
 */
double
MeasurementMatrix::assessHomography(const Matrix<ET>& H,
				    u_int frame0, u_int frame1) const
{
    ET	d_sum = 0.0;
    for (int j = 0; j < npoints(); ++j)
    {
	const Vector<ET>	&ldata = (*this)[j](3*frame0, 3),
				&rdata = (*this)[j](3*frame1, 3);
	Vector<ET>		r = H * ldata;
	r /= r[2];
	d_sum += r.dist(rdata);
    }

    return d_sum / npoints();
}

//! カメラ行列の精度を評価する．
/*!
  精度は，異なる2つのフレーム間でassessFundamental()
  によって特徴点像とエピポーラ線の距離の平均値を計算し，これをすべての
  2フレームの組合わせに対して行ってその平均値により評価する．
  \param P	評価の対象となる\f$3F\times 4\f$カメラ行列．
  \return	特徴点像とエピポーラ線との距離の平均値．
 */
double
MeasurementMatrix::assessError(const Matrix<ET>& P) const
{
    u_int	ncombinations = 0;
    ET		d_sum = 0.0;
    for (int i = 0; i < nframes(); ++i)
    {
	const Matrix<ET>&	Pi = P(3*i, 0, 3, 4);
	
	for (int j = i+1; j < nframes(); ++j)
	{
	    const Matrix<ET>&		Pj = P(3*j, 0, 3, 4);
	    SVDecomposition<ET>	svd(Pi);
	    const Vector<ET>&		c = svd.Ut()[3];
	    const Matrix<ET>&		F = (Pj * c).skew() * Pj * Pi.trns()
					  * (Pi * Pi.trns()).inv();

	    d_sum += assessFundamental(F, i, j);
	    ++ncombinations;
	}
    }
    
    return d_sum / ncombinations;
}

/*
 *  private member functions.
 */
std::istream&
MeasurementMatrix::get(std::istream& in, int j, u_int nfrms)
{
    char	c;

    while (in.get(c))			// Skip white spaces other than '\n'.
	if (!isspace(c) || c == '\n')
	    break;

    if (!in || c == '\n')
    {
	resize(j, 3*nfrms);
	return in;
    }
    in.putback(c);
    Vector<ET> val;
    in >> val;
    get(in, j + 1, val.dim() / 2);
    for (int i = 0; i < nframes(); ++i)
    {
	(*this)[j][3*i]   = val[2*i];
	(*this)[j][3*i+1] = val[2*i+1];
	(*this)[j][3*i+2] = 1.0;
    }
    return in;
}    

Matrix<double>
MeasurementMatrix::initializeCalibrationWithPlanes
	(Array<CanonicalCamera>& cameras) const
{
    if (nframes() < 3)
	throw std::invalid_argument("TU::MeasurementMatrix::initializeCalibrationWithPlanes(): not enough frames!!");
	
  // Compute homographies.
    Array<Matrix<ET> >	Ht(nframes()-1);
    for (int i = 0; i < Ht.dim(); ++i)
	Ht[i] = homography(0, i+1).trns();
    
  // Compute IAG(Image of Absolute Conic).
    Matrix<ET>	A(2*Ht.dim(), 6);
    for (int i = 0; i < Ht.dim(); ++i)
    {
	A[2*i  ] = extpro(Ht[i][0], Ht[i][0]) - extpro(Ht[i][1], Ht[i][1]);
	A[2*i+1] = extpro(Ht[i][0], Ht[i][1]);
    }
    Vector<ET>		evalue;
    const Matrix<ET>&	evector = (A.trns() * A).eigen(evalue);
#ifdef DEBUG
    std::cerr << "TU::MeasurementMatrix::initializeCalibrationWithPlanes():: eigen values = "
	      << evalue;
#endif
    Vector<ET>		a;
    if (Ht.dim() < 3)			// two reference planes.
	a = evector[5][1] * evector[4] - evector[4][1] * evector[5];
    else
	a = evector[5];
    Matrix<ET>	omega(3, 3);
    omega[0][0]		      = a[0];
    omega[0][1] = omega[1][0] = a[1];
    omega[0][2] = omega[2][0] = a[2];
    omega[1][1]		      = a[3];
    omega[1][2] = omega[2][1] = a[4];
    omega[2][2]		      = a[5];
    if (omega.det() < 0.0)
	omega *= -1.0;

  // Extract intrinsic parameters from IAG.
    const Matrix<ET>	&Kinv = omega.cholesky(), &Ktinv = Kinv.trns();
    cameras.resize(Ht.dim());
    for (int i = 0; i < cameras.dim(); ++i)
    {
	const Matrix<ET>&	tmp = Ht[i] * Ktinv;
	SVDecomposition<ET>	svd(tmp(0, 0, 2, 3));
	Matrix<ET>		R(3, 3);
	R(0, 0, 2, 3) = svd.Vt().trns() * svd.Ut()(0, 0, 2, 3);
	R[2] = R[0] ^ R[1];
	
	const Vector<ET>&	t = -2.0 / (svd[0] + svd[1]) * R * tmp[2];

	cameras[i].setRotation(R.trns()).setTranslation(t);
    }

    Matrix<ET>	K(Kinv.inv());
    K /= K[2][2];

    return K;
}

void
MeasurementMatrix::initializeFocalLengthsEstimation(Matrix<ET>& P,
						    Matrix<ET>& Xt) const
{
    if (nframes() < 2)
	throw std::invalid_argument("TU::MeasurementMatrix::initializeFocalLengthsEstimation: Two or more frames required!!");
    
  // Compute B = H * Ht.
    Matrix<ET>	Q(4*nframes(), 10);
    for (int i = 0; i < nframes(); ++i)
    {
	Q[4*i  ] = extpro(P[3*i  ], P[3*i  ])
		 - extpro(P[3*i+1], P[3*i+1]);		// aa - bb
	Q[4*i+1] = extpro(P[3*i  ], P[3*i+1]);		// ab
	Q[4*i+2] = extpro(P[3*i  ], P[3*i+2]);		// ac
	Q[4*i+3] = extpro(P[3*i+1], P[3*i+2]);		// bc
    }
    Vector<ET>		evalue;
    const Vector<ET>	b = (Q.trns()*Q).eigen(evalue)[9];
#ifdef DEBUG	// Check the order of magnitute of last two eigen values.
    std::cerr << "evalue of QQ = " << evalue;
#endif
    Matrix<ET>	B(4, 4);
    B[0][0]	      = b[0];
    B[0][1] = B[1][0] = b[1];
    B[0][2] = B[2][0] = b[2];
    B[0][3] = B[3][0] = b[3];
    B[1][1]	      = b[4];
    B[1][2] = B[2][1] = b[5];
    B[1][3] = B[3][1] = b[6];
    B[2][2]	      = b[7];
    B[2][3] = B[3][2] = b[8];
    B[3][3]	      = b[9];

  // Factor B into H and Ht.
    Matrix<ET>	Ht = B.eigen(evalue);
    if (evalue[0] < 0.0)
	evalue *= -1.0;
#ifdef DEBUG	// First three eigen values should be positive.
    std::cerr << "evalue of B = " << evalue;
#endif
    for (int i = 0; i < 3; ++i)
	if (evalue[i] >= 0.0)
	    Ht[i] *= sqrt(evalue[i]);
	else
	    throw std::domain_error("TU::MeasurementMatirix::initializeFocalLengtsEstimation: cannot decompose B into H and Ht due to nevative eigen value!");

  // Compute inverse of H.
    Matrix<ET>	Hinv(4, 4);
    (Hinv[0] = Ht[0]) /= evalue[0];
    (Hinv[1] = Ht[1]) /= evalue[1];
    (Hinv[2] = Ht[2]) /= evalue[2];
    Hinv[3] = Ht[3];

  // Transform from projective to Euclidean world.
    P  *= Ht.trns();
    Xt *= Hinv.trns();

  // Ensure left 3*3 part of the projection matrix having positive determinant.
    for (int i = 0; i < nframes(); ++i)
	if (P(3*i, 0, 3, 3).det() < 0.0)
	    P(3*i, 0, 3, 4) *= -1.0;
    
  // Normalize so that 4-th component of Xt is 1.
    for (int j = 0; j < npoints(); ++j)
	Xt[j] /= Xt[j][3];

  // Determine sign of Ht[3] so that object points are in front of the cameras.
    int	sig = 0;
    for (int i = 0; i < nframes(); ++i)
	for (int j = 0; j < npoints(); ++j)
	{
#ifdef DEBUG
	    std::cerr << "depth[" << i << "][" << j << "] = " << P[3*i+2]*Xt[j]
		      << std::endl;
#endif
	    sig += sign(P[3*i+2] * Xt[j]);
	}
    if (sig < 0)
    {
	P(0, 3, P.nrow(), 1)   *= -1.0;
	Xt(0, 0, Xt.nrow(), 3) *= -1.0;
    }
}

/************************************************************************
*  class MeasurementMatrix::CostH					*
************************************************************************/
Vector<double>
MeasurementMatrix::CostH::operator ()(const AT& h) const
{
    Vector<ET>	val(2 * npoints());
    for (int j = 0; j < npoints(); ++j)
    {
	const Vector<ET>&	x = _Wt[j](3*_frame0, 3);
	ET			w = h(6, 3) * x;
	val[2*j  ] = h(0, 3) * x / w - _Wt[j][3*_frame1];
	val[2*j+1] = h(3, 3) * x / w - _Wt[j][3*_frame1+1];
    }
    
    return val;
}

Matrix<double>
MeasurementMatrix::CostH::jacobian(const AT& h) const
{
    Matrix<ET>	J(2 * npoints(), 9);
    for (int j = 0; j < npoints(); ++j)
    {
	const Vector<ET>&	x = _Wt[j](3*_frame0, 3);
	ET			w = h(6, 3) * x;
	J[2*j](0, 3) = J[2*j](6, 3) = J[2*j+1](3, 3) = J[2*j+1](6, 3) = x;
	(J[2*j  ](6, 3) *= (h(0, 3) * x)) /= -w;
	(J[2*j+1](6, 3) *= (h(3, 3) * x)) /= -w;
	J(2*j, 0, 2, 9) /= w;
    }
    
    return J;
}

void
MeasurementMatrix::CostH::update(AT& h, const Vector<ET>& dh) const
{
    ET	l = h.length();
    (h -= dh).normalize() *= l;
}

/************************************************************************
*  class MeasurementMatrix::CostPF					*
************************************************************************/
MeasurementMatrix::CostPF::CostPF(const Matrix<ET>& Wt0, u_int niter_max)
    :_niter_max(niter_max), _Wt0(Wt0), _s(), _Ut(), _Vt()
{
}

inline int
MeasurementMatrix::CostPF::frame_index(u_int p) const
{
    return p / (npoints()-1) + 1;
}

inline int
MeasurementMatrix::CostPF::point_index(u_int p) const
{
    return p % (npoints()-1) + 1;
}

/*
 *  Value of the cost function: J = \sum_{n=4}^N \sigma^2
 */
double
MeasurementMatrix::CostPF::operator ()(const Vector<ET>& mu)
{
  /* Compute projective measurement matrix. */
    Matrix<ET>	Wt(_Wt0);
    for (int p = 0; p < mu.dim(); ++p)
	Wt[point_index(p)](3*frame_index(p), 3) *= mu[p];

  /* Decompose it by using SVD. */
    SVDecomposition<ET>	svd(Wt.trns());
    _s  = svd.diagonal()(0, N());
    _Ut = svd.Ut();
    _Vt = svd.Vt();

    return _s(4, N()-4).square();
}

/*
 *  \TUvec{S}{m\dot np} =
 *	\TUtvec{v}{n}\TUpartial{\TUvec{W}{}}{\mu_p}\TUvec{u}{m}
 */
Matrix<double>
MeasurementMatrix::CostPF::S(int m) const
{
    Matrix<ET>	Sm(_Vt.nrow(), (nframes()-1)*(npoints()-1));
    
    for (int p = 0; p < Sm.ncol(); ++p)
    {
	int	i = frame_index(p), j = point_index(p);
	for (int n = 0; n < _Vt.nrow(); ++n)
	    Sm[n][p] = _Vt[n](3*i, 3) * _Wt0[j](3*i, 3) * _Ut[m][j];
    }
    return Sm;
}

/*
 *  \TUvec{T}{m\dot np} =
 *	\TUtvec{v}{m}\TUpartial{\TUvec{W}{}}{\mu_p}\TUvec{u}{n}
 */
Matrix<double>
MeasurementMatrix::CostPF::T(int m) const
{
    Matrix<ET>	Tm(_Ut.nrow(), (nframes()-1)*(npoints()-1));

    for (int p = 0; p < Tm.ncol(); ++p)
    {
	int	i = frame_index(p), j = point_index(p);
	for (int n = 0; n < _Ut.nrow(); ++n)
	    Tm[n][p] = _Vt[m](3*i, 3) * _Wt0[j](3*i, 3) * _Ut[n][j];
    }
    return Tm;
}

double
MeasurementMatrix::CostPF::minimize(Vector<ET>& mu)
{
    ET	val = (*this)(mu);
    
    for (int iter = 0; iter < _niter_max; ++iter)
    {
	print(iter, val, mu);
      //      	std::cerr << "    sigma = " << _s;
	
      /* Make a gradient vector and a Hessian matrix. */
	Vector<ET>	grad(mu.dim());
	Matrix<ET>	H(mu.dim(), mu.dim()), dH(mu.dim(), mu.dim());
	
	for (int p = 0; p < H.nrow(); ++p)
	{
	    H[p][p] = _Wt0[point_index(p)](3*frame_index(p), 3).square();
	    grad[p] = mu[p] * H[p][p];
	}
	
	for (int m = 0; m < 4; ++m)	// for first four sigular values...
	{
	    const Matrix<ET>	&Sm = S(m), &Tm = T(m);
	    
	    for (int p = 0; p < H.nrow(); ++p)
	    {
		for (int q = 0; q <= p; ++q)
		{
		    ET	ddsigma = 0.0;	// 2nd derivative of S-value
		    for (int n = 0; n < N(); ++n)
			if (n != m)
			    ddsigma += ((_s[m] * (Sm[n][p] * Sm[n][q] +
						  Tm[n][p] * Tm[n][q]) +
					 _s[n] * (Sm[n][p] * Tm[n][q] +
						  Tm[n][p] * Sm[n][q])) /
					((_s[m] + _s[n]) * (_s[m] - _s[n])));
		    H[p][q] -= (Sm[m][p] * Sm[m][q] + _s[m] * ddsigma);

		    ddsigma = 0.0;		// small term of ddsigma
		    for (int n = N(); n < _Vt.nrow(); ++n)
			ddsigma += (Sm[n][p] * Sm[n][q] / _s[m]);
		    for (int n = N(); n < _Ut.nrow(); ++n)
			ddsigma += (Tm[n][p] * Tm[n][q] / _s[m]);
		    dH[p][q] -= (_s[m] * ddsigma);
		}
		grad[p] -= (_s[m] * Sm[m][p]);
	    }
	}
	H.symmetrize();
	dH.symmetrize();

      /* Update projective depths: mu */
	Vector<ET>	dmu = grad;
	ET		val_next = (*this)(mu - dmu.solve(H+dH));
	if (val_next > val)
	{
	    dmu = grad;
	    val_next = (*this)(mu - dmu.solve(H));
	}
	mu -= dmu;

	if (fabs(val_next - val) < 1.0e-12)
	    return val_next;
	
	val = val_next;
    }

    std::cerr << "TU::MeasurementMatrix::CostPF::minimize: Too many iterations!!"
	      << std::endl;
    return val;
}

void
MeasurementMatrix::CostPF::print(int i, ET val, const Vector<ET>& mu) const
{
#ifdef DEBUG
    std::cerr << std::setw(3) << i << ": (" << val << ')' << mu;
#else
    std::cerr << i << '\t' << val << std::endl;
#endif
}

/************************************************************************
*  class MeasurementMatrix::CostPM					*
************************************************************************/
Vector<double>
MeasurementMatrix::CostPM::operator ()(const AT& p) const
{
    Vector<ET>	q(4);
    q(0, 3) = p;
    q[3] = p * p;

    return _AA * q - _b;
}

Matrix<double>
MeasurementMatrix::CostPM::jacobian(const AT& p) const
{
    Matrix<ET>	J;
    J = _AA(0, 0, _AA.nrow(), 3);
    for (int i = 0; i < J.nrow(); ++i)
	J[i] += (2.0 * _AA[i][3] * p);

    return J;
}

/************************************************************************
*  class MeasurementMatrix::CostCP					*
************************************************************************/
Vector<double>
MeasurementMatrix::CostCP::operator ()(const ATA& K,
				       const ATB& camera, int i) const
{
  // Compute errors for all points.
    Vector<ET>	val(2 * npoints());
    for (int j = 0; j < npoints(); ++j)
    {
	Vector<ET>	X(3);
	X(0, 2) = _Wt[j](0, 2);		// 3D reference points (X[2] = 0.0)
      	const Point2<ET>&	u = K(camera.xc(X));
	val[2*j]   = u[0] - _Wt[j][3*(i+1)];
	val[2*j+1] = u[1] - _Wt[j][3*(i+1)+1];
    }

    return val;
}

Matrix<double>
MeasurementMatrix::CostCP::jacobianA(const ATA& K,
				     const ATB& camera, int i) const
{
    Matrix<ET>		J(2 * npoints(), adim());
    for (int j = 0; j < npoints(); ++j)
    {
	Vector<ET>	X(3);
	X(0, 2) = _Wt[j](0, 2);		// 3D reference points (X[2] = 0.0)
	J(2*j, 0, 2, J.ncol()) = K.jacobianK(camera.xc(X));
    }
    
    return J;
}

Matrix<double>
MeasurementMatrix::CostCP::jacobianB(const ATA& K,
				     const ATB& camera, int i) const
{
    Matrix<ET>	KK(2 * npoints(), 6);
    for (int j = 0; j < npoints(); ++j)
    {
	Vector<ET>	X(3);
	X(0, 2) = _Wt[j](0, 2);		// 3D reference points (X[2] = 0.0)
	KK(2*j, 0, 2, KK.ncol())
	    = K.jacobianXC(camera.xc(X)) * camera.jacobianPc(X);
    }
    
    return KK;
}

void
MeasurementMatrix::CostCP::updateA(ATA& K, const Vector<ET>& dK) const
{
    K.update(dK);
}

void
MeasurementMatrix::CostCP::updateB(ATB& camera,
				   const Vector<ET>& dcamera) const
{
    camera.update(dcamera);
}

}
#if defined(__GNUG__) || defined(__INTEL_COMPILER)
#  include "TU/Array++.cc"
#  include "TU/Calib++.cc"
namespace TU
{
template Camera::Intrinsic
MeasurementMatrix::calibrateWithPlanes<Camera::Intrinsic>
    (Array<CanonicalCamera>& cameras, bool doRefinement) const;
template CameraWithDistortion::Intrinsic
MeasurementMatrix::calibrateWithPlanes<CameraWithDistortion::Intrinsic>
    (Array<CanonicalCamera>& cameras, bool doRefinement) const;

template void
MeasurementMatrix::bundleAdjustment(Array<CanonicalCamera>& cameras,
				    Matrix<ET>& Xt) const;
template void
MeasurementMatrix::bundleAdjustment(Array<CameraWithFocalLength>& cameras,
				    Matrix<ET>& Xt) const;

template void
MeasurementMatrix::bundleAdjustment(CameraWithFocalLength::Intrinsic& K,
				    Array<CanonicalCamera>&,
				    Matrix<ET>& Xt) const;

template void
MeasurementMatrix::bundleAdjustmentWithFixedCameraCenters
    (Array<CanonicalCamera>& cameras, Matrix<ET>& Xt) const;
template void
MeasurementMatrix::bundleAdjustmentWithFixedCameraCenters
    (Array<CameraWithFocalLength>& cameras, Matrix<ET>& Xt) const;
}

#endif
