/*
 *  $Id: LinearMapping.cc,v 1.1 2007-02-05 23:24:03 ueshiba Exp $
 */
#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  class ProjectiveMapping						*
************************************************************************/
//! 入力空間と出力空間の次元を指定して射影変換オブジェクトを生成する．
/*!
  恒等変換として初期化される．
  \param inDim	入力空間の次元
  \param outDim	出力空間の次元
*/
ProjectiveMapping::ProjectiveMapping(u_int inDim, u_int outDim)
    :_T(outDim + 1, inDim + 1)
{
    u_int	n = std::min(inDim, outDim);
    for (int i = 0; i < n; ++i)
	_T[i][i] = 1.0;
    _T[outDim][inDim] = 1.0;
}
    
/************************************************************************
*  class ProjectiveMapping::Cost					*
************************************************************************/
Vector<double>
ProjectiveMapping::Cost::operator ()(const AT& T) const
{
    const u_int	outDim = T.nrow() - 1;
    Vector<ET>	val(npoints()*outDim);
    for (int n = 0; n < npoints(); ++n)
    {
	const Vector<ET>&	y = T * _X[n];
	val(n*outDim, outDim) = y(0, outDim)/y[outDim] - _Y[n];
    }
    
    return val;
}
    
Matrix<double>
ProjectiveMapping::Cost::jacobian(const AT& T) const
{
    const u_int	inDim1 = T.ncol(), outDim = T.nrow() - 1;
    Matrix<ET>	J(npoints()*outDim, (outDim + 1)*inDim1);
    for (int n = 0; n < npoints(); ++n)
    {
	const Vector<ET>&	y = T * _X[n];

	for (int j = 0; j < outDim; ++j)
	{
	    Vector<ET>&	tmp = J[n*outDim + j];
	    
	    tmp(j*inDim1, inDim1)  = _X[n];
	    (tmp(outDim*inDim1, inDim1) = _X[n]) *= (-y[j]/y[outDim]);
	}
	J(n*outDim, 0, outDim, J.ncol()) /= y[outDim];
    }

    return J;
}
    
void
ProjectiveMapping::Cost::update(AT& T, const Vector<ET>& dt) const
{
    Vector<ET>	t(T);
    ET		l = t.length();
    (t -= dt).normalize() *= l;
}
    
/************************************************************************
*  class AffineMapping							*
************************************************************************/
//! このアフィン変換の並行移動部分を表現するベクトルを返す．
/*! 
  \return	#outDim()次元ベクトル
*/
Vector<double>
AffineMapping::b() const
{
    Vector<double>	bb(outDim());
    for (int j = 0; j < bb.dim(); ++j)
	bb[j] = _T[j][inDim()];

    return bb;
}
    
}

