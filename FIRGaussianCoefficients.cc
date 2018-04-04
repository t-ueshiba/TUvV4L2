/*!
  \file		FIRGaussianCoefficients.cc
  \author	Toshio UESHIBA
  \brief	クラス TU::FIRGaussianConvolver の実装
*/
#include "TU/FIRGaussianConvolver.h"
#include <cmath>

namespace TU
{
/************************************************************************
*  class FIRGaussianCoefficients<D, T>					*
************************************************************************/
//! Gauss核を初期化する
/*!
  \param sigma	Gauss核のスケール
  \return	このGauss核
*/
template <size_t D, class T> void
FIRGaussianCoefficients<D, T>::initialize(coeff_type sigma)
{
    using namespace	std;

  // 0/1/2階微分のための係数を計算する．
    for (size_t i = 0; i <= D; ++i)
    {
	coeff_type	dx = coeff_type(i) / sigma, dxdx = dx*dx;
	
	_c0[D+i] = _c0[D-i] = exp(-coeff_type(0.5) * dxdx);
	_c1[D+i] = dx * _c0[D+i];
	_c1[D-i] = -_c1[D+i];
	_c2[D+i] = _c2[D-i] = (dxdx - coeff_type(1)) * _c0[D+i];
    }

  // 0階微分用の係数を正規化する．
    coeff_type	sum = _c0[D];
    for (size_t i = 1; i <= D; ++i)
	sum += (2 * _c0[D+i]);
    for (size_t i = 0; i <= 2*D; ++i)
	_c0[i] /= abs(sum);

  // 1階微分用の係数を正規化して格納する．
    sum = 0;
    for (size_t i = 1; i <= D; ++i)
	sum += (2 * i * _c1[D+i]);
    for (size_t i = 0; i <= 2*D; ++i)
	_c1[i] /= abs(sum);

  // 2階微分用の係数を正規化して格納する．
    sum = 0;
    for (size_t i = 1; i <= D; ++i)
	sum += (i * i * _c2[D+i]);
    for (size_t i = 0; i <= 2*D; ++i)
	_c2[i] /= abs(sum);

#ifdef _DEBUG
    cerr << "c0: " << _c0;
    cerr << "c1: " << _c1;
    cerr << "c2: " << _c2;
#endif
}

template
void FIRGaussianCoefficients<16, float>::initialize(coeff_type sigma)	;
template
void FIRGaussianCoefficients<16, double>::initialize(coeff_type sigma)	;
}
