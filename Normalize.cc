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
 *  $Id: Normalize.cc,v 1.5 2008-09-10 05:10:42 ueshiba Exp $
 */
#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  class Normalize							*
************************************************************************/
//! 正規化変換行列を返す．
/*!
  \return	変換行列:
		\f$
		\TUvec{T}{} = 
		\TUbeginarray{ccc}
		 s^{-1} \TUvec{I}{d} & -s^{-1}\TUvec{c}{} \\ \TUtvec{0}{d} & 1
		\TUendarray
		\f$
*/
Matrix<double>
Normalize::T() const
{
    Matrix<double>	TT(spaceDim()+1, spaceDim()+1);
    for (int i = 0; i < spaceDim(); ++i)
    {
	TT[i][i] = 1.0 / _scale;
	TT[i][spaceDim()] = -_centroid[i] / _scale;
    }
    TT[spaceDim()][spaceDim()] = 1.0;

    return TT;
}

//! 正規化変換の転置行列を返す．
/*!
  \return	変換の転置行列:
		\f$
		\TUtvec{T}{} = 
		\TUbeginarray{ccc}
		 s^{-1} \TUvec{I}{d} & \TUvec{0}{d} \\ -s^{-1}\TUtvec{c}{} & 1
		\TUendarray
		\f$
*/
Matrix<double>
Normalize::Tt() const
{
    Matrix<double>	TTt(spaceDim()+1, spaceDim()+1);
    for (int i = 0; i < spaceDim(); ++i)
    {
	TTt[i][i] = 1.0 / _scale;
	TTt[spaceDim()][i] = -_centroid[i] / _scale;
    }
    TTt[spaceDim()][spaceDim()] = 1.0;

    return TTt;
}

//! 正規化変換の逆行列を返す．
/*!
  \return	変換の逆行列:
		\f$
		\TUinv{T}{} = 
		\TUbeginarray{ccc}
		 s \TUvec{I}{d} & \TUvec{c}{} \\ \TUtvec{0}{d} & 1
		\TUendarray
		\f$
*/
Matrix<double>
Normalize::Tinv() const
{
    Matrix<double>	TTinv(spaceDim()+1, spaceDim()+1);
    for (int i = 0; i < spaceDim(); ++i)
    {
	TTinv[i][i] = _scale;
	TTinv[i][spaceDim()] = _centroid[i];
    }
    TTinv[spaceDim()][spaceDim()] = 1.0;

    return TTinv;
}

//! 正規化変換の逆行列の転置を返す．
/*!
  \return	変換の逆行列の転置:
		\f$
		\TUtinv{T}{} = 
		\TUbeginarray{ccc}
		 s \TUvec{I}{d} & \TUvec{0}{d} \\ \TUtvec{c}{} & 1
		\TUendarray
		\f$
*/
Matrix<double>
Normalize::Ttinv() const
{
    Matrix<double>	TTtinv(spaceDim()+1, spaceDim()+1);
    for (int i = 0; i < spaceDim(); ++i)
    {
	TTtinv[i][i] = _scale;
	TTtinv[spaceDim()][i] = _centroid[i];
    }
    TTtinv[spaceDim()][spaceDim()] = 1.0;

    return TTtinv;
}
 
}
