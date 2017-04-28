/*
 *  平成21-22年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2009-2010.
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
 *  $Id: Feature.h,v 1.6 2010-11-22 06:16:22 ueshiba Exp $
 */
/*!
  \file		Feature.h
  \brief	クラス#TU::Featureの定義と実装
*/
#ifndef __TU_FEATURE_H
#define __TU_FEATURE_H

#include "TU/Geometry++.h"
#include "TU/Manip.h"

namespace TU
{
/************************************************************************
*  class Feature<T, D>							*
************************************************************************/
//! 特徴を表すクラス
template <class T, size_t D>
class Feature : public Point2f
{
  public:
    using value_type	= T;
    using Descriptor	= Vector<T, D>;	//!< 特徴ベクトルの型

    constexpr static size_t	DescriptorDim = D;

  public:
    Feature(float x=0, float y=0,
	    float sig=0, float ang=0, float scr=0, int tr=0)
	:Point2f({x, y}), sigma(sig), angle(ang), score(scr), trace(tr)	{}
    
    size_t		descriptorDim()				const	;
    float		diffSigma(const Feature& feature)	const	;
    float		diffAngle(const Feature& feature)	const	;
    T			sqdistOfFeature(const Feature& feature)	const	;
    T			sqdistOfFeature(const Feature& feature,
					T	       ceil)	const	;
    std::istream&	read(std::istream& in)				;
    std::ostream&	write(std::ostream& out)		const	;

  public:
    float		sigma;		//!< スケール値
    float		angle;		//!< 角度
    float		score;		//!< スコア
    int			trace;		//!< Hessianのトレースの符号
    Descriptor		descriptor;	//!< 記述子
};

//! 特徴の記述子の次元を返す．
/*!
  \return	特徴の記述子の次元
*/
template <class T, size_t D> inline size_t
Feature<T, D>::descriptorDim() const
{
    return descriptor.size();
}

//! 他の特徴とのスケール値の差を返す．
/*!
  \param feature	比較対象となる特徴
  \return		スケール値の差の絶対値
*/
template <class T, size_t D> inline float
Feature<T, D>::diffSigma(const Feature& feature) const
{
    return std::abs(sigma - feature.sigma);
}
    
//! 他の特徴との角度の差を返す．
/*!
  \param feature	比較対象となる特徴
  \return		角度の差の絶対値
*/
template <class T, size_t D> inline float
Feature<T, D>::diffAngle(const Feature& feature) const
{
    float	diff = angle - feature.angle;
    if (diff > M_PI)
	diff -= 2*M_PI;
    else if (diff < -M_PI)
	diff += 2*M_PI;
    return std::abs(diff);
}
    
//! 他の特徴との特徴ベクトルの二乗距離を返す．
/*!
  \param feature	比較対象となる特徴
  \return		特徴ベクトル間の二乗距離
*/
template <class T, size_t D> inline T
Feature<T, D>::sqdistOfFeature(const Feature& feature) const
{
    T	val = 0;
    for (size_t i = 0; i < descriptor.size(); ++i)
    {
	const T	diff = descriptor[i] - feature.descriptor[i];
	val += diff * diff;
    }
    return val;
}

//! 他の特徴との特徴ベクトルの二乗距離を返す．
/*!
  \param feature	比較対象となる特徴
  \return		特徴ベクトル間の二乗距離
*/
template <class T, size_t D> inline T
Feature<T, D>::sqdistOfFeature(const Feature& feature, T ceil) const
{
    T	val = 0;
    for (size_t i = 0; i < descriptor.size(); ++i)
    {
	const T	diff = descriptor[i] - feature.descriptor[i];
	val += diff * diff;
	if (val >= ceil)
	    break;
    }
    return val;
}

//! 入力ストリームから特徴を読み込む(Binary)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, size_t D> inline std::istream&
Feature<T, D>::read(std::istream& in)
{
    T	tmp[4];
    if (!in.read((char*)tmp, sizeof(tmp)))
	return in;
    (*this)[0] = tmp[0];
    (*this)[1] = tmp[1];
    sigma      = tmp[2];
    angle      = tmp[3];
    
    for (size_t i = 0; i < descriptor.size(); ++i)
    {
	T	tmp;
	if (!in.read((char*)&tmp, sizeof(tmp)))
	    break;
	descriptor[i] = tmp;
    }

    return in;
}
    
template <> inline std::istream&
Feature<int, 128u>::read(std::istream& in)
{
    double	tmp[4];
    if (!in.read((char*)tmp, sizeof(tmp)))
	return in;
    (*this)[0] = tmp[0];
    (*this)[1] = tmp[1];
    sigma      = tmp[2];
    angle      = tmp[3];
    
    for (size_t i = 0; i < descriptor.size(); ++i)
    {
	u_char	tmp;
	if (!in.read((char*)&tmp, sizeof(tmp)))
	    break;
	descriptor[i] = tmp;
    }

    return in;
}
    
//! 出力ストリームに特徴を書き出す(Binary)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, size_t D> inline std::ostream&
Feature<T, D>::write(std::ostream& out) const
{
    T	tmp[4];
    tmp[0] = (*this)[0];
    tmp[1] = (*this)[1];
    tmp[2] = sigma;
    tmp[3] = angle;
    if (!out.write((const char*)tmp, sizeof(tmp)))
	return out;
    
    for (size_t i = 0; i < descriptor.size(); ++i)
    {
	T	tmp = descriptor[i];
	if (!out.write((const char*)&tmp, sizeof(tmp)))
	    break;
    }
    
    return out;
}
    
template <> inline std::ostream&
Feature<int, 128u>::write(std::ostream& out) const
{
    double	tmp[4];
    tmp[0] = (*this)[0];
    tmp[1] = (*this)[1];
    tmp[2] = sigma;
    tmp[3] = angle;
    if (!out.write((const char*)tmp, sizeof(tmp)))
	return out;
    
    for (size_t i = 0; i < descriptor.size(); ++i)
    {
	u_char	tmp = descriptor[i];
	if (!out.write((const char*)&tmp, sizeof(tmp)))
	    break;
    }
    
    return out;
}
    
//! 入力ストリームから特徴を読み込む(ASCII)．
/*!
  \param in		入力ストリーム
  \param feature	特徴の読み込み先
  \return		inで指定した入力ストリーム
*/
template <class T, size_t D> inline std::istream&
operator >>(std::istream& in, Feature<T, D>& feature)
{
    return feature.get(in) >> feature.sigma >> feature.angle
			   >> feature.descriptor >> skipl;
}

//! 出力ストリームに特徴を書き出す(ASCII)．
/*!
  \param out		出力ストリーム
  \param feature	特徴
  \return		outで指定した出力ストリーム
*/
template <class T, size_t D> inline std::ostream&
operator <<(std::ostream& out, const Feature<T, D>& feature)
{
    return feature.put(out) << ' ' << feature.sigma << ' ' << feature.angle
			    << feature.descriptor;
}

//! 2つの特徴のスコアを比較する.
/*!
  \param feature0	特徴
  \param feature1	特徴
  \return		feature0のスコアがfeature1のスコアよりも小さければ
			true, そうでなければfalse
*/
template <class T, size_t D> inline bool
operator <(const Feature<T, D>& feature0, const Feature<T, D>& feature1)
{
    return feature0.score < feature1.score;
}

//! 2つの特徴のスコアを比較する.
/*!
  \param feature0	特徴
  \param feature1	特徴
  \return		feature0のスコアがfeature1のスコアよりも大きければ
			true, そうでなければfalse
*/
template <class T, size_t D> inline bool
operator >(const Feature<T, D>& feature0, const Feature<T, D>& feature1)
{
    return feature0.score > feature1.score;
}

/************************************************************************
*  type definitions							*
************************************************************************/
typedef Feature<int, 128u>	SIFT;
typedef Feature<float, 64u>	SURF;
typedef Feature<float, 128u>	SURF128;

}
#endif	// !__TU_FEATURE_H
