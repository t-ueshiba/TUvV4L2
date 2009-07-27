/*
 *  $Id: types.h,v 1.11 2009-07-27 07:32:05 ueshiba Exp $
 */
/*!
  \mainpage	libTUTools++ - 配列，ベクトル，行列，画像等の基本的なデータ型とそれに付随したアルゴリズムを収めたライブラリ
  \anchor	libTUTools

  \section copyright 著作権
  平成14-19年（独）産業技術総合研究所 著作権所有

  創作者：植芝俊夫

  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
  等の行為を禁止します．
   
  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
  よび創作者は責任を負いません。

  Copyright 2002-2007.
  National Institute of Advanced Industrial Science and Technology (AIST)

  Creator: Toshio UESHIBA

  [AIST Confidential and all rights reserved.]
  This program is confidential. Any using, copying, changing or
  giving any information concerning with this program to others
  without permission by the copyright holder are strictly prohibited.

  [No Warranty.]
  The copyright holder or the creator are not responsible for any
  damages caused by using this program.

  \section abstract 概要
  libTUTools++は，配列，ベクトル，行列，画像等の基本的なデータ型とそれ
  に付随したアルゴリズムを収めたライブラリである．現在実装されている主
  要なクラスおよび関数はおおまかに以下の分野に分類される．

  <b>1次元および2次元配列</b>
  - #TU::Array
  - #TU::Array2

  <b>ベクトルと行列および線形計算</b>
  - #TU::Vector
  - #TU::Matrix
  - #TU::LUDecomposition
  - #TU::Householder
  - #TU::QRDecomposition
  - #TU::TriDiagonal
  - #TU::BiDiagonal
  - #TU::SVDecomposition
  - #TU::BlockMatrix

  <b>非線形最適化</b>
  - #TU::NullConstraint
  - #TU::ConstNormConstraint
  - #Matrix<typename F::value_type> TU::minimizeSquare(const F&, const G&, AT&, int, double)
  - #Matrix<typename F::value_type> TU::minimizeSquareSparse(const F&, const G&, ATA&, IB, IB, int, double)

  <b>RANSAC</b>
  - #typename Pointset::Container TU::ransac(const PointSet&, Model&, Conform, double)

  <b>点，直線，平面等の幾何要素とその変換</b>
  - #TU::Point2
  - #TU::Point3
  - #TU::HyperPlane
  - #TU::Normalize
  - #TU::ProjectiveMapping
  - #TU::AffineMapping

  <b>投影の幾何</b>
  - #TU::CanonicalCamera
  - #TU::CameraWithFocalLength
  - #TU::CameraWithEuclideanImagePlane
  - #TU::Camera
  - #TU::CameraWithDistortion
  
  <b>画素と画像</b>
  - #TU::RGB
  - #TU::BGR
  - #TU::RGBA
  - #TU::ABGR
  - #TU::YUV444
  - #TU::YUV422
  - #TU::YUV411
  - #TU::ImageLine
  - #TU::Image
  - #TU::GenericImage
  - #TU::Movie

  <b>画像処理</b>
  - #TU::EdgeDetector
  - #TU::CorrectIntensity
  - #TU::Warp
  - #TU::IntegralImage
  - #TU::DiagonalIntegralImage
  
  <b>画像に限らない信号処理</b>
  - #TU::IIRFilter
  - #TU::BilateralIIRFilter
  - #TU::BilateralIIRFilter2
  - #TU::DericheConvolver
  - #TU::DericheConvolver2
  - #TU::GaussianConvolver
  - #TU::GaussianConvolver2
  
  <b>特殊データ構造</b>
  - #TU::List
  - #TU::Heap
  - #TU::PSTree

  <b>Bezier曲線とBezier曲面</b>
  - #TU::BezierCurve
  - #TU::BezierSurface
  
  <b>B-Spline曲線とB-Spline曲面</b>
  - #TU::BSplineCurve
  - #TU::BSplineSurface
  
  <b>メッシュ</b>
  - #TU::Mesh

  <b>標準ライブラリの補強</b>
  - #const T& std::min(const T&, const T&, const T&)
  - #const T& std::min(const T&, const T&, const T&, const T&)
  - #const T& std::max(const T&, const T&, const T&)
  - #const T& std::max(const T&, const T&, const T&, const T&)
  - #Iter TU::pull_if(Iter, Iter, Pred)
  - #T TU::diff(const T&, const T&)
  - #TU::mbr_iterator
  - #std::istream& TU::ign(std::istream&)
  - #std::istream& TU::skipl(std::istream&)
  - #TU::IOManip
  - #TU::IManip1
  - #TU::OManip1
  - #TU::IOManip1
  - #TU::IManip2
  - #TU::OManip2

  <b>メモリ管理</b>
  - #TU::Allocator

  <b>シリアルインタフェース</b>
  - #TU::Serial
  - #TU::Puma
  - #TU::Pata
  - #TU::Microscope
  - #TU::TriggerGenerator
  
  \file		types.h
  \brief	基本的なデータ型をグローバルな名前空間に追加
*/
#ifndef __TUtypes_h
#define __TUtypes_h

#ifdef WIN32
typedef unsigned int	size_t;			//!< 配列等のサイズを表す型
typedef unsigned char	u_char;			//!< 符号なし8bit整数
typedef unsigned short	u_short;		//!< 符号なし16bit整数
typedef unsigned int	u_int;			//!< 符号なし32bit整数
typedef unsigned long	u_long;			//!< 符号なし32/64bit整数
#else
#  include <sys/types.h>
#endif

typedef signed char		s_char;		//!< 符号付き8bit整数
typedef long long		int64;		//!< 符号付き64bit整数
typedef unsigned long long	u_int64;	//!< 符号なし64bit整数

#endif	/*  !__TUtypes_h	*/
