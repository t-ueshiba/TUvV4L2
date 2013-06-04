/*
 *  $Id$
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
  - #TU::BlockDiagonalMatrix
  - #TU::SparseMatrix
  - #TU::BandMatrix

  <b>非線形最適化</b>
  - #TU::NullConstraint
  - #TU::ConstNormConstraint
  - #TU::minimizeSquare(const F&, const G&, AT&, u_int, double)
  - #TU::minimizeSquareSparse(const F&, const G&, ATA&, IB, IB, u_int, double)

  <b>RANSAC</b>
  - #TU::ransac(const PointSet&, Model&, Conform, double)

  <b>グラフカット</b>
  - #boost::GraphCuts

  <b>動的計画法</b>
  - #TU::DP

  <b>点，直線，平面等の幾何要素とその変換</b>
  - #TU::Point1
  - #TU::Point2
  - #TU::Point3
  - #TU::HyperPlane
  - #TU::Normalize
  - #TU::Projectivity
  - #TU::Affinity
  - #TU::Rigidity
  - #TU::Homography
  - #TU::Affinity2
  - #TU::BoundingBox
  
  <b>投影の幾何</b>
  - #TU::IntrinsicBase
  - #TU::IntrinsicWithFocalLength
  - #TU::IntrinsicWithEuclideanImagePlane
  - #TU::Intrinsic
  - #TU::IntrinsicWithDistortion
  - #TU::CanonicalCamera
  - #TU::Camera
  
  <b>画素と画像</b>
  - #TU::RGB
  - #TU::BGR
  - #TU::RGBA
  - #TU::ABGR
  - #TU::ARGB
  - #TU::BGRA
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
  - #TU::Filter2
  - #TU::SeparableFilter2
  - #TU::IIRFilter
  - #TU::BidirectionalIIRFilter
  - #TU::BidirectionalIIRFilter2
  - #TU::DericheConvolver
  - #TU::DericheConvolver2
  - #TU::GaussianConvolver
  - #TU::GaussianConvolver2
  - #TU::BoxFilter
  - #TU::BoxFilter2
  - #TU::GuidedFilter
  - #TU::GuidedFilter2

  <b>特殊データ構造</b>
  - #TU::List
  - #TU::Heap
  - #TU::PSTree
  - #TU::NDTree
  
  <b>Bezier曲線とBezier曲面</b>
  - #TU::BezierCurve
  - #TU::BezierSurface
  
  <b>B-spline曲線とB-spline曲面</b>
  - #TU::BSplineKnots
  - #TU::BSplineCurve
  - #TU::BSplineSurface
  
  <b>メッシュ</b>
  - #TU::Mesh

  <b>アルゴリズム</b>
  - #std::min(const T&, const T&, const T&)
  - #std::min(const T&, const T&, const T&, const T&)
  - #std::max(const T&, const T&, const T&)
  - #std::max(const T&, const T&, const T&, const T&)
  - #TU::pull_if(Iter, Iter, Pred)
  - #TU::diff(const T&, const T&)
  - #TU::op3x3(Iterator begin, Iterator end, OP op)
  - #TU::max3x3(P p, P q, P r)
  - #TU::min3x3(P p, P q, P r)
  - #TU::mopOpen(Iterator begin, Iterator end, u_int niter)
  - #TU::mopClose(Iterator begin, Iterator end, u_int niter)

  <b>関数オブジェクト</b>
  - #TU::unarize
  - #TU::seq_transform
  - #TU::mem_var_t
  - #TU::const_mem_var_t
  - #TU::mem_var_ref_t
  - #TU::const_mem_var_ref_t

  <b>反復子</b>
  - #TU::make_mbr_iterator(Iterator i, S T::* m)
  - #TU::make_const_mbr_iterator(Iterator i, S T::* m)
  - #TU::make_first_iterator(Iterator i)
  - #TU::make_const_first_iterator(Iterator i)
  - #TU::make_second_iterator(Iterator i)
  - #TU::make_const_second_iterator(Iterator i)
  - #TU::assignment_iterator
  - #TU::assignment2_iterator
  - #TU::row_iterator
  - #TU::vertical_iterator
  - #TU::box_filter_iterator
  - #TU::iir_filter_iterator
  - #TU::ring_iterator
  
  <b>マニピュレータ</b>
  - #TU::skipl(std::istream&)
  - #TU::IOManip
  - #TU::IManip1
  - #TU::OManip1
  - #TU::IOManip1
  - #TU::IManip2
  - #TU::OManip2

  <b>ストリーム
  - #TU::fdistream
  - #TU::fdostream
  - #TU::fdstream
  
  <b>シリアルインタフェース</b>
  - #TU::Serial
  - #TU::TriggerGenerator
  - #TU::PM16C_04
  - #TU::SHOT602

  <b>乱数発生器</b>
  - #TU::Random

  <b>SIMD命令</b>
  - #mm::vec
  
  \file		types.h
  \brief	基本的なデータ型をグローバルな名前空間に追加
*/
#ifndef __TUtypes_h
#define __TUtypes_h

#ifdef WIN32
#  ifdef _USRDLL
#    define __PORT	__declspec(dllexport)
#  else
#    define __PORT	__declspec(dllimport)
#  endif
#  define _USE_MATH_DEFINES	    // <math.h>のM_PI等の定義を有効化
#  define _CRT_SECURE_NO_WARNINGS   // C標準関数使用時の警告を抑制
#  define _CRT_NONSTDC_NO_DEPRECATE // POSIX関数使用時の警告を抑制
#  define _SCL_SECURE_NO_WARNINGS   // C++標準関数使用時の警告を抑制

typedef unsigned int		size_t;		//!< 配列等のサイズを表す型
typedef unsigned char		u_char;		//!< 符号なし8bit整数
typedef unsigned short		u_short;	//!< 符号なし16bit整数
typedef unsigned int		u_int;		//!< 符号なし32bit整数
typedef unsigned long		u_long;		//!< 符号なし32/64bit整数
typedef signed char		int8_t;		//!< 符号付き8bit整数
typedef short			int16_t;	//!< 符号付き16bit整数
typedef int			int32_t;	//!< 符号付き32bit整数
typedef long long		int64_t;	//!< 符号付き64bit整数
typedef unsigned char		u_int8_t;	//!< 符号なし8bit整数
typedef unsigned short		u_int16_t;	//!< 符号なし16bit整数
typedef unsigned int		u_int32_t;	//!< 符号なし32bit整数
typedef unsigned long long	u_int64_t;	//!< 符号なし64bit整数
#else
#  define __PORT
#  include <sys/types.h>
#endif

typedef signed char		s_char;		//!< 符号付き8bit整数

#endif	/*  !__TUtypes_h	*/
