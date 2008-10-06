/*
 *  $Id: types.h,v 1.9 2008-10-06 01:41:25 ueshiba Exp $
 */
/*!
  \mainpage	libTUTools++ - 配列，ベクトル，行列，画像など基本的なデータ型とそれに付随したアルゴリズムを収めたライブラリ
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
  に付随したアルゴリズムを収めたライブラリである．

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
