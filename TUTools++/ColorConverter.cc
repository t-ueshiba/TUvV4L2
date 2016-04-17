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
 *  $Id: ColorConverter.cc 1202 2012-09-25 00:03:13Z ueshiba $
 */
#include "TU/Image++.h"

namespace TU
{
namespace detail
{
/************************************************************************
*  global variables							*
************************************************************************/
const ColorConverter	colorConverter;

/************************************************************************
*  class detail::ColorConverter						*
************************************************************************/
ColorConverter::ColorConverter()
{
  // RGB -> YUV変換テーブルを作成
    for (int i = -255; i <= 255; ++i)
    {
	_u[255 + i] = limit(_ku * i + 128);
	_v[255 + i] = limit(_kv * i + 128);
    }
		    
  // YUV -> RGB変換テーブルを作成
    for (int i = 0; i < 256; ++i)
    {
	constexpr auto	au = _yb / (_ku * _yg);
	constexpr auto	av = _yr / (_kv * _yg);
	
	_r [i] = int((i - 128) / _kv);
	_gu[i] = scaleUp(au * (i - 128));
	_gv[i] = scaleUp(av * (i - 128));
	_b [i] = int((i - 128) / _ku);
    }
}

}	// namespace detail
}	// namespace TU
