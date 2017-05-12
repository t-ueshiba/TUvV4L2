/*!
  \file		ColorConverter.cc
  \author	Toshio UESHIBA
  \brief	クラス TU::detail::ColorConverter の実装
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
	_u[255 + i] = limit(round<int>(_ku * i) + 128);
	_v[255 + i] = limit(round<int>(_kv * i) + 128);
    }
		    
  // YUV -> RGB変換テーブルを作成
    for (int i = 0; i < 256; ++i)
    {
	constexpr auto	au = _yb / (_ku * _yg);
	constexpr auto	av = _yr / (_kv * _yg);
	
	_r [i] = round<int>((i - 128) / _kv);
	_gu[i] = scaleUp(au * (i - 128));
	_gv[i] = scaleUp(av * (i - 128));
	_b [i] = round<int>((i - 128) / _ku);
    }
}

}	// namespace detail
}	// namespace TU
