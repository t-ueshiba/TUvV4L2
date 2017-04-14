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
 *  $Id$
 */
#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class GenericImage							*
************************************************************************/
//! 入力ストリームから画像の画素データを読み込む．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
std::istream&
GenericImage::restoreData(std::istream& in)
{
    _colormap.resize(_typeInfo.ncolors);
    _colormap.restore(in);

    size_t	npads = type2nbytes(_typeInfo.type, true);
    if (_typeInfo.bottomToTop)
    {
	for (auto line = _a.rbegin(); line != _a.rend(); ++line)
	    if (!line->restore(in) || !in.ignore(npads))
		break;
    }
    else
    {
	for (auto line = _a.begin(); line != _a.end(); ++line)
	    if (!line->restore(in) || !in.ignore(npads))
		break;
    }

    return in;
}

//! 出力ストリームに画像の画素データを書き出す．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
std::ostream&
GenericImage::saveData(std::ostream& out) const
{
    if (_colormap.size() > 0)
    {
	_colormap.save(out);
	for (size_t i = _colormap.size(); i < 256; ++i)
	    out.put(0).put(0).put(0).put(0);
    }
    
    Array<u_char>	pad(type2nbytes(_typeInfo.type, true));
    if (_typeInfo.bottomToTop)
    {
	for (auto line = _a.rbegin(); line != _a.rend(); ++line)
	    if (!line->save(out) || !pad.save(out))
		break;
    }
    else
    {
	for (auto line = _a.begin(); line != _a.end(); ++line)
	    if (!line->save(out) || !pad.save(out))
		break;
    }
    
    return out;
}

size_t
GenericImage::_width() const
{
    return (_a.ncol()*8) / type2depth(_typeInfo.type);
}

size_t
GenericImage::_height() const
{
    return _a.nrow();
}

ImageBase::Type
GenericImage::_defaultType() const
{
    return _typeInfo.type;
}

void
GenericImage::_resize(size_t h, size_t w, const TypeInfo& typeInfo)
{
    _typeInfo = typeInfo;
    w = (type2depth(_typeInfo.type)*w + 7) / 8;
    _a.resize(h, w);
}

}
