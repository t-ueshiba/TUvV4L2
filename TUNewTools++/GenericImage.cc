/*
 *  $Id: GenericImage.cc 1945 2016-02-21 03:44:08Z ueshiba $
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
	    if (!in.read(std::begin(*line), std::size(*line)) ||
		!in.ignore(npads))
		break;
    }
    else
    {
	for (auto line : _a)
	    if (!in.read(std::begin(line), std::size(line)) ||
		!in.ignore(npads))
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
	    if (!out.write(std::begin(*line), std::size(*line)) ||
		!pad.save(out))
		break;
    }
    else
    {
	for (auto line : _a)
	    if (!out.write(std::begin(line), std::size(line)) ||
		!pad.save(out))
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
