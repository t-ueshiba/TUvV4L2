/*
 *  平成9年 電子技術総合研究所 植芝俊夫 著作権所有
 *
 *  著作者による許可なしにこのプログラムの第三者への開示、複製、改変、
 *  使用等その他の著作人格権を侵害する行為を禁止します。
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *
 *  Copyright 1996
 *  Toshio UESHIBA, Electrotechnical Laboratory
 *
 *  All rights reserved.
 *  Any changing, copying or giving information about source programs of
 *  any part of this software and/or documentation without permission of the
 *  authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damage in use of this program.
 */

/*
 *  $Id: Image++.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include "TU/Image++.h"
#ifdef WIN32
#  include <winsock2.h>
#else
#  include <netinet/in.h>
#endif

namespace TU
{
/************************************************************************
*  class ImageLine<T>							*
************************************************************************/
template <class T> template <class S> const S*
ImageLine<T>::fill(const S* src)
{
    T* dst = *this;
    for (int n = dim() + 1; --n; )
	*dst++ = T(*src++);
    return src;
}

template <class T> const YUV422*
ImageLine<T>::fill(const YUV422* src)
{
    register T* dst = *this;
    for (register u_int u = 0; u < dim(); u += 2)
    {
	*dst++ = fromYUV<T>(src[0].y, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[1].y, src[0].x, src[1].x);
	src += 2;
    }
    return src;
}

template <class T> const YUV411*
ImageLine<T>::fill(const YUV411* src)
{
    register T*  dst = *this;
    for (register u_int u = 0; u < dim(); u += 4)
    {
	*dst++ = fromYUV<T>(src[0].y0, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[0].y1, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[1].y0, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[1].y1, src[0].x, src[1].x);
	src += 2;
    }
    return src;
}

template <class S> const S*
ImageLine<YUV422>::fill(const S* src)
{
    YUV422* dst = *this;
    for (int n = dim() + 1; --n; )
	*dst++ = YUV422(*src++);
    return src;
}

template <class S> const S*
ImageLine<YUV411>::fill(const S* src)
{
    YUV411* dst = *this;
    for (int n = dim() + 1; --n; )
	*dst++ = YUV411(*src++);
    return src;
}

/************************************************************************
*  class Image<T>							*
************************************************************************/
template <class T> std::istream&
Image<T>::restore(std::istream& in)
{
    switch (restoreHeader(in))
    {
      case U_CHAR:
	return restoreRows<u_char>(in);
      case SHORT:
	return restoreRows<short>(in);
      case FLOAT:
	return restoreRows<float>(in);
      case DOUBLE:
	return restoreRows<double>(in);
      case RGB_24:
	return restoreRows<RGB>(in);
      case YUV_444:
	return restoreRows<YUV444>(in);
      case YUV_422:
	return restoreRows<YUV422>(in);
      case YUV_411:
	return restoreRows<YUV411>(in);
    }
    return in;
}

template <class T> std::ostream&
Image<T>::save(std::ostream& out, Type type) const
{
    saveHeader(out, type);
    switch (type)
    {
      case U_CHAR:
	return saveRows<u_char>(out);
      case SHORT:
	return saveRows<short>(out);
      case FLOAT:
	return saveRows<float>(out);
      case DOUBLE:
	return saveRows<double>(out);
      case RGB_24:
	return saveRows<RGB>(out);
      case YUV_444:
	return saveRows<YUV444>(out);
      case YUV_422:
	return saveRows<YUV422>(out);
      case YUV_411:
	return saveRows<YUV411>(out);
    }
    return out;
}

template <class T> template <class S> std::istream&
Image<T>::restoreRows(std::istream& in)
{
    ImageLine<S>	buf(width());
    for (int v = 0; v < height(); )
    {
	if (!buf.restore(in))
	    break;
	(*this)[v++].fill((S*)buf);
    }
    return in;
}

template <class T> template <class D> std::ostream&
Image<T>::saveRows(std::ostream& out) const
{
    ImageLine<D>	buf(width());
    for (int v = 0; v < height(); )
    {
	buf.fill((T*)(*this)[v++]);
	if (!buf.save(out))
	    break;
    }
    return out;
}

template <class T> u_int
Image<T>::_width() const
{
    return width();
}

template <class T> u_int
Image<T>::_height() const
{
    return height();
}

template <class T> void
Image<T>::resize(u_int h, u_int w)
{
    Array2<ImageLine<T> >::resize(h, w);
}

template <class T> void
Image<T>::resize(T* p, u_int h, u_int w)
{
    Array2<ImageLine<T> >::resize(p, h, w);
}
 
}
