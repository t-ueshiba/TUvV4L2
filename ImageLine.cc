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
*  class ImageLine<YUV422>						*
************************************************************************/
const YUV444*
ImageLine<YUV422>::fill(const YUV444* src)
{
    register YUV422* dst = *this;
    for (register int n = dim(); n > 0; n -= 2)
    {
	dst->x = src->u;
	dst->y = src->y;
	++dst;
	dst->x = src->v;
	++src;
	dst->y = src->y;
	++dst;
	++src;
    }
    return src;
}

const YUV411*
ImageLine<YUV422>::fill(const YUV411* src)
{
    register YUV422* dst = *this;
    for (register int n = dim(); n > 0; n -= 4)
    {
	dst->x = src[0].x;
	dst->y = src[0].y0;
	++dst;
	dst->x = src[1].x;
	dst->y = src[0].y1;
	++dst;
	dst->x = src[0].x;
	dst->y = src[1].y0;
	++dst;
	dst->x = src[1].x;
	dst->y = src[1].y1;
	++dst;
	src += 2;
    }
    return src;
}

/************************************************************************
*  class ImageLine<YUYV422>						*
************************************************************************/
const YUV444*
ImageLine<YUYV422>::fill(const YUV444* src)
{
    register YUYV422* dst = *this;
    for (register int n = dim(); n > 0; n -= 2)
    {
	dst->y = src->y;
	dst->x = src->u;
	++dst;
	dst->x = src->v;
	++src;
	dst->y = src->y;
	++dst;
	++src;
    }
    return src;
}

const YUV411*
ImageLine<YUYV422>::fill(const YUV411* src)
{
    register YUYV422* dst = *this;
    for (register int n = dim(); n > 0; n -= 4)
    {
	dst->y = src[0].y0;
	dst->x = src[0].x;
	++dst;
	dst->y = src[0].y1;
	dst->x = src[1].x;
	++dst;
	dst->y = src[1].y0;
	dst->x = src[0].x;
	++dst;
	dst->y = src[1].y1;
	dst->x = src[1].x;
	++dst;
	src += 2;
    }
    return src;
}

/************************************************************************
*  class ImageLine<YUV411>						*
************************************************************************/
const YUV444*
ImageLine<YUV411>::fill(const YUV444* src)
{
    register YUV411* dst = *this;
    for (register int n = dim(); n > 0; n -= 2)
    {
	dst->x  = src[0].u;
	dst->y0 = src[0].y;
	dst->y1 = src[1].y;
	++dst;
	dst->x  = src[0].v;
	dst->y0 = src[2].y;
	dst->y1 = src[3].y;
	++dst;
	src += 4;
    }
    return src;
}

const YUV422*
ImageLine<YUV411>::fill(const YUV422* src)
{
    register YUV411* dst = *this;
    for (register int n = dim(); n > 0; n -= 2)
    {
	dst->x  = src[0].x;
	dst->y0 = src[0].y;
	dst->y1 = src[1].y;
	++dst;
	dst->x  = src[1].x;
	dst->y0 = src[2].y;
	dst->y1 = src[3].y;
	++dst;
	src += 4;
    }
    return src;
}
 
const YUYV422*
ImageLine<YUV411>::fill(const YUYV422* src)
{
    register YUV411* dst = *this;
    for (register int n = dim(); n > 0; n -= 2)
    {
	dst->x  = src[0].x;
	dst->y0 = src[0].y;
	dst->y1 = src[1].y;
	++dst;
	dst->x  = src[1].x;
	dst->y0 = src[2].y;
	dst->y1 = src[3].y;
	++dst;
	src += 4;
    }
    return src;
}
 
}
