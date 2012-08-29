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
 *  $Id: Icon.cc,v 1.6 2012-08-29 21:17:18 ueshiba Exp $  
 */
#include "TU/v/Icon.h"
#include <stdexcept>

namespace TU
{
namespace v
{
/************************************************************************
*  class Icon								*
************************************************************************/
Icon::Icon(const Colormap& colormap, const u_char data[])
    :_display(colormap.display()),
     _pixmap(XCreatePixmap(_display, DefaultRootWindow(_display),
			   data[0], data[1], colormap.vinfo().depth))
{
    u_long	pixel[5];

    GC		gc = DefaultGC(_display, colormap.vinfo().screen);
    XGCValues	values;
    XGetGCValues(_display, gc, GCForeground, &values);	// keep foreground
    
    int		i = 2;
    for (int v = 0; v < data[1]; ++v)
	for (int u = 0; u < data[0]; ++u)
	{
	    if (data[i] >= sizeof(pixel) / sizeof(pixel[0]))
	    {
		throw std::runtime_error("TU::v::Icon::Icon(): invalid input icon data!");
	    }
	    XSetForeground(_display, gc, pixel[data[i++]]);
	    XDrawPoint(_display, _pixmap, gc, u, v);
	}
    XChangeGC(_display, gc, GCForeground, &values);	// restore foreground
}

Icon::~Icon()
{
    XFreePixmap(_display, _pixmap);
}

}
}
