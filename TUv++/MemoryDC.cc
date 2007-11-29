/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: MemoryDC.cc,v 1.3 2007-11-29 07:06:07 ueshiba Exp $
 */
#include "TU/v/MemoryDC.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MemoryDC						*
************************************************************************/
/*
 *  Public member functions
 */
MemoryDC::MemoryDC(Colormap& cmap, u_int w, u_int h)
    :XDC(w, h, cmap, XDefaultGC(cmap.display(), cmap.vinfo().screen)),
     _pixmap(XCreatePixmap(colormap().display(),
			   DefaultRootWindow(colormap().display()), w, h,
			   colormap().vinfo().depth))
{
}

MemoryDC::~MemoryDC()
{
    XFreePixmap(colormap().display(), _pixmap);
}

DC&
MemoryDC::setSize(u_int width, u_int height, u_int mul, u_int div)
{
    XDC::setSize(width, height, mul, div);
  // Viewport の中でこの widget を小さくするとき, 以前描画したものの残
  // 骸が余白に残るのは見苦しいので、widget 全体をクリアしておく。また、
  // 直接 graphic hardware にアクセスする API （XIL など）と実行順序が
  // 入れ替わることを防ぐため、XSync() を呼ぶ（XDC.cc 参照）。
    XClearWindow(colormap().display(), drawable());
    XSync(colormap().display(), False);
    return *this;
}

/*
 *  Protected member functions
 */
Drawable
MemoryDC::drawable() const
{
    return _pixmap;
}

void
MemoryDC::initializeGraphics()
{
}

DC&
MemoryDC::repaintUnderlay()
{
    return *this;
}

DC&
MemoryDC::repaintOverlay()
{
    return *this;
}

/*
 *  Private member functions
 */

}
}
