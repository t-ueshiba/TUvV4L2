/*
 *  $Id: MemoryDC.cc,v 1.1 2004-09-27 08:32:00 ueshiba Exp $
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
