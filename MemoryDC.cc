/*
 *  $Id$  
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
    MemoryDC::MemoryDC(Colormap& cmap, size_t w, size_t h, float zoom)
    :XDC(w, h, zoom, cmap, XDefaultGC(cmap.display(), cmap.vinfo().screen)),
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
MemoryDC::setSize(size_t width, size_t height, float zoom)
{
    XDC::setSize(width, height, zoom);
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
