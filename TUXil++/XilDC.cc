/*
 *  $Id: XilDC.cc,v 1.2 2002-07-25 02:38:08 ueshiba Exp $
 */
#include "TU/v/XilDC.h"
#include <X11/Xmu/Converters.h>

// Solaris2.5.1 の XIL は、xil_copy_with_planemask() が正しく動かない
// （dst = (dst & ~mask) | (src & mask) となるべきなのに、(dst & ~
// mask)がセットされないで既存の値が消えてしまう。ただし, Create3D で 
// localに走らせたときは大丈夫。）ので、overlay を使っていても 
// XilImage をwindow に表示すると overlay の内容が消去されてしまう。こ
// れは、ユーザ側ではどうしようもない。そこで、とりあえずは、planemask 
// をサポートしないことにした。-DTUXilPP_UsePlaneMasksを指定してコンパ
// イルすれば、これがサポートされる。

namespace TU
{
namespace v
{
/************************************************************************
*  static functions							*
************************************************************************/
static const int	MonoShift  =  0;
static const int	RedShift   =  0;
static const int	GreenShift =  8;
static const int	BlueShift  = 16;

static int
windowNBands(const Colormap& colormap)
{
    return (colormap.vinfo().depth == 8 ? 1 : 3);
}

static Array<u_char>
monoPixels(const Colormap& colormap, int shift)
{
    Array<u_char>	pixels(256);
    for (int i = 0; i < 256; ++i)
	pixels[i] = (colormap.getUnderlayPixel(u_char(i), 0, 0)>>shift) & 0xff;
    
    return pixels;
}

static Array<u_char>
grayPixels(const Colormap& colormap)
{
    if (windowNBands(colormap) == 1)
	return monoPixels(colormap, MonoShift);
    else
    {
	Array<u_char>	pixels(256*3);
	for (int i = 0; i < 256; ++i)
	{
	    Pixel	pixel = colormap.getUnderlayPixel(u_char(i), 0, 0);
	    pixels[3*i  ] = (pixel >> BlueShift)  & 0xff;
	    pixels[3*i+1] = (pixel >> GreenShift) & 0xff;
	    pixels[3*i+2] = (pixel >> RedShift)   & 0xff;
	}
	return pixels;
    }
}

static Array<u_char>
signedPixels(const Colormap& colormap)
{
    if (windowNBands(colormap) == 1)
    {
	Array<u_char>	pixels(256);
	for (int i = 0; i < 256; ++i)
	    pixels[i] = colormap.getUnderlayPixel(s_char(i < 128 ? i : i-256),
						  0, 0) & 0xff;
	return pixels;
    }
    else
    {
	Array<u_char>	pixels(256*3);
	for (int i = 0; i < 256; ++i)
	{
	    Pixel pixel
		    = colormap.getUnderlayPixel(s_char(i < 128 ? i : i-256),
						    0, 0);
	    pixels[3*i  ] = (pixel >> BlueShift)  & 0xff;
	    pixels[3*i+1] = (pixel >> GreenShift) & 0xff;
	    pixels[3*i+2] = (pixel >> RedShift)	  & 0xff;
	}
	return pixels;
    }
}

static Array<int>
colorcubeStride(const Colormap& colormap)
{
    Array<int>	stride(3);
    stride[0] = colormap.bStride();
    stride[1] = colormap.gStride();
    stride[2] = colormap.rStride();
    
    return stride;
}

static Array<u_int>
colorcubeDim(const Colormap& colormap)
{
    Array<u_int>	dim(3);
    dim[0] = colormap.bDim();
    dim[1] = colormap.gDim();
    dim[2] = colormap.rDim();
    
    return dim;
}

/************************************************************************
*  class XilDC							*
************************************************************************/
/*
 *  Public member functions
 */
XilDC::XilDC(CanvasPane& parentCanvasPane, u_int w, u_int h)
    :CanvasPaneDC(parentCanvasPane, w, h),
     XilObject(),
   // このコンストラクタが呼ばれる時点ではまだ widget が realize されて
   // いないので、window から _xilDisplay を作ることはできない。つまり、
   // initializeGraphics() でやらなければならない。
     _xilDisplay(0),
     _xilTmpImage(xil_create(xilstate(), width(), height(),
			     windowNBands(colormap()), XIL_BYTE)),
#ifdef TUXilPP_UsePlaneMasks
     _xilTmpImage2(xil_create(xilstate(), width(), height(),
			      windowNBands(colormap()), XIL_BYTE)),
#endif
     _grayLookup(xil_lookup_create(xilstate(), XIL_BYTE, XIL_BYTE,
				   windowNBands(colormap()),
				   256, 0, grayPixels(colormap()))),
     _colorLookup(0),
     _ditherMask(xil_dithermask_get_by_name(xilstate(), "dm443")),
     _useSignedmap(0)
{
    if (windowNBands(colormap()) == 3)		// full color window ?
    {
	_lookups[0]  = xil_lookup_create(xilstate(), XIL_BYTE, XIL_BYTE, 1,
					 256, 0,
					 monoPixels(colormap(), BlueShift));
	_lookups[1]  = xil_lookup_create(xilstate(), XIL_BYTE, XIL_BYTE, 1,
					 256, 0,
					 monoPixels(colormap(), GreenShift));
	_lookups[2]  = xil_lookup_create(xilstate(), XIL_BYTE, XIL_BYTE, 1,
					 256, 0,
					 monoPixels(colormap(), RedShift));
	_colorLookup = xil_lookup_create_combined(xilstate(), _lookups, 3);
    }
    else					// pseudo color window.
    {
	_lookups[0]  = _lookups[1] = _lookups[2] = 0;
	_colorLookup = xil_colorcube_create(xilstate(), XIL_BYTE, XIL_BYTE, 3, 
					    colormap().getUnderlayPixel(
						u_char(0), 0, 0),
					    colorcubeStride(colormap()),
					    colorcubeDim(colormap()));
    }
					
  // xil は viewport 中の window のうち実際に表示されている部分にしか
  // 描画しないので、今まで隠れていた部分が scroll によって現れたときに
  // 正しく描画されるためには、backingstore を off にしなければならない。
    XtVaSetValues(widget(), XtNbackingStore, NotUseful, NULL);
}

XilDC::~XilDC()
{
    xil_lookup_destroy(_colorLookup);
    if (_lookups[2] != 0)
	xil_lookup_destroy(_lookups[2]);
    if (_lookups[1] != 0)
	xil_lookup_destroy(_lookups[1]);
    if (_lookups[0] != 0)
	xil_lookup_destroy(_lookups[0]);
    xil_lookup_destroy(_grayLookup);
#ifdef TUXilPP_UsePlaneMasks
    xil_destroy(_xilTmpImage2);
#endif
    xil_destroy(_xilTmpImage);
}

DC&
XilDC::setSize(u_int w, u_int h, u_int mul, u_int div)
{
    CanvasPaneDC::setSize(w, h, mul, div);
    _xilDisplay  = xil_create_from_window(xilstate(),
					  colormap().display(), drawable());
    xil_set_origin(_xilDisplay, log2devU(0), log2devV(0));
    return *this;
}

DC&
XilDC::setOffset(int u0, int v0)
{
    CanvasPaneDC::setOffset(u0, v0);
    xil_set_origin(_xilDisplay, log2devU(0), log2devV(0));
    return *this;
}

#ifdef TUXilPP_UsePlaneMasks
DC&
XilDC::setLayer(Layer layer)
{
    CanvasPaneDC::setLayer(layer);
    if (getLayer() == DC::UNDERLAY)
    {
	if (windowNBands(colormap()) == 3)
	{
	    _planeMasks[0] = (colormap().getUnderlayPlanes() >> BlueShift)
			   & 0xff;
	    _planeMasks[1] = (colormap().getUnderlayPlanes() >> GreenShift)
			   & 0xff;
	    _planeMasks[2] = (colormap().getUnderlayPlanes() >> RedShift)
			   & 0xff;
	}
	else
	{
	    _planeMasks[0] = colormap().underlayPlanes() & 0xff;
	    _planeMasks[1] = _planeMasks[2] = 0;
	}
    }
    else
    {
	if (windowNBands(colormap()) == 3)
	{
	    _planeMasks[0] = (colormap().getOverlayPlanes() >> BlueShift)
			   & 0xff;
	    _planeMasks[1] = (colormap().getOverlayPlanes() >> GreenShift)
			   & 0xff;
	    _planeMasks[2] = (colormap().getOverlayPlanes() >> RedShift)
			   & 0xff;
	}
	else
	{
	    _planeMasks[0] = colormap().getOverlayPlanes() & 0xff;
	    _planeMasks[1] = _planeMasks[2] = 0;
	}
    }
#ifdef DEBUG
    cerr << "_planeMasks: "
	 << hex << _planeMasks[0] << ", "
	 << hex << _planeMasks[1] << ", "
	 << hex << _planeMasks[2] << endl;
#endif
    return *this;
}
#endif	// TUXilPP_UsePlaneMasks

DC&
XilDC::setSaturation(u_int saturation)
{
    CanvasPaneDC::setSaturation(saturation);
    xil_lookup_destroy(_grayLookup);
    _grayLookup = xil_lookup_create(xilstate(), XIL_BYTE, XIL_BYTE,
				    windowNBands(colormap()),
				    256, 0,
				    (_useSignedmap ? signedPixels(colormap())
						   : grayPixels(colormap())));
    return *this;
}

DC&
XilDC::operator <<(const Point2<int>& p)
{
    CanvasPaneDC::operator <<(p);
    return *this;
}

DC&
XilDC::operator <<(const LineP2<double>& l)
{
    CanvasPaneDC::operator <<(l);
    return *this;
}

DC&
XilDC::operator <<(const Image<u_char>& image)
{
    CanvasPaneDC::operator <<(image);
    return *this;
}

DC&
XilDC::operator <<(const Image<s_char>& image)
{
    CanvasPaneDC::operator <<(image);
    return *this;
}

DC&
XilDC::operator <<(const Image<short>& image)
{
    CanvasPaneDC::operator <<(image);
    return *this;
}

DC&
XilDC::operator <<(const Image<BGR>& image)
{
    CanvasPaneDC::operator <<(image);
    return *this;
}

DC&
XilDC::operator <<(const Image<ABGR>& image)
{
    CanvasPaneDC::operator <<(image);
    return *this;
}

XilDC&
XilDC::operator <<(::XilImage image)
{
    ::XilImage	tmp = image;

  // Create temporary 3-band image from input 4-band image.
    if (xil_get_nbands(image) == 4)
	tmp = xil_create_child(image, 0, 0,
			       xil_get_width(image), xil_get_height(image),
			       1, 3);

  // Do main job.
    if (xil_get_nbands(tmp) == 1)			// monochrome image ?
	if (_useSignedmap)
	    setSignedmap();
	else
	    setGraymap();
    else
	setColorcube();

#ifdef TUXilPP_UsePlaneMasks
#  define OUTPUT	_xilTmpImage2
#else
#  define OUTPUT	_xilDisplay
#endif
    if (!hasScale())
    {
#ifdef TUXilPP_UsePlaneMasks
	if (xil_get_width(_xilTmpImage2)  != xil_get_width(tmp) ||
	    xil_get_height(_xilTmpImage2) != xil_get_width(tmp))
	{
	    xil_destroy(_xilTmpImage2);
	    _xilTmpImage2 = xil_create(xilstate(),
				       xil_get_width(tmp), xil_get_height(tmp),
				       xil_get_nbands(_xilDisplay), XIL_BYTE);
	}
#endif
	if (xil_get_nbands(tmp) == 1)			// monochrome image ?
	    xil_lookup(tmp, OUTPUT, _grayLookup);
	else if (xil_get_nbands(_xilDisplay) == 3)	// fullcolor window ?
	    xil_lookup(tmp, OUTPUT, _colorLookup);
	else						// pseudo color window.
	    xil_ordered_dither(tmp, OUTPUT, _colorLookup, _ditherMask);
#ifdef TUXilPP_UsePlaneMasks
	xil_copy_with_planemask(OUTPUT, _xilDisplay, _planeMasks);
#endif
    }
    else
    {
      // Do looking-up first and then scale the image.
	if (xil_get_width(_xilTmpImage)  != xil_get_width(tmp) ||
	    xil_get_height(_xilTmpImage) != xil_get_height(tmp))
	{
	    xil_destroy(_xilTmpImage);
	    _xilTmpImage = xil_create(xilstate(),
				      xil_get_width(tmp), xil_get_height(tmp),
				      xil_get_nbands(_xilDisplay), XIL_BYTE);
	}
	if (xil_get_nbands(tmp) == 1)			// monochrome image ?
	    xil_lookup(tmp, _xilTmpImage, _grayLookup);
	else if (xil_get_nbands(_xilDisplay) == 3)	// fullcolor window ?
	    xil_lookup(tmp, _xilTmpImage, _colorLookup);
	else						// pseudo color window.
	    xil_ordered_dither(tmp, _xilTmpImage, _colorLookup, _ditherMask);
#ifdef TUXilPP_UsePlaneMasks
	if (xil_get_width(_xilTmpImage2) !=
	    log2devR(xil_get_width(_xilTmpImage)) ||
	    xil_get_height(_xilTmpImage2) !=
	    log2devR(xil_get_width(_xilTmpImage)))
	{
	    xil_destroy(_xilTmpImage2);
	    _xilTmpImage2 = xil_create(xilstate(),
				       log2devR(xil_get_width(_xilTmpImage)),
				       log2devR(xil_get_height(_xilTmpImage)),
				       xil_get_nbands(_xilDisplay), XIL_BYTE);
	}
#endif
	xil_scale(_xilTmpImage, OUTPUT, "nearest",
		  (float)mul()/(float)div(), (float)mul()/(float)div());
#ifdef TUXilPP_UsePlaneMasks
	xil_copy_with_planemask(OUTPUT, _xilDisplay, _planeMasks);
#endif
    }

  // Destroy temorarily created 3-band image.
    if (xil_get_nbands(image) == 4)
	xil_destroy(tmp);

    return *this;
}

/*
 *  Protected member functions
 */
void
XilDC::initializeGraphics()
{
    _xilDisplay = xil_create_from_window(xilstate(),
					 colormap().display(), drawable());
    xil_set_origin(_xilDisplay, log2devU(0), log2devV(0));
}

XDC&
XilDC::setGraymap()
{
    CanvasPaneDC::setGraymap();
    xil_lookup_destroy(_grayLookup);
    _grayLookup = xil_lookup_create(xilstate(), XIL_BYTE, XIL_BYTE,
				    windowNBands(colormap()),
				    256, 0, grayPixels(colormap()));
    return *this;
}

XDC&
XilDC::setSignedmap()
{
    CanvasPaneDC::setSignedmap();
    xil_lookup_destroy(_grayLookup);
    _grayLookup = xil_lookup_create(xilstate(), XIL_BYTE, XIL_BYTE,
				    windowNBands(colormap()),
				    256, 0, signedPixels(colormap()));
    return *this;
}
 
}
}
