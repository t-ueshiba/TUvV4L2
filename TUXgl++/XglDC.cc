/*
 *  $Id: XglDC.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include "TU/v/XglDC.h"
#include <stdexcept>
#include <X11/Xmu/Converters.h>

namespace TU
{
namespace v
{
/************************************************************************
*  class XglDC								*
************************************************************************/
XglDC::XglDC(CanvasPane& parentCanvasPane, u_int w, u_int h)
    :CanvasPaneDC(parentCanvasPane, w, h),
     CanvasPaneDC3(parentCanvasPane, w, h),
     _xglwin(0),
     _xglctx(0),
     _wc_to_cc(xgl_object_create(xglstate(),
				 XGL_TRANS,		NULL,
				 XGL_TRANS_DATA_TYPE,	XGL_DATA_DBL,
				 XGL_TRANS_DIMENSION,	XGL_TRANS_3D,
				 NULL)),
     _cc_to_vdc(xgl_object_create(xglstate(),
				  XGL_TRANS,		NULL,
				  XGL_TRANS_DATA_TYPE,	XGL_DATA_DBL,
				  XGL_TRANS_DIMENSION,	XGL_TRANS_3D,
				  NULL)),
     _pcache(xgl_object_create(xglstate(), XGL_PCACHE, NULL, NULL))
{
    XtVaSetValues(widget(), XtNbackingStore, NotUseful, NULL);

  // Set CC to VDC transform: simple perspective projection with z = w.
    Xgl_matrix_d3d	matrix;
    matrix[0][0] = matrix[1][1] = matrix[2][2] = matrix[2][3] = 1.0;
		   matrix[0][1] = matrix[0][2] = matrix[0][3] =
    matrix[1][0]		= matrix[1][2] = matrix[1][3] =
    matrix[2][0] = matrix[2][1] =
    matrix[3][0] = matrix[3][1] = matrix[3][2] = matrix[3][3] = 0.0;
    xgl_transform_write_specific(_cc_to_vdc, matrix,
				 XGL_TRANS_MEMBER_LIM_PERSPECTIVE);

  // Bind Pcache to the context.
    xgl_object_set(_pcache, XGL_PCACHE_CONTEXT, _xglctx, NULL);
}

XglDC::~XglDC()
{
    xgl_object_destroy(_pcache);
    xgl_object_destroy(_cc_to_vdc);
    xgl_object_destroy(_wc_to_cc);
    xgl_object_destroy(_xglctx);
    xgl_object_destroy(_xglwin);
}

DC&
XglDC::setSize(u_int width, u_int height, u_int mul, u_int div)
{
    CanvasPaneDC::setSize(width, height, mul, div);
    return setViewport();
}

DC&
XglDC::setLayer(Layer layer)
{
    CanvasPaneDC::setLayer(layer);

    Xgl_cmap	xglcmap;
    xgl_object_get(_xglwin, XGL_RAS_COLOR_MAP, &xglcmap);
    if (getLayer() == DC::UNDERLAY)
    {
	Array<u_long>	pixels = colormap().getUnderlayPixels();

	if (colormap().getMode() == Colormap::RGBColor &&
	    colormap().vinfo().c_class == PseudoColor)
	{
	    Xgl_usgn32	dims[3];
	    dims[0] = colormap().rDim();
	    dims[1] = colormap().gDim();
	    dims[2] = colormap().bDim();
	    xgl_object_set(xglcmap,
			   XGL_CMAP_NAME,		(::Colormap)colormap(),
			   XGL_CMAP_COLOR_CUBE_SIZE,	dims,
			   NULL);
	}
	else
	{
	    xgl_object_set(xglcmap,
			   XGL_CMAP_NAME,		(::Colormap)colormap(),
			   XGL_CMAP_COLOR_TABLE_SIZE,	pixels.dim(),
			   NULL);
	}
	xgl_object_set(_xglwin,
		       XGL_WIN_RAS_PIXEL_MAPPING,	(u_long*)pixels,
		       NULL);
	xgl_object_set(_xglctx,
		       XGL_CTX_PLANE_MASK,	colormap().getUnderlayPlanes(),
		       NULL);
    }
    else
    {
	if (colormap().getOverlayPlanes() != 0)
	{
	    Array<u_long>	pixels = colormap().getOverlayPixels();

	    xgl_object_set(xglcmap,
			   XGL_CMAP_NAME,		(::Colormap)colormap(),
			   XGL_CMAP_COLOR_TABLE_SIZE,	pixels.dim(),
			   NULL);
	    xgl_object_set(_xglwin,
			   XGL_WIN_RAS_PIXEL_MAPPING,	(u_long*)pixels,
			   NULL);
	}
	xgl_object_set(_xglctx,
		       XGL_CTX_PLANE_MASK,	colormap().getOverlayPlanes(),
		       NULL);
    }

    return *this;
}

DC3&
XglDC::setInternal(int u0, int v0, double ku, double kv,
		   double near, double far)
{
    Xgl_bounds_d3d	vdc_window;
    vdc_window.xmin = -u0 / ku;
    vdc_window.xmax = (width()  - u0) / ku;
    vdc_window.ymin = -v0 / kv;
    vdc_window.ymax = (height() - v0) / kv;
    vdc_window.zmin = near;
    vdc_window.zmax = far;
    xgl_object_set(_xglctx, XGL_CTX_VDC_WINDOW, &vdc_window, NULL);

    return setViewport();
}

DC3&
XglDC::setExternal(const Vector<double>& t, const Matrix<double>& Rt)
{
    Xgl_matrix_d3d	matrix;
    for (int i = 0; i < 3; ++i)
    {
	for (int j = 0; j < 3; ++j)
	    matrix[i][j] = Rt[j][i];
	matrix[i][3] = matrix[3][i] = 0.0;
    }
    matrix[3][3] = 1.0;
    xgl_transform_write_specific(_wc_to_cc, matrix, XGL_TRANS_MEMBER_ROTATION);

    Xgl_pt_d3d	pt;
    pt.x = -t[0];
    pt.y = -t[1];
    pt.z = -t[2];
    Xgl_pt	offset;
    offset.pt_type = XGL_PT_D3D;
    offset.pt.d3d  = &pt;
    xgl_transform_translate(_wc_to_cc, &offset, XGL_TRANS_PRECONCAT);

    return setViewTransform();
}

const DC3&
XglDC::getInternal(int& u0, int& v0, double& ku, double& kv,
		   double& near, double& far) const
{
    Xgl_bounds_d3d	vdc_window;
    xgl_object_get(_xglctx, XGL_CTX_VDC_WINDOW, &vdc_window);
    ku	 = width()  / (vdc_window.xmax - vdc_window.xmin);
    kv	 = height() / (vdc_window.ymax - vdc_window.ymin);
    u0	 = -ku * vdc_window.xmin;
    v0	 = -ku * vdc_window.ymin;
    near = vdc_window.zmin;
    far	 = vdc_window.zmax;
    
    return *this;
}

const DC3&
XglDC::getExternal(Vector<double>& t, Matrix<double>& Rt) const
{
    Xgl_trans	view_trans;
    xgl_object_get(_xglctx, XGL_CTX_VIEW_TRANS, &view_trans);
    Xgl_matrix_d3d	matrix;
    xgl_transform_read(view_trans, matrix);
    Rt.resize(3, 3);
    Rt[0][0] = matrix[0][0];
    Rt[0][1] = matrix[1][0];
    Rt[0][2] = matrix[2][0];
    Rt[1][0] = matrix[0][1];
    Rt[1][1] = matrix[1][1];
    Rt[1][2] = matrix[2][1];
    Rt[2][0] = matrix[0][2];
    Rt[2][1] = matrix[1][2];
    Rt[2][2] = matrix[2][2];
    t.resize(3);
    t[0] = -matrix[3][0];
    t[1] = -matrix[3][1];
    t[2] = -matrix[3][2];
    t *= Rt;

    return *this;
}

DC3&
XglDC::translate(double dist)
{
    DC3::translate(dist);

    Xgl_pt_d3d	pt;
    pt.x = pt.y = pt.z = 0.0;
    switch (getAxis())
    {
      case DC3::X:
	pt.x = -dist;
	break;

      case DC3::Y:
	pt.y = -dist;
	break;

      case DC3::Z:
	pt.z = -dist;
	break;
    }

    Xgl_pt	displacement;
    displacement.pt_type = XGL_PT_D3D;
    displacement.pt.d3d  = &pt;
    xgl_transform_translate(_wc_to_cc, &displacement, XGL_TRANS_POSTCONCAT);

    return setViewTransform();
}

DC3&
XglDC::rotate(double angle)
{
    Xgl_axis	axis;
    switch (getAxis())
    {
      case DC3::X:
	axis = XGL_AXIS_X;
        break;
        
      case DC3::Y:
	axis = XGL_AXIS_Y;
        break;

      default:
	axis = XGL_AXIS_Z;
        break;
    }

    Xgl_pt_d3d  pt;
    pt.x = pt.y = 0.0;
    pt.z = -getDistance();
    Xgl_pt      displacement;
    displacement.pt_type = XGL_PT_D3D;
    displacement.pt.d3d  = &pt;

    xgl_transform_translate(_wc_to_cc, &displacement, XGL_TRANS_POSTCONCAT);
    xgl_transform_rotate(_wc_to_cc, -angle, axis, XGL_TRANS_POSTCONCAT);
    pt.z = -pt.z;
    xgl_transform_translate(_wc_to_cc, &displacement, XGL_TRANS_POSTCONCAT);

    return setViewTransform();
}

XglDC&
XglDC::clearXgl()
{
    xgl_window_raster_resize(_xglwin);
    xgl_context_new_frame(_xglctx);
    return *this;
}

XglDC&
XglDC::syncXgl()
{
    xgl_context_flush(_xglctx, XGL_FLUSH_SYNCHRONIZE);
    return *this;
}

/*
 *  Protected member functions
 */
void
XglDC::initializeGraphics()
{
  // Create XGL window and XGL context.
    Xgl_X_window	xglxwin;
    xglxwin.X_display = colormap().display();
    xglxwin.X_window  = drawable();
    xglxwin.X_screen  = colormap().vinfo().screen;

    Xgl_obj_desc	desc;
    desc.win_ras.type = XGL_WIN_X;
    desc.win_ras.desc = &xglxwin;

    _xglwin = xgl_object_create(xglstate(),
				XGL_WIN_RAS,		&desc,
			      	XGL_DEV_COLOR_TYPE,
				(colormap().getMode() ==
				 Colormap::IndexedColor ?
				 XGL_COLOR_INDEX : XGL_COLOR_RGB),
				NULL);
    _xglctx = xgl_object_create(xglstate(),
				XGL_3D_CTX,		NULL,
				XGL_CTX_DEVICE,		_xglwin,
				XGL_CTX_DEFERRAL_MODE,	XGL_DEFER_ASTI,
				XGL_CTX_VDC_MAP,	XGL_VDC_MAP_OFF, //?
				NULL);
    if (_xglwin == NULL || _xglctx == NULL)
	throw std::runtime_error("TU::v::XglDC::initializeGraphics(): failed to create window/context!!");
    
  // Set initial internal and external parameters.
    CanvasPaneDC3::initializeGraphics();
}

/*
 *  Private member functions
 */
XglDC&
XglDC::setViewTransform()
{
    Xgl_trans   view_trans;
    xgl_object_get(_xglctx, XGL_CTX_VIEW_TRANS, &view_trans);
    xgl_transform_multiply(view_trans, _wc_to_cc, _cc_to_vdc);

    return *this;
}

XglDC&
XglDC::setViewport()
{
    Xgl_bounds_d3d	dc_viewport;
    xgl_object_get(_xglctx, XGL_CTX_DC_VIEWPORT, &dc_viewport);
    dc_viewport.xmin = dc_viewport.ymin = 0.0;
    dc_viewport.xmax = deviceWidth();
    dc_viewport.ymax = deviceHeight();
    xgl_object_set(_xglctx, XGL_CTX_DC_VIEWPORT, &dc_viewport, NULL);

    return *this;
}

/************************************************************************
*  Manipulators								*
************************************************************************/
XglDC&
clearXgl(XglDC& dc)
{
    return dc.clearXgl();
}

XglDC&
syncXgl(XglDC& dc)
{
    return dc.syncXgl();
}
 
}
}
