/*
 *  $Id: MyDialog.h,v 1.4 2012-08-29 19:35:49 ueshiba Exp $
 */
#include <gtk/gtk.h>
#include "TU/V4L2++.h"

namespace TU
{
/************************************************************************
*  class MyDialog							*
************************************************************************/
/*!
  V4L2Cameraについて，その注目領域(ROI: Region of Interest)
  を指定するためのdialog．
 */
class MyDialog
{
  public:
    MyDialog(const V4L2Camera& camera)					;
    ~MyDialog()								;
    
    void	getROI(size_t& u0, size_t& v0,
		       size_t& width, size_t& height)		const	;
    
  private:
    GtkWidget*	_dialog;
    GtkObject*	_u0;
    GtkObject*	_v0;
    GtkObject*	_width;
    GtkObject*	_height;
};
 
}
