/*
 *  $Id: MyDialog.h,v 1.1 2002-12-18 04:34:08 ueshiba Exp $
 */
#include <gtk/gtk.h>
#include "TU/Ieee1394++.h"

namespace TU
{
/************************************************************************
*  class MyDialog							*
************************************************************************/
/*!
  Format_7_x型のIeee1394Camera::Formatについて，その注目領域(ROI: Region
  of Interest)を指定するためのdialog．
 */
class MyDialog
{
  public:
    MyDialog(const Ieee1394Camera::Format_7_Info& fmt7info)		;
    ~MyDialog()								;
    
    void	getROI(u_int& u0, u_int& v0,
		       u_int& width, u_int& height)			;
    
  private:
    const Ieee1394Camera::Format_7_Info&	_fmt7info;
    GtkWidget* const				_dialog;
    GtkObject* const				_u0;
    GtkObject* const				_v0;
    GtkObject* const				_width;
    GtkObject* const				_height;
};
 
}
