/*
 * test1394camera: test program controlling an IIDC 1394-based Digital Camera
 * Copyright (C) 2003 Toshio UESHIBA
 *   National Institute of Advanced Industrial Science and Technology (AIST)
 *
 * Written by Toshio UESHIBA <t.ueshiba@aist.go.jp>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  $Id: MyDialog.h,v 1.3 2008-10-15 00:22:29 ueshiba Exp $
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
    
    Ieee1394Camera::PixelFormat	getROI(u_int& u0, u_int& v0,
				       u_int& width, u_int& height)	;
    
  private:
    const Ieee1394Camera::Format_7_Info&	_fmt7info;
    GtkWidget* const				_dialog;
    GtkObject* const				_u0;
    GtkObject* const				_v0;
    GtkObject* const				_width;
    GtkObject* const				_height;
    Ieee1394Camera::PixelFormat			_pixelFormat;
};
 
}
