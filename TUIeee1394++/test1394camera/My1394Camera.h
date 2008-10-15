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
 *  $Id: My1394Camera.h,v 1.5 2008-10-15 00:22:29 ueshiba Exp $
 */
#include <gtk/gtk.h>
#include "TU/Ieee1394++.h"

namespace TU
{
/*!
  RGB カラー画像の画素を表す構造体．
*/
struct MyRGB
{
    u_char	r;	//!< 赤
    u_char	g;	//!< 青
    u_char	b;	//!< 緑
};

/************************************************************************
*  class My1394Camera							*
************************************************************************/
/*!
  IEEE1394デジタルカメラを表すクラス．さらに，GTK+ を用いた画像表示のための
  canvas (GTK+ の drawing area widget)，入力画像バッファ，RGB カラー画像
  表示用のバッファを確保する機能を持つ．
*/
class My1394Camera : public Ieee1394Camera
{
  public:
    My1394Camera(bool i1394b, u_int64 uniqId)				;
    ~My1394Camera()							;

  //! 画像の表示領域となるキャンバスを返す．
    GtkWidget*		canvas()				const	;
    Ieee1394Camera&	setFormatAndFrameRate(Format format,
					      FrameRate rate)		;
    void		idle()						;
    void		draw()						;
    std::ostream&	save(std::ostream& out)			const	;
    
  private:
    GtkWidget* const	_canvas;	// 画像を表示する領域
    u_char*		_buf;		// 入力画像用バッファ
    MyRGB*		_rgb;		// RGB カラー画像(表示用)バッファ
};

inline GtkWidget*
My1394Camera::canvas() const
{
    return _canvas;
}

}
