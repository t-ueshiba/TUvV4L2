/*
 * testIIDCcamera: test program controlling an IIDC-based Digital Camera
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
 *  $Id: MyIIDCCamera.h,v 1.8 2012-08-29 19:35:49 ueshiba Exp $
 */
#include <gtk/gtk.h>
#include <vector>
#include "TU/IIDC++.h"

namespace TU
{
/************************************************************************
*  class MyIIDCCamera							*
************************************************************************/
/*!
  IIDCデジタルカメラを表すクラス．さらに，GTK+ を用いた画像表示のための
  canvas (GTK+ の drawing area widget)，入力画像バッファ，RGB カラー画像
  表示用のバッファを確保する機能を持つ．
*/
class MyIIDCCamera : public IIDCCamera
{
  public:
			MyIIDCCamera(uint64_t uniqId)			;
			MyIIDCCamera(const MyIIDCCamera& camera)= delete;
    MyIIDCCamera&	operator =(const MyIIDCCamera&)		= delete;
			MyIIDCCamera(MyIIDCCamera&& camera)	= default;
    MyIIDCCamera&	operator =(MyIIDCCamera&&)		= default;

  //! 画像の表示領域となるキャンバスを返す．
    GtkWidget*		canvas()				const	;
    IIDCCamera&		setFormatAndFrameRate(Format format,
					      FrameRate rate)		;
    void		idle()						;
    void		draw()						;
    std::ostream&	save(std::ostream& out)			const	;
    void		setCommands(GtkWidget* commands,
				    GtkWidget* parent)			;
    void		refreshCommands()				;
    
  private:
    mutable std::vector<u_char>	_buf;		// 入力画像用バッファ
    std::vector<RGB>		_rgb;		// RGB カラー画像(表示用)バッファ

    GtkWidget*			_canvas;	// 画像を表示する領域
    GtkWidget*			_commands;
    GtkWidget*			_comParent;
};

inline GtkWidget*
MyIIDCCamera::canvas() const
{
    return _canvas;
}

}
