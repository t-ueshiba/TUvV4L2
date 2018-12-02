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
 *  $Id: MyDialog.cc,v 1.5 2012-08-29 19:35:49 ueshiba Exp $
 */
#include "MyDialog.h"

namespace TU
{
/************************************************************************
*  local data								*
************************************************************************/
/*!
  Format_7_xでサポートされる画素フォーマットとその名称．
*/
struct MyPixelFormat
{
    const IIDCCamera::PixelFormat	pixelFormat;	//!< 画素フォーマット
    const char* const			name;		//!< その名称
    IIDCCamera::PixelFormat*		pixelFormatDst;	//!< フォーマット設定先
};
static MyPixelFormat	pixelFormat[] =
{
    {IIDCCamera::MONO_8,		"Y(mono)",		nullptr},
    {IIDCCamera::YUV_411,		"YUV(4:1:1)",		nullptr},
    {IIDCCamera::YUV_422,		"YUV(4:2:2)",		nullptr},
    {IIDCCamera::YUV_444,		"YUV(4:4:4)",		nullptr},
    {IIDCCamera::RGB_24,		"RGB",			nullptr},
    {IIDCCamera::MONO_16,		"Y(mono16)",		nullptr},
    {IIDCCamera::RGB_48,		"RGB(color48)",		nullptr},
    {IIDCCamera::SIGNED_MONO_16,	"Y(signed mono16)",	nullptr},
    {IIDCCamera::SIGNED_RGB_48,		"RGB(signed color48)",	nullptr},
    {IIDCCamera::RAW_8,			"RAW(raw8)",		nullptr},
    {IIDCCamera::RAW_16,		"RAW(raw16)",		nullptr}
};
static constexpr int	NPIXELFORMATS = sizeof(pixelFormat)
				      / sizeof(pixelFormat[0]);

/************************************************************************
*  callback functions							*
************************************************************************/
//! ROIの原点やサイズを与えられたステップ単位で変化させるためのコールバック関数．
/*!
  \param adj		ROIの原点やサイズを保持している adjuster
  \param userdata	変化のステップ
*/
static void
CBadjust(GtkAdjustment* adj, gpointer userdata)
{
    const auto	step = *static_cast<u_int*>(userdata);
    const auto	val  = gfloat(step * ((u_int(adj->value) + step/2) / step));
    gtk_adjustment_set_value(adj, val);
}
    
//! 画素形式を指定するためのコールバック関数．
/*!
  \param userdata	SettingPixelFormat (画素形式の設定値と設定先の2ツ組)
*/
static void
CBmenuitem(GtkMenuItem*, gpointer userdata)
{
    auto	pixelFormat = static_cast<MyPixelFormat*>(userdata);
    *(pixelFormat->pixelFormatDst) = pixelFormat->pixelFormat;
}
    
/************************************************************************
*  class MyDialog							*
************************************************************************/
//! ROIを指定するためのdialog windowを生成する
/*!
  \param fmt7info	ROIを指定したいFormat_7_xに関する情報．
*/
MyDialog::MyDialog(const IIDCCamera::Format_7_Info& fmt7info)
    :_fmt7info(fmt7info),
     _dialog(gtk_dialog_new()),
     _u0(gtk_adjustment_new(_fmt7info.u0, 0, _fmt7info.maxWidth,
			    _fmt7info.unitU0, _fmt7info.unitU0, 0.0)),
     _v0(gtk_adjustment_new(_fmt7info.v0, 0, _fmt7info.maxHeight,
			    _fmt7info.unitV0, _fmt7info.unitV0, 0.0)),
     _width(gtk_adjustment_new(_fmt7info.width, 0, _fmt7info.maxWidth,
			       _fmt7info.unitWidth, _fmt7info.unitWidth, 0.0)),
     _height(gtk_adjustment_new(_fmt7info.height, 0, _fmt7info.maxHeight,
				_fmt7info.unitHeight, _fmt7info.unitHeight,
				0.0)),
     _packetSize(gtk_adjustment_new(_fmt7info.bytePerPacket,
				    _fmt7info.unitBytePerPacket,
				    _fmt7info.maxBytePerPacket,
				    _fmt7info.unitBytePerPacket,
				    _fmt7info.unitBytePerPacket, 0.0)),
     _pixelFormat(_fmt7info.pixelFormat)
{
  // dialogの属性を設定．
    gtk_window_set_modal(GTK_WINDOW(_dialog), TRUE);
    gtk_window_set_title(GTK_WINDOW(_dialog), "Setting ROI for Format_7_x");
    gtk_window_set_policy(GTK_WINDOW(_dialog), FALSE, FALSE, TRUE);
    
  // 変化をステップ単位に強制するため，コールバック関数を登録．
    gtk_signal_connect(GTK_OBJECT(_u0), "value_changed",
		       GTK_SIGNAL_FUNC(CBadjust),
		       (gpointer)&_fmt7info.unitU0);
    gtk_signal_connect(GTK_OBJECT(_v0), "value_changed",
		       GTK_SIGNAL_FUNC(CBadjust),
		       (gpointer)&_fmt7info.unitV0);
    gtk_signal_connect(GTK_OBJECT(_width), "value_changed",
		       GTK_SIGNAL_FUNC(CBadjust),
		       (gpointer)&_fmt7info.unitWidth);
    gtk_signal_connect(GTK_OBJECT(_height), "value_changed",
		       GTK_SIGNAL_FUNC(CBadjust),
		       (gpointer)&_fmt7info.unitHeight);
    gtk_signal_connect(GTK_OBJECT(_packetSize), "value_changed",
		       GTK_SIGNAL_FUNC(CBadjust),
		       (gpointer)&_fmt7info.unitBytePerPacket);
    
  // widgetを並べるためのtable．
    const auto	table = gtk_table_new(2, 6, FALSE);

  // 4つのadjustmentのためのlabelと操作用scaleを生成してtableにパック．
    auto	label = gtk_label_new("u0:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 0, 1);
    auto	scale = gtk_hscale_new(GTK_ADJUSTMENT(_u0));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 0, 1);
    
    label = gtk_label_new("v0:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 1, 2);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_v0));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 1, 2);
    
    label = gtk_label_new("width:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 2, 3);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_width));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 2, 3);
    
    label = gtk_label_new("height:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 3, 4);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_height));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 3, 4);

    label = gtk_label_new("packet size:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 4, 5);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_packetSize));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 4, 5);

  // PixelFormat設定用のオプションメニューを生成してtableにパック．
    const auto	menu = gtk_menu_new();
    guint	current = 0;
    for (int nitems = 0, i = 0; i < NPIXELFORMATS; ++i)
	if (_fmt7info.availablePixelFormats & pixelFormat[i].pixelFormat)
	{
	    const auto
		item = gtk_menu_item_new_with_label(pixelFormat[i].name);
	    gtk_menu_append(GTK_MENU(menu), item);
	    pixelFormat[i].pixelFormatDst = &_pixelFormat;
	    gtk_signal_connect(GTK_OBJECT(item), "activate",
			       GTK_SIGNAL_FUNC(CBmenuitem),
			       (gpointer)&pixelFormat[i]);
	    if (_fmt7info.pixelFormat == pixelFormat[i].pixelFormat)
		current = nitems;
	    ++nitems;
	}
    const auto	option_menu = gtk_option_menu_new();
    gtk_option_menu_set_menu(GTK_OPTION_MENU(option_menu), menu);
    gtk_option_menu_set_history(GTK_OPTION_MENU(option_menu), current);
    gtk_table_attach_defaults(GTK_TABLE(table), option_menu, 1, 2, 5, 6);
    
  // ROI設定用のボタンを生成してtableにパック．
    const auto	button = gtk_button_new_with_label("Set");
    gtk_signal_connect(GTK_OBJECT(button), "clicked",
		       GTK_SIGNAL_FUNC(gtk_main_quit), _dialog);
    gtk_table_attach_defaults(GTK_TABLE(table), button, 0, 2, 6, 7);

  // tableをdialogに収める．
    gtk_container_add(GTK_CONTAINER(GTK_DIALOG(_dialog)->action_area), table);

  // メッセージをdialogに収める．
    label = gtk_label_new("Set the Region of Interest(ROI).");
    gtk_container_add(GTK_CONTAINER(GTK_DIALOG(_dialog)->vbox), label);
    
    gtk_widget_show_all(_dialog);

    gtk_main();			// ROI設定用のボタンを押すとループを抜け出す．
}

//! ROIを指定するためのdialog windowを破壊する
MyDialog::~MyDialog()
{
    gtk_widget_destroy(_dialog);
}

//! 指定されたROIをdialog windowから読み取る
/*!
  \param u0	ROIの左上隅の横座標．
  \param v0	ROIの左上隅の縦座標．
  \param width	ROIの幅．
  \param height	ROIの高さ．
 */
IIDCCamera::PixelFormat
MyDialog::getROI(u_int& u0, u_int& v0,
		 u_int& width, u_int& height, u_int& packetSize)
{
    u0		= u_int(GTK_ADJUSTMENT(_u0)->value);
    v0		= u_int(GTK_ADJUSTMENT(_v0)->value);
    width	= u_int(GTK_ADJUSTMENT(_width)->value);
    height	= u_int(GTK_ADJUSTMENT(_height)->value);
    packetSize	= u_int(GTK_ADJUSTMENT(_packetSize)->value);

    return _pixelFormat;
}
    
};
