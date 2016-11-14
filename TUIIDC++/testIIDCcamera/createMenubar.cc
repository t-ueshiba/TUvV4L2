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
 *  $Id: createMenubar.cc,v 1.11 2012-08-29 19:35:49 ueshiba Exp $
 */
#if HAVE_CONFIG_H
#  include <config.h>
#endif
#include "MyIIDCCamera.h"
#include "MyDialog.h"
#include <iostream>
#include <iomanip>
#include <fstream>

#define DEFAULT_IMAGE_FILE_NAME	"testIIDCcamera.pbm"

namespace TU
{
/************************************************************************
*  local data								*
************************************************************************/
/*!
  カメラ, 画像フォーマット, フレームレートの3ツ組．コールバック関数:
  CBmenuitem() の引数として渡される．
 */
struct FormatAndFrameRate
{
    MyIIDCCamera*		camera;
    IIDCCamera::Format		format;
    IIDCCamera::FrameRate	frameRate;
};
static FormatAndFrameRate	fmtAndFRate[IIDCCamera::NFORMATS *
					    IIDCCamera::NRATES];

/*!
  カメラとfile selection widgetの2ツ組．コールバック関数:
  CBfileSelectionOK() の引数として渡される．
 */
struct CameraAndFileSelection
{
    MyIIDCCamera*		camera;
    GtkWidget*			filesel;
};
static CameraAndFileSelection	cameraAndFileSelection;

/************************************************************************
*  callback functions							*
************************************************************************/
//! フォーマットとフレームレートを設定するためのコールバック関数．
/*!
  \param userdata	FormatAndFrameRate (カメラ, 設定すべきフォーマット,
			設定すべきフレームレートの3ツ組)
*/
static void
CBmenuitem(GtkMenuItem*, gpointer userdata)
{
    const auto	fmtAndFRate = static_cast<FormatAndFrameRate*>(userdata);
    if (fmtAndFRate->format >= IIDCCamera::Format_7_0)
    {
	MyDialog	dialog(fmtAndFRate->camera->
			       getFormat_7_Info(fmtAndFRate->format));
	u_int		u0, v0, width, height;
	const auto	pixelFormat = dialog.getROI(u0, v0, width, height);
	fmtAndFRate->camera->
	    setFormat_7_ROI(fmtAndFRate->format, u0, v0, width, height).
	    setFormat_7_PixelFormat(fmtAndFRate->format, pixelFormat);
    }
    fmtAndFRate->camera->setFormatAndFrameRate(fmtAndFRate->format,
					       fmtAndFRate->frameRate);
}

//! 選択されたファイルに画像をセーブするためのコールバック関数．
/*!
  \param userdata	MyIIDCCamera (IIDCカメラ)
*/
static void
CBfileSelectionOK(GtkWidget* filesel, gpointer userdata)
{
    const auto		camAndFSel = static_cast<CameraAndFileSelection*>(
					 userdata);
    std::ofstream	out(gtk_file_selection_get_filename(
				GTK_FILE_SELECTION(camAndFSel->filesel)));
    if (out)
	camAndFSel->camera->save(out);
    gtk_widget_destroy(camAndFSel->filesel);
}

//! 画像をセーブするファイルを選択するdialogを表示するためのコールバック関数．
/*!
  \param userdata	MyIIDCCamera (IIDCカメラ)
*/
static void
CBsave(GtkMenuItem*, gpointer userdata)
{
    const auto	filesel = gtk_file_selection_new("Save image");
    gtk_signal_connect(GTK_OBJECT(filesel), "destroy",
		       GTK_SIGNAL_FUNC(gtk_main_quit), filesel);
    cameraAndFileSelection.camera  = static_cast<MyIIDCCamera*>(userdata);
    cameraAndFileSelection.filesel = filesel;
    gtk_signal_connect(GTK_OBJECT(GTK_FILE_SELECTION(filesel)->ok_button),
		       "clicked", (GtkSignalFunc)CBfileSelectionOK,
		       &cameraAndFileSelection);
    gtk_signal_connect_object(GTK_OBJECT(GTK_FILE_SELECTION(filesel)
					 ->cancel_button), "clicked",
			      (GtkSignalFunc)gtk_widget_destroy,
			      GTK_OBJECT(filesel));
    gtk_file_selection_set_filename(GTK_FILE_SELECTION(filesel),
				    DEFAULT_IMAGE_FILE_NAME);
    gtk_widget_show(filesel);
    gtk_main();
}

//! カメラの設定値を標準出力に書き出して終了するためのコールバック関数．
/*!
  \param userdata	MyIIDCCamera (IIDCカメラ)
*/
static void
CBexit(GtkMenuItem*, gpointer userdata)
{
    gtk_main_quit();
}

/************************************************************************
*  global functions							*
************************************************************************/
//! メニューバーを生成する．
/*!
  IIDCカメラがサポートしている画像フォーマットとフレームレートを調べて
  メニュー項目を決定する．
  \param camera		IIDCカメラ
  \return		生成されたメニューバー
*/
GtkWidget*
createMenubar(MyIIDCCamera& camera)
{
    const auto	menubar	= gtk_menu_bar_new();

  // "File"メニューを生成．
    auto	menu = gtk_menu_new();
    auto	item = gtk_menu_item_new_with_label("Save");
    gtk_signal_connect(GTK_OBJECT(item), "activate",
		       GTK_SIGNAL_FUNC(CBsave), &camera);
    gtk_menu_append(GTK_MENU(menu), item);
    item = gtk_menu_item_new_with_label("Quit");
    gtk_signal_connect(GTK_OBJECT(item), "activate",
		       GTK_SIGNAL_FUNC(CBexit), &camera);
    gtk_menu_append(GTK_MENU(menu), item);
    item = gtk_menu_item_new_with_label("File");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), menu);
    gtk_menu_bar_append(GTK_MENU_BAR(menubar), item);

  // "Format"メニューを生成．
    menu = gtk_menu_new();
  // 現在指定されている画像フォーマットおよびフレームレートを調べる．
    const auto	current_format = camera.getFormat();
    const auto	current_rate   = camera.getFrameRate();
    int		nitems = 0;
    for (const auto& format : IIDCCamera::formatNames)
    {
      // このフォーマットがサポートされているか調べる．
	const auto	inq = camera.inquireFrameRate(format.format);
	GtkWidget*	submenu = nullptr;
	for (const auto& frameRate : IIDCCamera::frameRateNames)
	{
	  // このフレームレートがサポートされているか調べる．
	    if (inq & frameRate.frameRate)
	    {
	      // フレームレートを指定するためのサブメニューを作る．
		if (submenu == 0)
		    submenu = gtk_menu_new();
		const auto
		    item = gtk_menu_item_new_with_label(frameRate.name);
		gtk_menu_append(GTK_MENU(submenu), item);
		fmtAndFRate[nitems].camera = &camera;
		fmtAndFRate[nitems].format = format.format;
		fmtAndFRate[nitems].frameRate = frameRate.frameRate;
		gtk_signal_connect(GTK_OBJECT(item), "activate",
				   GTK_SIGNAL_FUNC(CBmenuitem),
				   &fmtAndFRate[nitems]);
		++nitems;
	    }
	}
	
      // 少なくとも1つのフレームレートがサポートされていれば，この
      // フォーマットがサポートされていることになる．
	if (submenu != 0)
	{
	    const auto	item = gtk_menu_item_new_with_label(format.name);
	    gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), submenu);
	    gtk_menu_append(GTK_MENU(menu), item);
	}
    }
    item = gtk_menu_item_new_with_label("Format");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), menu);
    gtk_menu_bar_append(GTK_MENU_BAR(menubar), item);

    gtk_widget_show_all(menubar);
    
    return menubar;
}
 
}
