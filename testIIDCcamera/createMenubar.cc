/*
 * testIIDCcamera: test program controlling an IIDC 1394-based Digital Camera
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
  カメラがサポートする画像フォーマットとその名称．
*/
struct MyFormat
{
    const IIDCCamera::Format	format;		//!< 画像フォーマット
    const char* const		name;		//!< その名称
    
};
static const MyFormat	format[] =
{
    {IIDCCamera::YUV444_160x120,   "160x120-YUV(4:4:4)"},
    {IIDCCamera::YUV422_320x240,   "320x240-YUV(4:2:2)"},
    {IIDCCamera::YUV411_640x480,   "640x480-YUV(4:1:1)"},
    {IIDCCamera::YUV422_640x480,   "640x480-YUV(4:2:2)"},
    {IIDCCamera::RGB24_640x480,    "640x480-RGB"},
    {IIDCCamera::MONO8_640x480,    "640x480-Y(mono)"},
    {IIDCCamera::MONO16_640x480,   "640x480-Y(mono16)"},
    {IIDCCamera::YUV422_800x600,   "800x600-YUV(4:2:2)"},
    {IIDCCamera::RGB24_800x600,    "800x600-RGB"},
    {IIDCCamera::MONO8_800x600,    "800x600-Y(mono)"},
    {IIDCCamera::YUV422_1024x768,  "1024x768-YUV(4:2:2)"},
    {IIDCCamera::RGB24_1024x768,   "1024x768-RGB"},
    {IIDCCamera::MONO8_1024x768,   "1024x768-Y(mono)"},
    {IIDCCamera::MONO16_800x600,   "800x600-Y(mono16)"},
    {IIDCCamera::MONO16_1024x768,  "1024x768-Y(mono16)"},
    {IIDCCamera::YUV422_1280x960,  "1280x960-YUV(4:2:2)"},
    {IIDCCamera::RGB24_1280x960,   "1280x960-RGB"},
    {IIDCCamera::MONO8_1280x960,   "1280x960-Y(mono)"},
    {IIDCCamera::YUV422_1600x1200, "1600x1200-YUV(4:2:2)"},
    {IIDCCamera::RGB24_1600x1200,  "1600x1200-RGB"},
    {IIDCCamera::MONO8_1600x1200,  "1600x1200-Y(mono)"},
    {IIDCCamera::MONO16_1280x960,  "1280x960-Y(mono16)"},
    {IIDCCamera::MONO16_1600x1200, "1600x1200-Y(mono16)"},
    {IIDCCamera::Format_7_0,       "Format_7_0"},
    {IIDCCamera::Format_7_1,       "Format_7_1"},
    {IIDCCamera::Format_7_2,       "Format_7_2"},
    {IIDCCamera::Format_7_3,       "Format_7_3"},
    {IIDCCamera::Format_7_4,       "Format_7_4"},
    {IIDCCamera::Format_7_5,       "Format_7_5"},
    {IIDCCamera::Format_7_6,       "Format_7_6"},
    {IIDCCamera::Format_7_6,       "Format_7_7"}
};
static const int	NFORMATS = sizeof(format)/sizeof(format[0]);

/*!
  カメラがサポートするフレームレートとその名称．
*/
struct MyFrameRate
{
    const IIDCCamera::FrameRate	frameRate;	//!< フレームレート
    const char* const		name;		//!< その名称
};
static const MyFrameRate	frameRate[] =
{
    {IIDCCamera::FrameRate_1_875, "1.875fps"},
    {IIDCCamera::FrameRate_3_75,  "3.75fps"},
    {IIDCCamera::FrameRate_7_5,   "7.5fps"},
    {IIDCCamera::FrameRate_15,    "15fps"},
    {IIDCCamera::FrameRate_30,    "30fps"},
    {IIDCCamera::FrameRate_60,    "60fps"},
    {IIDCCamera::FrameRate_120,   "120fps"},
    {IIDCCamera::FrameRate_240,   "240fps"},
    {IIDCCamera::FrameRate_x,     "custom"}
};
static const int	NRATES=sizeof(frameRate)/sizeof(frameRate[0]);

/*!
  カメラ, 画像フォーマット, フレームレートの3ツ組．コールバック関数:
  CBmenuitem() の引数として渡される．
 */
struct FormatAndFrameRate
{
    MyIIDCCamera*		camera;
    IIDCCamera::Format	format;
    IIDCCamera::FrameRate	frameRate;
};
static FormatAndFrameRate	fmtAndFRate[NFORMATS * NRATES];

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
*  static functions							*
************************************************************************/
static std::ostream&
operator <<(std::ostream& out, const MyIIDCCamera& camera)
{
    using namespace	std;
    
    for (int i = 0; i < NFORMATS; ++i)
	if (camera.getFormat() == format[i].format)
	{
	    out << format[i].name;
	    break;
	}

    for (int i = 0; i < NRATES; ++i)
	if (camera.getFrameRate() == frameRate[i].frameRate)
	{
	    out << ' ' << frameRate[i].name;
	    break;
	}

    struct
    {
	IIDCCamera::Feature	feature;
	const char*		name;
    } feature[] =
    {
	{IIDCCamera::BRIGHTNESS,	"BRIGHTNESS"},
	{IIDCCamera::AUTO_EXPOSURE,	"AUTO_EXPOSURE"},
	{IIDCCamera::SHARPNESS,		"SHARPNESS"},
	{IIDCCamera::WHITE_BALANCE,	"WHITE_BALANCE"},
	{IIDCCamera::HUE,		"HUE"},
	{IIDCCamera::SATURATION,	"SATURATION"},
	{IIDCCamera::GAMMA,		"GAMMA"},
	{IIDCCamera::SHUTTER,		"SHUTTER"},
	{IIDCCamera::GAIN,		"GAIN"},
	{IIDCCamera::IRIS,		"IRIS"},
	{IIDCCamera::FOCUS,		"FOCUS"},
	{IIDCCamera::TEMPERATURE,	"TEMPERATURE"},
	{IIDCCamera::ZOOM,		"ZOOM"},
	{IIDCCamera::PAN,		"PAN"},
	{IIDCCamera::TILT,		"TILT"}
    };
    const int	nfeatures = sizeof(feature) / sizeof(feature[0]);
		    
    for (int i = 0; i < nfeatures; ++i)
    {
	u_int	inq = camera.inquireFeatureFunction(feature[i].feature);
	if ((inq & IIDCCamera::Presence) &&
	    (inq & IIDCCamera::Manual)   &&
	    (inq & IIDCCamera::ReadOut))
	{
	    out << ' ' << feature[i].name << ' ';

	    if ((inq & IIDCCamera::Auto) && 
		camera.isAuto(feature[i].feature))
		out << -1;
	    else if (feature[i].feature == IIDCCamera::WHITE_BALANCE)
	    {
		u_int	ub, vr;
		camera.getWhiteBalance(ub, vr);
		out << ub << ' ' << vr;
	    }
	    else
		out << camera.getValue(feature[i].feature);
	}
    }

    return out << endl;
}

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
    FormatAndFrameRate*	fmtAndFRate = (FormatAndFrameRate*)userdata;
    if (fmtAndFRate->format >= IIDCCamera::Format_7_0)
    {
	MyDialog	dialog(fmtAndFRate->camera->
			       getFormat_7_Info(fmtAndFRate->format));
	u_int		u0, v0, width, height;
	IIDCCamera::PixelFormat
			pixelFormat = dialog.getROI(u0, v0, width, height);
	fmtAndFRate->camera->
	    setFormat_7_ROI(fmtAndFRate->format, u0, v0, width, height).
	    setFormat_7_PixelFormat(fmtAndFRate->format, pixelFormat);
    }
    fmtAndFRate->camera->setFormatAndFrameRate(fmtAndFRate->format,
					       fmtAndFRate->frameRate);
}

//! 選択されたファイルに画像をセーブするためのコールバック関数．
/*!
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
*/
static void
CBfileSelectionOK(GtkWidget* filesel, gpointer userdata)
{
    CameraAndFileSelection*	camAndFSel = (CameraAndFileSelection*)userdata;
    std::ofstream	out(gtk_file_selection_get_filename(
				GTK_FILE_SELECTION(camAndFSel->filesel)));
    if (out)
	camAndFSel->camera->save(out);
    gtk_widget_destroy(camAndFSel->filesel);
}

//! 画像をセーブするファイルを選択するdialogを表示するためのコールバック関数．
/*!
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
*/
static void
CBsave(GtkMenuItem*, gpointer userdata)
{
    GtkWidget*		filesel = gtk_file_selection_new("Save image");
    gtk_signal_connect(GTK_OBJECT(filesel), "destroy",
		       GTK_SIGNAL_FUNC(gtk_main_quit), filesel);
    cameraAndFileSelection.camera  = (MyIIDCCamera*)userdata;
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
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
*/
static void
CBexit(GtkMenuItem*, gpointer userdata)
{
    using namespace	std;
    
    MyIIDCCamera*	camera = (MyIIDCCamera*)userdata;
    camera->stopContinuousShot();
    cout << "0x" << hex << setw(16) << setfill('0')
	 << camera->globalUniqueId() << dec << ' ' << *camera;
    gtk_exit(0);
}

/************************************************************************
*  global functions							*
************************************************************************/
//! メニューバーを生成する．
/*!
  IEEE1394カメラがサポートしている画像フォーマットとフレームレートを調べて
  メニュー項目を決定する．
  \param camera		IEEE1394カメラ
  \return		生成されたメニューバー
*/
GtkWidget*
createMenubar(MyIIDCCamera& camera)
{
    GtkWidget*	menubar	= gtk_menu_bar_new();

  // "File"メニューを生成．
    GtkWidget*	menu = gtk_menu_new();
    GtkWidget*	item = gtk_menu_item_new_with_label("Save");
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
    IIDCCamera::Format		current_format = camera.getFormat();
    IIDCCamera::FrameRate	current_rate   = camera.getFrameRate();
    int	nitems = 0;
    for (int i = 0; i < NFORMATS; ++i)	// 全てのフォーマットについて...
    {
      // このフォーマットがサポートされているか調べる．
	u_int		inq = camera.inquireFrameRate(format[i].format);
	GtkWidget*	submenu = 0;
	for (int j = 0; j < NRATES; ++j) // 全てのフレームレートについて...
	{
	  // このフレームレートがサポートされているか調べる．
	    if (inq & frameRate[j].frameRate)
	    {
	      // フレームレートを指定するためのサブメニューを作る．
		if (submenu == 0)
		    submenu = gtk_menu_new();
		GtkWidget* item
		    = gtk_menu_item_new_with_label(frameRate[j].name);
		gtk_menu_append(GTK_MENU(submenu), item);
		fmtAndFRate[nitems].camera = &camera;
		fmtAndFRate[nitems].format = format[i].format;
		fmtAndFRate[nitems].frameRate = frameRate[j].frameRate;
		gtk_signal_connect(GTK_OBJECT(item), "activate",
				   GTK_SIGNAL_FUNC(CBmenuitem),
				   (gpointer)&fmtAndFRate[nitems]);
		++nitems;
	    }
	}
	
      // 少なくとも1つのフレームレートがサポートされていれば，この
      // フォーマットがサポートされていることになる．
	if (submenu != 0)
	{
	    GtkWidget*	item = gtk_menu_item_new_with_label(format[i].name);
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
