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
 *  $Id: createCommands.cc,v 1.9 2012-08-29 19:35:49 ueshiba Exp $
 */
#if HAVE_CONFIG_H
#  include <config.h>
#endif
#include "MyIIDCCamera.h"

namespace TU
{
/************************************************************************
*  local data								*
************************************************************************/
/*!
  カメラがサポートする機能とその名称．
*/
struct MyFeature
{
    const IIDCCamera::Feature	feature;	//!< カメラの機能
    const char* const		name;		//!< その名称
};
static const MyFeature	feature[] =
{
    {IIDCCamera::BRIGHTNESS,	"Brightness:"},
    {IIDCCamera::AUTO_EXPOSURE,	"Auto exposure:"},
    {IIDCCamera::SHARPNESS,	"Sharpness:"},
    {IIDCCamera::WHITE_BALANCE,	"White bal.(U/B):"},
    {IIDCCamera::WHITE_BALANCE,	"White bal.(V/R):"},
    {IIDCCamera::HUE,		"Hue:"},
    {IIDCCamera::SATURATION,	"Saturation:"},
    {IIDCCamera::GAMMA,		"Gamma:"},
    {IIDCCamera::SHUTTER,	"Shutter:"},
    {IIDCCamera::GAIN,		"Gain:"},
    {IIDCCamera::IRIS,		"Iris:"},
    {IIDCCamera::FOCUS,		"Focus:"},
    {IIDCCamera::TEMPERATURE,	"Temperature:"},
    {IIDCCamera::ZOOM,		"Zoom:"}
};
static const int		NFEATURES = sizeof(feature)/sizeof(feature[0]);

/*!
  カメラとその機能の2ツ組．3つのコールバック関数: CBturnOnOff(),
  CBsetAutoManual(), CBsetValue() の引数として渡される．
 */
struct CameraAndFeature
{
    IIDCCamera*		camera;		//!< カメラ
    IIDCCamera::Feature	feature;	//!< 操作したい機能
};
static CameraAndFeature		cameraAndFeature[NFEATURES];

/************************************************************************
*  callback functions							*
************************************************************************/
//! キャプチャボタンがonの間定期的に呼ばれるidle用コールバック関数．
/*!
  カメラから画像を取り込んでcanvasに表示する．
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
  \return		TRUEを返す
*/
static gint
CBidle(gpointer userdata)
{
    MyIIDCCamera*	camera = (MyIIDCCamera*)userdata;
    camera->idle();
    return TRUE;
}

//! キャプチャボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  timerを activate/deactivate する．
  \param toggle		キャプチャボタン
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
*/
static void
CBcontinuousShot(GtkWidget* toggle, gpointer userdata)
{
    static gint		idleTag;
    MyIIDCCamera*	camera = (MyIIDCCamera*)userdata;
    if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
    {
	idleTag = gtk_idle_add(CBidle, camera);	// idle処理を開始する．
	camera->continuousShot();	// カメラからの画像出力を開始する．
    }
    else
    {
	gtk_idle_remove(idleTag);	// idle処理を中止する．
	camera->stopContinuousShot();	// カメラからの画像出力を停止する．
    }
}

//! トリガモードボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  trigger modeを on/off する．
  \param toggle		トリガモードボタン
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
*/
static void
CBtriggerMode(GtkWidget* toggle, gpointer userdata)
{
    MyIIDCCamera*	camera = (MyIIDCCamera*)userdata;
    if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
	camera->turnOn(IIDCCamera::TRIGGER_MODE)
	    .setTriggerMode(IIDCCamera::Trigger_Mode0);
    else
	camera->turnOff(IIDCCamera::TRIGGER_MODE);
}

//! on/off ボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  あるカメラ機能を on/off する．
  \param toggle		on/off ボタン
  \param userdata	CameraAndFeature (IEEE1394カメラと on/off
			したい機能の2ツ組)
*/
static void
CBturnOnOff(GtkWidget* toggle, gpointer userdata)
{
    CameraAndFeature*	cameraAndFeature = (CameraAndFeature*)userdata;
    if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
	cameraAndFeature->camera->turnOn(cameraAndFeature->feature);
    else
	cameraAndFeature->camera->turnOff(cameraAndFeature->feature);
}

//! auto/manual ボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  あるカメラ機能を auto/manual モードにする．
  \param toggle		on/off ボタン
  \param userdata	CameraAndFeature (IEEE1394カメラと auto/manual
			モードにしたい機能の2ツ組)
*/
static void
CBsetAutoManual(GtkWidget* toggle, gpointer userdata)
{
    CameraAndFeature*	cameraAndFeature = (CameraAndFeature*)userdata;
    if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
	cameraAndFeature->camera->setAutoMode(cameraAndFeature->feature);
    else
	cameraAndFeature->camera->setManualMode(cameraAndFeature->feature);
}

//! adjustment widget が動かされると呼ばれるコールバック関数．
/*!
  あるカメラ機能の値を設定する．
  \param adj		設定値を与える adjuster
  \param userdata	CameraAndFeature (IEEE1394カメラと値を設定したい
			機能の2ツ組)
*/
static void
CBsetValue(GtkAdjustment* adj, gpointer userdata)
{
    CameraAndFeature*	cameraAndFeature = (CameraAndFeature*)userdata;
    cameraAndFeature->camera->setValue(cameraAndFeature->feature, adj->value);
}

//! U/B値用 adjustment widget が動かされると呼ばれるコールバック関数．
/*!
  ホワイトバランスのU/B値を設定する．
  \param adj		設定値を与える adjuster
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
*/
static void
CBsetWhiteBalanceUB(GtkAdjustment* adj, gpointer userdata)
{
    MyIIDCCamera*	camera = (MyIIDCCamera*)userdata;
    u_int		ub, vr;
    camera->getWhiteBalance(ub, vr);
    ub = u_int(adj->value);
    camera->setWhiteBalance(ub, vr);
}

//! V/R値用 adjustment widget が動かされると呼ばれるコールバック関数．
/*!
  ホワイトバランスのV/R値を設定する．
  \param adj		設定値を与える adjuster
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
*/
static void
CBsetWhiteBalanceVR(GtkAdjustment* adj, gpointer userdata)
{
    MyIIDCCamera*	camera = (MyIIDCCamera*)userdata;
    u_int		ub, vr;
    camera->getWhiteBalance(ub, vr);
    vr = u_int(adj->value);
    camera->setWhiteBalance(ub, vr);
}

/************************************************************************
*  global functions							*
************************************************************************/
//! カメラの各種機能に設定する値を指定するためのコマンド群を生成する．
/*!
  IEEE1394カメラがサポートしている機能を調べて生成するコマンドを決定する．
  \param camera		IEEE1394カメラ
  \return		生成されたコマンド群が貼りつけられたテーブル
*/
GtkWidget*
createCommands(MyIIDCCamera& camera)
{
    GtkWidget*	commands = gtk_table_new(4, 2 + NFEATURES, FALSE);
    u_int	y = 0;

    gtk_table_set_row_spacings(GTK_TABLE(commands), 2);
    gtk_table_set_col_spacings(GTK_TABLE(commands), 5);
    
  // カメラからの画像取り込みをon/offするtoggle buttonを生成．
    GtkWidget* toggle = gtk_toggle_button_new_with_label("Capture");
  // コールバック関数の登録．
    gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
		       GTK_SIGNAL_FUNC(CBcontinuousShot), &camera);
  // カメラの現在の画像取り込み状態をtoggle buttonに反映．
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
				 (camera.inContinuousShot() ? TRUE : FALSE));
    gtk_table_attach_defaults(GTK_TABLE(commands), toggle, 1, 2, y, y+1);
    ++y;

  // もしもカメラがトリガモードをサポートしていれば．．．
    if (camera.inquireFeatureFunction(IIDCCamera::TRIGGER_MODE) &
	IIDCCamera::Presence)
    {
      // カメラのtrigger modeをon/offするtoggle buttonを生成．
	toggle = gtk_toggle_button_new_with_label("Trigger mode");
      // コールバック関数の登録．
	gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
			   GTK_SIGNAL_FUNC(CBtriggerMode), &camera);
      // カメラの現在のtrigger modeをtoggle buttonに反映．
	gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
	     (camera.isTurnedOn(IIDCCamera::TRIGGER_MODE) ? TRUE : FALSE));
	gtk_table_attach_defaults(GTK_TABLE(commands), toggle, 1, 2, y, y+1);
	++y;
    }
    
    for (int i = 0; i < NFEATURES; ++i)
    {
	u_int	inq = camera.inquireFeatureFunction(feature[i].feature);
	if (inq & IIDCCamera::Presence)  // この機能が存在？
	{
	    u_int	x = 2;
	    
	    if (inq & IIDCCamera::OnOff)  // on/off操作が可能？
	    {
	      // on/offを切り替えるtoggle buttonを生成．
		GtkWidget* toggle = gtk_toggle_button_new_with_label("On");
	      // コールバック関数の登録．
		cameraAndFeature[i].camera = &camera;
		cameraAndFeature[i].feature = feature[i].feature;
		gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
				   GTK_SIGNAL_FUNC(CBturnOnOff),
				   (gpointer)&cameraAndFeature[i]);
	      // カメラの現在のon/off状態をtoggle buttonに反映．
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
		    (camera.isTurnedOn(feature[i].feature) ? TRUE : FALSE));
		gtk_table_attach_defaults(GTK_TABLE(commands), toggle,
					  x, x+1, y, y+1);
		++x;
	    }

	    if (inq & IIDCCamera::Manual)  // manual操作が可能？
	    {
		if (inq & IIDCCamera::Auto)  // 自動設定が可能？
		{
		  // manual/autoを切り替えるtoggle buttonを生成．
		    GtkWidget*	toggle
				  = gtk_toggle_button_new_with_label("Auto");
		  // コールバック関数の登録．
		    cameraAndFeature[i].camera = &camera;
		    cameraAndFeature[i].feature = feature[i].feature;
		    gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
				       GTK_SIGNAL_FUNC(CBsetAutoManual),
				       (gpointer)&cameraAndFeature[i]);
		  // カメラの現在のauto/manual状態をtoggle buttonに反映．
		    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
			(camera.isAuto(feature[i].feature) ? TRUE : FALSE));
		    gtk_table_attach_defaults(GTK_TABLE(commands), toggle,
					      x, x+1, y, y+1);
		}
		
		GtkWidget*	label = gtk_label_new(feature[i].name);
		gtk_table_attach_defaults(GTK_TABLE(commands), label,
					  0, 1, y, y+1);
		if (inq & IIDCCamera::ReadOut)
		{
		  // この機能が取り得る値の範囲を調べる．
		    u_int	min, max;
		    camera.getMinMax(feature[i].feature, min, max);
		    if (feature[i].feature == IIDCCamera::WHITE_BALANCE)
		    {
		      // white balanceの現在の値を調べる．
			u_int	ub, vr;
			camera.getWhiteBalance(ub, vr);
		      // white balanceのUB値を与えるためのadjustment widgetを生成．
			GtkObject*	adj = gtk_adjustment_new(ub, min, max,
								 1.0, 1.0, 0.0);
		      // コールバック関数の登録．
			gtk_signal_connect(GTK_OBJECT(adj), "value_changed",
					   GTK_SIGNAL_FUNC(CBsetWhiteBalanceUB),
					   &camera);
		      // adjustmentを操作するためのscale widgetを生成．
			GtkWidget*	scale = gtk_hscale_new(
						    GTK_ADJUSTMENT(adj));
			gtk_scale_set_digits(GTK_SCALE(scale), 0);
			gtk_widget_set_usize(GTK_WIDGET(scale), 200, 40);
			gtk_table_attach_defaults(GTK_TABLE(commands), scale,
						  1, 2, y, y+1);
			++i;
			++y;
			GtkWidget*	label = gtk_label_new(feature[i].name);
			gtk_table_attach_defaults(GTK_TABLE(commands), label,
						  0, 1, y, y+1);
		      // white balanceのVR値を与えるためのadjustment widgetを生成．
			adj = gtk_adjustment_new(vr, min, max, 1.0, 1.0, 0.0);
		      // コールバック関数の登録．
			gtk_signal_connect(GTK_OBJECT(adj), "value_changed",
					   GTK_SIGNAL_FUNC(CBsetWhiteBalanceVR),
					   &camera);
		      // adjustmentを操作するためのscale widgetを生成．
			scale = gtk_hscale_new(GTK_ADJUSTMENT(adj));
			gtk_scale_set_digits(GTK_SCALE(scale), 0);
			gtk_widget_set_usize(GTK_WIDGET(scale), 200, 40);
			gtk_table_attach_defaults(GTK_TABLE(commands), scale,
						  1, 2, y, y+1);
		    }
		    else
		    {
		      // この機能の現在の値を調べる．
			int	val = camera.getValue(feature[i].feature);
		      // この機能に値を与えるためのadjustment widgetを生成．
			GtkObject*
				adj = gtk_adjustment_new(val, min, max,
							 1.0, 1.0, 0.0);
		      // コールバック関数の登録．
			cameraAndFeature[i].camera = &camera;
			cameraAndFeature[i].feature = feature[i].feature;
			gtk_signal_connect(GTK_OBJECT(adj), "value_changed",
					   GTK_SIGNAL_FUNC(CBsetValue),
					   (gpointer)&cameraAndFeature[i]);
		      // adjustmentを操作するためのscale widgetを生成．
			GtkWidget*
				scale = gtk_hscale_new(GTK_ADJUSTMENT(adj));
			gtk_scale_set_digits(GTK_SCALE(scale), 0);
			gtk_widget_set_usize(GTK_WIDGET(scale), 200, 40);
			gtk_table_attach_defaults(GTK_TABLE(commands), scale,
						  1, 2, y, y+1);
		    }
		}
	    }

	    ++y;
	}
    }

    gtk_widget_show_all(commands);

    return commands;
}
 
}
