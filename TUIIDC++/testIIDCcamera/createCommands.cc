/*
 * testIIDCcamera: test program controlling an IIDC-based Digital Camera
 * Copyright (C) 2003 Toshio UESHIBA
 *   National Institute of Advanced Industrial Science and Technolog (AIST)
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
#include <cmath>		// for log10()
#include <algorithm>		// for std::max()

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
static const MyFeature	features[] =
{
    {IIDCCamera::TRIGGER_MODE,	"Trigger mode"},
    {IIDCCamera::BRIGHTNESS,	"Brightness"},
    {IIDCCamera::AUTO_EXPOSURE,	"Auto exposure"},
    {IIDCCamera::SHARPNESS,	"Sharpness"},
    {IIDCCamera::WHITE_BALANCE,	"White bal.(U/B)"},
    {IIDCCamera::HUE,		"Hue"},
    {IIDCCamera::SATURATION,	"Saturation"},
    {IIDCCamera::GAMMA,		"Gamma"},
    {IIDCCamera::SHUTTER,	"Shutter"},
    {IIDCCamera::GAIN,		"Gain"},
    {IIDCCamera::IRIS,		"Iris"},
    {IIDCCamera::FOCUS,		"Focus"},
    {IIDCCamera::TEMPERATURE,	"Temperature"},
    {IIDCCamera::TRIGGER_DELAY,	"Trigger delay"},
    {IIDCCamera::FRAME_RATE,	"Frame rate"},
    {IIDCCamera::ZOOM,		"Zoom"}
};
static const size_t	NFEATURES = sizeof(features) / sizeof(features[0]);

/*!
  カメラとその機能の2ツ組．3つのコールバック関数: CBsetActive(),
  CBsetAuto(), CBsetValue() の引数として渡される．
 */
struct CameraAndFeature
{
    IIDCCamera*			camera;		//!< カメラ
    IIDCCamera::Feature		feature;	//!< 操作したい機能
    GtkWidget*			scale;
};
static CameraAndFeature		cameraAndFeatures[NFEATURES];

/************************************************************************
*  static functions							*
************************************************************************/
static void
setScale(IIDCCamera& camera, IIDCCamera::Feature feature, GtkWidget* scale)
{
    using namespace	std;
    
    if (camera.isAbsControl(feature))
    {
	float	min, max;
	camera.getAbsMinMax(feature, min, max);
	gtk_range_set_range(GTK_RANGE(scale), min, max);
	auto	step = (max - min)/100;
	gtk_range_set_increments(GTK_RANGE(scale), step, step);
	auto	digits = std::max(3, int(-log10(step)) + 2);
	gtk_scale_set_digits(GTK_SCALE(scale), digits);
	auto	val = camera.getAbsValue(feature);
	gtk_range_set_value(GTK_RANGE(scale), val);
    }
    else
    {
	u_int	min, max;
	camera.getMinMax(feature, min, max);
	gtk_range_set_range(GTK_RANGE(scale), min, max);
	gtk_range_set_increments(GTK_RANGE(scale), 1, 1);
	gtk_scale_set_digits(GTK_SCALE(scale), 0);
	auto	val = camera.getValue(feature);
	gtk_range_set_value(GTK_RANGE(scale), val);
    }
}
    
/************************************************************************
*  callback functions							*
************************************************************************/
//! キャプチャボタンがonの間定期的に呼ばれるidle用コールバック関数．
/*!
  カメラから画像を取り込んでcanvasに表示する．
  \param userdata	MyIIDCCamera (IIDCカメラ)
  \return		TRUEを返す
*/
static gint
CBidle(gpointer userdata)
{
    auto	camera = static_cast<MyIIDCCamera*>(userdata);
    camera->idle();
    return TRUE;
}

//! キャプチャボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  timerを activate/deactivate する．
  \param toggle		キャプチャボタン
  \param userdata	MyIIDCCamera (IIDCカメラ)
*/
static void
CBcontinuousShot(GtkWidget* toggle, gpointer userdata)
{
    static gint	idleTag;
    auto	camera = static_cast<MyIIDCCamera*>(userdata);
    if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
    {
	idleTag = gtk_idle_add(CBidle, camera);	// idle処理を開始する．
	camera->continuousShot(true);	// カメラからの画像出力を開始する．
    }
    else
    {
	gtk_idle_remove(idleTag);	// idle処理を中止する．
	camera->continuousShot(false);	// カメラからの画像出力を停止する．
    }
}

//! トリガモードボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  trigger modeを on/off する．
  \param toggle		トリガモードボタン
  \param userdata	MyIIDCCamera (IIDCカメラ)
*/
static void
CBtriggerMode(GtkWidget* toggle, gpointer userdata)
{
    auto	camera = static_cast<MyIIDCCamera*>(userdata);
    camera->setActive(IIDCCamera::TRIGGER_MODE,
		      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)));
}

//! on/off ボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  あるカメラ機能を on/off する．
  \param toggle		on/off ボタン
  \param userdata	CameraAndFeature (IIDCカメラと on/off
			したい機能の2ツ組)
*/
static void
CBsetActive(GtkWidget* toggle, gpointer userdata)
{
    auto	cameraAndFeature = static_cast<CameraAndFeature*>(userdata);
    auto	camera  = cameraAndFeature->camera;
    auto	feature = cameraAndFeature->feature;
    camera->setActive(feature,
		      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)));
}

//! auto/manual ボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  あるカメラ機能を auto/manual モードにする．
  \param toggle		on/off ボタン
  \param userdata	CameraAndFeature (IIDCカメラと auto/manual
			モードにしたい機能の2ツ組)
*/
static void
CBsetAuto(GtkWidget* toggle, gpointer userdata)
{
    auto	cameraAndFeature = static_cast<CameraAndFeature*>(userdata);
    auto	camera  = cameraAndFeature->camera;
    auto	feature = cameraAndFeature->feature;
    camera->setAuto(feature,
		    gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)));
}

//! absolute/relative ボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  あるカメラ機能を absolute/relative モードにする．
  \param toggle		on/off ボタン
  \param userdata	CameraAndFeature (IIDCカメラと absolute/relative
			モードにしたい機能の2ツ組)
*/
static void
CBsetAbsControl(GtkWidget* toggle, gpointer userdata)
{
    auto	cameraAndFeature = static_cast<CameraAndFeature*>(userdata);
    auto	camera  = cameraAndFeature->camera;
    auto	feature = cameraAndFeature->feature;
    camera->setAbsControl(
	feature, gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)));
    setScale(*camera, feature, cameraAndFeature->scale);
}

//! adjustment widget が動かされると呼ばれるコールバック関数．
/*!
  あるカメラ機能の値を設定する．
  \param adj		設定値を与える adjuster
  \param userdata	CameraAndFeature (IIDCカメラと値を設定したい
			機能の2ツ組)
*/
static void
CBsetValue(GtkAdjustment* adj, gpointer userdata)
{
    auto	cameraAndFeature = static_cast<CameraAndFeature*>(userdata);
    auto	camera  = cameraAndFeature->camera;
    auto	feature = cameraAndFeature->feature;
    if (camera->isAbsControl(feature))
	camera->setAbsValue(feature, adj->value);
    else
	camera->setValue(feature, adj->value);
}

//! U/B値用 adjustment widget が動かされると呼ばれるコールバック関数．
/*!
  ホワイトバランスのU/B値を設定する．
  \param adj		設定値を与える adjuster
  \param userdata	MyIIDCCamera (IIDCカメラ)
*/
static void
CBsetWhiteBalanceUB(GtkAdjustment* adj, gpointer userdata)
{
    auto	camera = static_cast<MyIIDCCamera*>(userdata);
    u_int	ub, vr;
    camera->getWhiteBalance(ub, vr);
    ub = u_int(adj->value);
    camera->setWhiteBalance(ub, vr);
}

//! V/R値用 adjustment widget が動かされると呼ばれるコールバック関数．
/*!
  ホワイトバランスのV/R値を設定する．
  \param adj		設定値を与える adjuster
  \param userdata	MyIIDCCamera (IIDCカメラ)
*/
static void
CBsetWhiteBalanceVR(GtkAdjustment* adj, gpointer userdata)
{
    auto	camera = static_cast<MyIIDCCamera*>(userdata);
    u_int	ub, vr;
    camera->getWhiteBalance(ub, vr);
    vr = u_int(adj->value);
    camera->setWhiteBalance(ub, vr);
}

/************************************************************************
*  global functions							*
************************************************************************/
//! カメラの各種機能に設定する値を指定するためのコマンド群を生成する．
/*!
  IIDCカメラがサポートしている機能を調べて生成するコマンドを決定する．
  \param camera		IIDCカメラ
  \return		生成されたコマンド群が貼りつけられたテーブル
*/
GtkWidget*
createCommands(MyIIDCCamera& camera)
{
    const auto	commands = gtk_table_new(4, 2 + NFEATURES, FALSE);
    u_int	y = 0;

    gtk_table_set_row_spacings(GTK_TABLE(commands), 2);
    gtk_table_set_col_spacings(GTK_TABLE(commands), 5);
    
  // カメラからの画像取り込みをon/offするtoggle buttonを生成．
    auto	toggle = gtk_toggle_button_new_with_label("Capture");
  // コールバック関数の登録．
    gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
		       GTK_SIGNAL_FUNC(CBcontinuousShot), &camera);
  // カメラの現在の画像取り込み状態をtoggle buttonに反映．
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
				 (camera.inContinuousShot() ? TRUE : FALSE));
    gtk_table_attach_defaults(GTK_TABLE(commands), toggle, 1, 2, y, y+1);
    ++y;

    size_t	ncmds = 0;
    for (const auto& feature : features)
    {
	auto	inq = camera.inquireFeatureFunction(feature.feature);

	if (!((inq & IIDCCamera::Presence) &&
	      (inq & IIDCCamera::Manual)   &&
	      (inq & IIDCCamera::ReadOut)))
	    continue;
	
	switch (feature.feature)
	{
	  case IIDCCamera::TRIGGER_MODE:
	  // カメラのtrigger modeをon/offするtoggle buttonを生成．
	    toggle = gtk_toggle_button_new_with_label(feature.name);
	  // コールバック関数の登録．
	    gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
			       GTK_SIGNAL_FUNC(CBtriggerMode), &camera);
	  // カメラの現在のtrigger modeをtoggle buttonに反映．
	    gtk_toggle_button_set_active(
		GTK_TOGGLE_BUTTON(toggle),
		(camera.isActive(IIDCCamera::TRIGGER_MODE) ? TRUE : FALSE));
	    gtk_table_attach_defaults(GTK_TABLE(commands), toggle,
				      1, 2, y, y+1);
	    break;

	  case IIDCCamera::WHITE_BALANCE:
	  {
	    auto	label = gtk_label_new(feature.name);
	    gtk_table_attach_defaults(GTK_TABLE(commands), label, 0, 1, y, y+1);
	    
	    u_int	min, max;
	    camera.getMinMax(feature.feature, min, max);
	    u_int	ub, vr;
	    camera.getWhiteBalance(ub, vr);

	    auto	adj = gtk_adjustment_new(ub, min, max, 1, 1, 0);
	  // コールバック関数の登録．
	    gtk_signal_connect(GTK_OBJECT(adj), "value_changed",
			       GTK_SIGNAL_FUNC(CBsetWhiteBalanceUB),
			       &camera);
	  // adjustmentを操作するためのscale widgetを生成．
	    auto	scale = gtk_hscale_new(GTK_ADJUSTMENT(adj));
	    gtk_scale_set_digits(GTK_SCALE(scale), 0);
	    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 40);
	    gtk_table_attach_defaults(GTK_TABLE(commands), scale, 1, 2, y, y+1);
	    ++y;

	    label = gtk_label_new("White bal.(V/R)");
	    gtk_table_attach_defaults(GTK_TABLE(commands), label, 0, 1, y, y+1);
	    adj = gtk_adjustment_new(vr, min, max, 1, 1, 0);
	  // コールバック関数の登録．
	    gtk_signal_connect(GTK_OBJECT(adj), "value_changed",
			       GTK_SIGNAL_FUNC(CBsetWhiteBalanceVR), &camera);
	  // adjustmentを操作するためのscale widgetを生成．
	    scale = gtk_hscale_new(GTK_ADJUSTMENT(adj));
	    gtk_scale_set_digits(GTK_SCALE(scale), 0);
	    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 40);
	    gtk_table_attach_defaults(GTK_TABLE(commands), scale, 1, 2, y, y+1);
	  }
	    break;

	  default:
	  {
	    const auto	label = gtk_label_new(feature.name);
	    gtk_table_attach_defaults(GTK_TABLE(commands), label, 0, 1, y, y+1);

	    const auto	adj = gtk_adjustment_new(0, 0, 1, 1, 1, 0);
	    const auto	scale = gtk_hscale_new(GTK_ADJUSTMENT(adj));

	    setScale(camera, feature.feature, scale);

	  // コールバック関数の登録．
	    cameraAndFeatures[ncmds].camera  = &camera;
	    cameraAndFeatures[ncmds].feature = feature.feature;
	    cameraAndFeatures[ncmds].scale   = scale;
	    gtk_signal_connect(GTK_OBJECT(adj), "value_changed",
			       GTK_SIGNAL_FUNC(CBsetValue),
			       &cameraAndFeatures[ncmds]);

	    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 40);
	    gtk_table_attach_defaults(GTK_TABLE(commands), scale, 1, 2, y, y+1);

	    if (inq & IIDCCamera::Abs_Control)  // 絶対値での操作が可能？
	    {
	      // absolute/relativeを切り替えるtoggle buttonを生成．
		toggle = gtk_toggle_button_new_with_label("Abs");
	      // コールバック関数の登録．
		gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
				   GTK_SIGNAL_FUNC(CBsetAbsControl),
				   &cameraAndFeatures[ncmds]);
	      // カメラの現在のabsolute/relative状態をtoggle buttonに反映．
		gtk_toggle_button_set_active(
		    GTK_TOGGLE_BUTTON(toggle),
		    (camera.isAbsControl(feature.feature) ? TRUE : FALSE));
		gtk_table_attach_defaults(GTK_TABLE(commands),
					  toggle, 4, 5, y, y+1);
	    }

	    if (inq & IIDCCamera::OnOff)  // on/off操作が可能？
	    {
	      // on/offを切り替えるtoggle buttonを生成．
		toggle = gtk_toggle_button_new_with_label("On");
	      // コールバック関数の登録．
		gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
				   GTK_SIGNAL_FUNC(CBsetActive),
				   &cameraAndFeatures[ncmds]);
	      // カメラの現在のon/off状態をtoggle buttonに反映．
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
					     (camera.isActive(feature.feature) ?
					      TRUE : FALSE));
		gtk_table_attach_defaults(GTK_TABLE(commands),
					  toggle, 2, 3, y, y+1);
	    }

	    if (inq & IIDCCamera::Auto)  // 自動設定が可能？
	    {
	      // manual/autoを切り替えるtoggle buttonを生成．
		toggle = gtk_toggle_button_new_with_label("Auto");
	      // コールバック関数の登録．
		gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
				   GTK_SIGNAL_FUNC(CBsetAuto),
				   &cameraAndFeatures[ncmds]);
	      // カメラの現在のauto/manual状態をtoggle buttonに反映．
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
					     (camera.isAuto(feature.feature) ?
					      TRUE : FALSE));
		gtk_table_attach_defaults(GTK_TABLE(commands),
					  toggle, 3, 4, y, y+1);
	    }
	    
	    ++ncmds;
	  }
	    break;
	}
	    
	++y;
    }

    gtk_widget_show_all(commands);

    return commands;
}
 
}
