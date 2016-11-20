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
  カメラとそのトリガモードの2ツ組．
*/
struct CameraAndTriggerMode
{
    IIDCCamera*				camera;		//!< カメラ
    const IIDCCamera::TriggerModeName*	mode;		//!< トリガモード
    GtkWidget*				button;
};
static CameraAndTriggerMode	cameraAndTriggerModes[IIDCCamera::NTRIGGERMODES];
    
/*!
  カメラとその機能およびそれを操作するscaleウィジェットの3ツ組．
  コールバック関数: CBsetActive(), CBsetAuto(), CBsetAbsControl(), CBsetValue()
  の引数として渡される．
 */
struct CameraAndFeature
{
    IIDCCamera*			camera;		//!< カメラ
    IIDCCamera::Feature		feature;	//!< 操作したい機能
    GtkWidget*			scale;
    GtkWidget*			scale2;
};
static CameraAndFeature		cameraAndFeatures[IIDCCamera::NFEATURES];

/************************************************************************
*  static functions							*
************************************************************************/
static void
setScale(const CameraAndFeature& cameraAndFeature)
{
    using namespace	std;

    const auto	camera  = cameraAndFeature.camera;
    const auto	feature = cameraAndFeature.feature;
    const auto	scale   = cameraAndFeature.scale;
    const auto	scale2  = cameraAndFeature.scale2;
    
    if (camera->isAbsControl(feature))
    {
	float	min, max;
	camera->getMinMax(feature, min, max);
	gtk_range_set_range(GTK_RANGE(scale), min, max);
	const auto	step = (max - min)/100;
	gtk_range_set_increments(GTK_RANGE(scale), step, step);
	const auto	digits = std::max(3, int(-log10(step)) + 2);
	gtk_scale_set_digits(GTK_SCALE(scale), digits);

	if (feature == IIDCCamera::WHITE_BALANCE)
	{
	    gtk_range_set_range(GTK_RANGE(scale2), min, max);
	    gtk_range_set_increments(GTK_RANGE(scale2), step, step);
	    gtk_scale_set_digits(GTK_SCALE(scale2), digits);

	    float	ub, vr;
	    camera->getWhiteBalance(ub, vr);
	    gtk_range_set_value(GTK_RANGE(scale),  ub);
	    gtk_range_set_value(GTK_RANGE(scale2), vr);
	}
	else
	    gtk_range_set_value(GTK_RANGE(scale), camera->getValue<float>(feature));
    }
    else
    {
	u_int	min, max;
	camera->getMinMax(feature, min, max);
	gtk_range_set_range(GTK_RANGE(scale), min, max);
	gtk_range_set_increments(GTK_RANGE(scale), 1, 1);
	gtk_scale_set_digits(GTK_SCALE(scale), 0);

	if (feature == IIDCCamera::WHITE_BALANCE)
	{
	    gtk_range_set_range(GTK_RANGE(scale2), min, max);
	    gtk_range_set_increments(GTK_RANGE(scale2), 1, 1);
	    gtk_scale_set_digits(GTK_SCALE(scale2), 0);

	    u_int	ub, vr;
	    camera->getWhiteBalance(ub, vr);
	    gtk_range_set_value(GTK_RANGE(scale),  ub);
	    gtk_range_set_value(GTK_RANGE(scale2), vr);
	}
	else
	    gtk_range_set_value(GTK_RANGE(scale), camera->getValue<u_int>(feature));
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
    const auto	camera = static_cast<MyIIDCCamera*>(userdata);
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
    const auto	camera = static_cast<MyIIDCCamera*>(userdata);
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

//! メニューボタンが押されるとメニューをポップアップするコールバック関数．
/*!
  \param menu		ポップアップするメニュー
  \param event		ボタンが押された時のイベント
*/
static gboolean
CBbuttonPress(GtkWidget* menu, GdkEvent* event)
{
    if (event->type == GDK_BUTTON_PRESS)
    {
	const auto bevent = reinterpret_cast<GdkEventButton*>(event);
	gtk_menu_popup(GTK_MENU(menu), NULL, NULL, NULL, NULL,
		       bevent->button, bevent->time);
	return TRUE;
    }
    else
	return FALSE;
}
    
//! トリガモード選択のメニューボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  トリガモードを選択する．
  \param item		メニューアイテム
  \param userdata	CameraAndTriggerMode (IIDCカメラとトリガモードの2ツ組)
*/
static void
CBsetTriggerMode(GtkWidget* item, gpointer userdata)
{
    const auto	cameraAndTriggerMode
		    = static_cast<const CameraAndTriggerMode*>(userdata);
    const auto	camera = cameraAndTriggerMode->camera;
    const auto	mode   = cameraAndTriggerMode->mode;
    const auto	button = cameraAndTriggerMode->button;
    camera->setTriggerMode(mode->mode);
    gtk_button_set_label(GTK_BUTTON(button), mode->name);
}
    
//! トリガ極性選択ボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  トリガ極性を選択する．
  \param item		+/- ボタン
  \param userdata	IIDCカメラ
*/
static void
CBsetTriggerPolarity(GtkWidget* toggle, gpointer userdata)
{
    const auto	camera = static_cast<IIDCCamera*>(userdata);
    camera->setTriggerPolarity(
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
    const auto	cameraAndFeature = static_cast<const CameraAndFeature*>(userdata);
    const auto	camera  = cameraAndFeature->camera;
    const auto	feature = cameraAndFeature->feature;
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
    const auto	cameraAndFeature = static_cast<const CameraAndFeature*>(userdata);
    const auto	camera  = cameraAndFeature->camera;
    const auto	feature = cameraAndFeature->feature;
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
    const auto	cameraAndFeature = static_cast<const CameraAndFeature*>(userdata);
    const auto	camera  = cameraAndFeature->camera;
    const auto	feature = cameraAndFeature->feature;
    camera->setAbsControl(feature,
			  gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)));
    setScale(*cameraAndFeature);
}

//! scale widget が動かされると呼ばれるコールバック関数．
/*!
  あるカメラ機能の値を設定する．
  \param scale		設定値を与える scale
  \param userdata	CameraAndFeature (IIDCカメラと値を設定したい
			機能の2ツ組)
*/
static void
CBsetValue(GtkScale* scale, gpointer userdata)
{
    const auto	cameraAndFeature = static_cast<const CameraAndFeature*>(userdata);
    const auto	camera  = cameraAndFeature->camera;
    const auto	feature = cameraAndFeature->feature;
    if (feature == IIDCCamera::WHITE_BALANCE)
	if (camera->isAbsControl(feature))
	    camera->setWhiteBalance(float(gtk_range_get_value(
					      GTK_RANGE(cameraAndFeature->scale))),
				    float(gtk_range_get_value(
					      GTK_RANGE(cameraAndFeature->scale2))));
	else
	    camera->setWhiteBalance(u_int(gtk_range_get_value(
					      GTK_RANGE(cameraAndFeature->scale))),
				    u_int(gtk_range_get_value(
					      GTK_RANGE(cameraAndFeature->scale2))));
    else
	if (camera->isAbsControl(feature))
	    camera->setValue(feature, float(gtk_range_get_value(GTK_RANGE(scale))));
	else
	    camera->setValue(feature, u_int(gtk_range_get_value(GTK_RANGE(scale))));
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
    const auto	commands = gtk_table_new(4, 2 + IIDCCamera::NFEATURES, FALSE);
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
    for (const auto& feature : IIDCCamera::featureNames)
    {
	const auto	inq = camera.inquireFeatureFunction(feature.feature);

	if (!((inq & IIDCCamera::Presence) &&
	      (inq & IIDCCamera::Manual)   &&
	      (inq & IIDCCamera::ReadOut)))
	    continue;
	
	cameraAndFeatures[ncmds].camera  = &camera;
	cameraAndFeatures[ncmds].feature = feature.feature;

	const auto	label = gtk_label_new(feature.name);
	gtk_table_attach_defaults(GTK_TABLE(commands), label, 0, 1, y, y+1);

	switch (feature.feature)
	{
	  case IIDCCamera::TRIGGER_MODE:
	  {
	  // カメラのtrigger modeをon/offするtoggle buttonを生成．
	    const auto	button = gtk_button_new();
	    const auto	menu   = gtk_menu_new();
	    size_t	nmodes = 0;
	    for (const auto& triggerMode : IIDCCamera::triggerModeNames)
		if (inq & triggerMode.mode)
		{
		    if (camera.getTriggerMode() == triggerMode.mode)
			gtk_button_set_label(GTK_BUTTON(button), triggerMode.name);
		    const auto
			item = gtk_menu_item_new_with_label(triggerMode.name);
		    gtk_menu_append(GTK_MENU(menu), item);
		  // コールバック関数の登録．
		    cameraAndTriggerModes[nmodes].camera = &camera;
		    cameraAndTriggerModes[nmodes].mode   = &triggerMode;
		    cameraAndTriggerModes[nmodes].button = button;
		    gtk_signal_connect(GTK_OBJECT(item), "activate",
				       GTK_SIGNAL_FUNC(CBsetTriggerMode),
				       &cameraAndTriggerModes[nmodes]);
		    gtk_widget_show(item);
		    ++nmodes;
		}
	  // ポップアップのためのコールバック関数の登録．
	    g_signal_connect_swapped(button, "event",
				     G_CALLBACK(CBbuttonPress), menu);
	    gtk_table_attach_defaults(GTK_TABLE(commands), button, 1, 2, y, y+1);
	  }
	    break;

	  case IIDCCamera::WHITE_BALANCE:
	  {
	    const auto	scale  = gtk_hscale_new_with_range(0, 1, 1);
	    const auto	scale2 = gtk_hscale_new_with_range(0, 1, 1);

	    cameraAndFeatures[ncmds].scale  = scale;
	    cameraAndFeatures[ncmds].scale2 = scale2;
	    setScale(cameraAndFeatures[ncmds]);
	    
	  // コールバック関数の登録．
	    gtk_signal_connect(GTK_OBJECT(scale), "value_changed",
			       GTK_SIGNAL_FUNC(CBsetValue),
			       &cameraAndFeatures[ncmds]);
	    gtk_signal_connect(GTK_OBJECT(scale2), "value_changed",
			       GTK_SIGNAL_FUNC(CBsetValue),
			       &cameraAndFeatures[ncmds]);

	    gtk_widget_set_usize(GTK_WIDGET(scale),  200, 40);
	    gtk_widget_set_usize(GTK_WIDGET(scale2), 200, 40);
	    gtk_table_attach_defaults(GTK_TABLE(commands), scale, 1, 2, y, y+1);
	    ++y;
	    const auto	label2 = gtk_label_new("White bal.(V/R)");
	    gtk_table_attach_defaults(GTK_TABLE(commands), label2, 0, 1, y, y+1);
	    gtk_table_attach_defaults(GTK_TABLE(commands), scale2, 1, 2, y, y+1);
	  }
	    break;

	  default:
	  {
	    const auto	scale = gtk_hscale_new_with_range(0, 1, 1);

	    cameraAndFeatures[ncmds].scale = scale;
	    setScale(cameraAndFeatures[ncmds]);

	  // コールバック関数の登録．
	    gtk_signal_connect(GTK_OBJECT(scale), "value_changed",
			       GTK_SIGNAL_FUNC(CBsetValue),
			       &cameraAndFeatures[ncmds]);

	    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 40);
	    gtk_table_attach_defaults(GTK_TABLE(commands), scale, 1, 2, y, y+1);
	  }
	    break;
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
	    gtk_table_attach_defaults(GTK_TABLE(commands), toggle, 2, 3, y, y+1);
	}

	if (inq & IIDCCamera::Auto)  // 自動設定が可能？
	{
	    if (feature.feature == IIDCCamera::TRIGGER_MODE)
	    {
	      // manual/autoを切り替えるtoggle buttonを生成．
		toggle = gtk_toggle_button_new_with_label("(+)");
	      // コールバック関数の登録．
		gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
				   GTK_SIGNAL_FUNC(CBsetTriggerPolarity), &camera);
	      // カメラの現在のauto/manual状態をtoggle buttonに反映．
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
					     (camera.getTriggerPolarity() ?
					      TRUE : FALSE));
	    }
	    else
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
	    }
	    gtk_table_attach_defaults(GTK_TABLE(commands), toggle, 3, 4, y, y+1);
	}
	    
	if (inq & IIDCCamera::Abs_Control)  // 絶対値での操作が可能？
	{
	  // absolute/relativeを切り替えるtoggle buttonを生成．
	    toggle = gtk_toggle_button_new_with_label("Abs");
	  // コールバック関数の登録．
	    gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
			       GTK_SIGNAL_FUNC(CBsetAbsControl),
			       &cameraAndFeatures[ncmds]);
	  // カメラの現在のabsolute/relative状態をtoggle buttonに反映．
	    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
					 (camera.isAbsControl(feature.feature) ?
					  TRUE : FALSE));
	    gtk_table_attach_defaults(GTK_TABLE(commands), toggle, 4, 5, y, y+1);
	}
	
	++ncmds;
	++y;
    }

    gtk_widget_show_all(commands);

    return commands;
}
 
}
