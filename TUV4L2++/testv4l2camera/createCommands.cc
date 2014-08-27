/*
 *  $Id: createCommands.cc,v 1.9 2012-08-29 19:35:49 ueshiba Exp $
 */
#include <list>
#include <boost/foreach.hpp>
#include "MyV4L2Camera.h"

namespace TU
{
/************************************************************************
*  local data								*
************************************************************************/
/*!
  カメラとその機能の2ツ組．コールバック関数: CBsetValue() の引数として渡される．
 */
struct CameraAndFeature
{
    V4L2Camera*		camera;		//!< カメラ
    V4L2Camera::Feature	feature;	//!< 操作したい機能
    int			val;
};
static std::list<CameraAndFeature>	cameraAndFeatures;
    
/************************************************************************
*  callback functions							*
************************************************************************/
//! キャプチャボタンがonの間定期的に呼ばれるidle用コールバック関数．
/*!
  カメラから画像を取り込んでcanvasに表示する．
  \param userdata	MyV4L2Camera (V4L2カメラ)
  \return		TRUEを返す
*/
static gint
CBidle(gpointer userdata)
{
    MyV4L2Camera*	camera = (MyV4L2Camera*)userdata;
    camera->idle();
    return TRUE;
}

//! キャプチャボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  timerを activate/deactivate する．
  \param toggle		キャプチャボタン
  \param userdata	MyV4L2Camera (V4L2カメラ)
*/
static void
CBcontinuousShot(GtkWidget* toggle, gpointer userdata)
{
    static gint		idleTag;
    MyV4L2Camera*	camera = (MyV4L2Camera*)userdata;
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

//! on/off ボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  あるカメラ機能を on/off する．
  \param toggle		on/off ボタン
  \param userdata	CameraAndFeature (V4L2カメラと on/off
			したい機能の2ツ組)
*/
static void
CBturnOnOff(GtkWidget* toggle, gpointer userdata)
{
    CameraAndFeature*	cameraAndFeature = (CameraAndFeature*)userdata;
    cameraAndFeature->camera->setValue(
	cameraAndFeature->feature,
	(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)) ? 1 : 0));
}

//! adjustment widget が動かされると呼ばれるコールバック関数．
/*!
  あるカメラ機能の値を設定する．
  \param adj		設定値を与える adjuster
  \param userdata	CameraAndFeature (V4L2カメラと値を設定したい
			機能の2ツ組)
*/
static void
CBsetValue(GtkAdjustment* adj, gpointer userdata)
{
    CameraAndFeature*	cameraAndFeature = (CameraAndFeature*)userdata;
    cameraAndFeature->camera->setValue(cameraAndFeature->feature, adj->value);
}

//! フォーマットとフレームレートを設定するためのコールバック関数．
/*!
  \param userdata	CameraAndFeature (カメラ, 設定すべき機能,
			設定すべき値の3ツ組)
*/
static void
CBmenuitem(GtkMenuItem*, gpointer userdata)
{
    CameraAndFeature*	cameraAndFeature = (CameraAndFeature*)userdata;
    cameraAndFeature->camera->setValue(cameraAndFeature->feature,
				       cameraAndFeature->val);
}

/************************************************************************
*  global functions							*
************************************************************************/
//! カメラの各種機能に設定する値を指定するためのコマンド群を生成する．
/*!
  V4L2カメラがサポートしている機能を調べて生成するコマンドを決定する．
  \param camera		V4L2カメラ
  \return		生成されたコマンド群が貼りつけられたテーブル
*/
GtkWidget*
createCommands(MyV4L2Camera& camera)
{
    V4L2Camera::FeatureRange	features = camera.availableFeatures();
    GtkWidget*			commands = gtk_table_new(
					     2,
					     1 + std::distance(features.first,
							       features.second),
					     FALSE);
    gtk_table_set_row_spacings(GTK_TABLE(commands), 2);
    gtk_table_set_col_spacings(GTK_TABLE(commands), 5);
    
  // カメラからの画像取り込みをon/offするtoggle buttonを生成．
    GtkWidget* toggle = gtk_toggle_button_new_with_label("Capture");

  // コールバック関数の登録．
    gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
		       GTK_SIGNAL_FUNC(CBcontinuousShot), &camera);

  // カメラの現在の画像取り込み状態をtoggle buttonに反映．
    size_t	y = 0;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
				 (camera.inContinuousShot() ? TRUE : FALSE));
    gtk_table_attach_defaults(GTK_TABLE(commands), toggle, 0, 2, y, y+1);
    ++y;

    BOOST_FOREACH (V4L2Camera::Feature feature, features)
    {
	V4L2Camera::MenuItemRange
	    menuItems = camera.availableMenuItems(feature);

	if (menuItems.first == menuItems.second)
	{
	    int	min, max, step;
	    camera.getMinMaxStep(feature, min, max, step);

	    if (min == 0 && max == 1)
	    {
	      // on/offを切り替えるtoggle buttonを生成．
		GtkWidget* toggle = gtk_toggle_button_new_with_label(
					camera.getName(feature).c_str());

	      // コールバック関数の登録．
		cameraAndFeatures.push_back(CameraAndFeature());
		CameraAndFeature&
		    cameraAndFeature = cameraAndFeatures.back();
		cameraAndFeature.camera  = &camera;
		cameraAndFeature.feature = feature;
		gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
				   GTK_SIGNAL_FUNC(CBturnOnOff),
				   (gpointer)&cameraAndFeature);

	      // カメラの現在のon/off状態をtoggle buttonに反映．
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
		    (camera.getValue(feature) ? TRUE : FALSE));
		gtk_table_attach_defaults(GTK_TABLE(commands), toggle,
					  0, 2, y, y+1);
	    }
	    else
	    {
		GtkWidget*	label = gtk_label_new(
					    camera.getName(feature).c_str());
		gtk_table_attach_defaults(GTK_TABLE(commands), label,
					  0, 1, y, y+1);
		
	      // この機能の現在の値を調べる．
		int		val = camera.getValue(feature);

	      // この機能に値を与えるためのadjustment widgetを生成．
		GtkObject*	adj = gtk_adjustment_new(val, min, max,
							 1.0, 1.0, 0.0);

	      // コールバック関数の登録．
		cameraAndFeatures.push_back(CameraAndFeature());
		CameraAndFeature&
		    cameraAndFeature = cameraAndFeatures.back();
		cameraAndFeature.camera  = &camera;
		cameraAndFeature.feature = feature;
		gtk_signal_connect(GTK_OBJECT(adj), "value_changed",
				   GTK_SIGNAL_FUNC(CBsetValue),
				   (gpointer)&cameraAndFeature);

	      // adjustmentを操作するためのscale widgetを生成．
		GtkWidget*	scale = gtk_hscale_new(GTK_ADJUSTMENT(adj));
		gtk_scale_set_digits(GTK_SCALE(scale), 0);
		gtk_widget_set_usize(GTK_WIDGET(scale), 200, 40);
		gtk_table_attach_defaults(GTK_TABLE(commands), scale,
					  1, 2, y, y+1);
	    }
	}
	else
	{
	    GtkWidget*	label = gtk_label_new(camera.getName(feature).c_str());
	    gtk_table_attach_defaults(GTK_TABLE(commands), label, 0, 1, y, y+1);
		
	    GtkWidget*	box   = gtk_vbox_new(TRUE, 0);
	    GtkWidget*	radioButton = 0;
	    BOOST_FOREACH (const V4L2Camera::MenuItem& menuItem, menuItems)
	    {
		if (radioButton == 0)
		{
		    radioButton = gtk_radio_button_new_with_label(
				      0, menuItem.name.c_str());
		    GtkWidget*	entry = gtk_entry_new();
		    gtk_container_add(GTK_CONTAINER(radioButton), entry);
		}
		else
		    radioButton = gtk_radio_button_new_with_label_from_widget(
				      GTK_RADIO_BUTTON(radioButton),
				      menuItem.name.c_str());
		gtk_box_pack_start(GTK_BOX(box), radioButton, TRUE, TRUE, 0);
		
		cameraAndFeatures.push_back(CameraAndFeature());
		CameraAndFeature&	cameraAndFeature
					    = cameraAndFeatures.back();
		cameraAndFeature.camera  = &camera;
		cameraAndFeature.feature = feature;
		cameraAndFeature.val	 = menuItem.index;
		gtk_signal_connect(GTK_OBJECT(radioButton), "activate",
				   GTK_SIGNAL_FUNC(CBmenuitem),
				   (gpointer)&cameraAndFeature);
	    }

	    gtk_table_attach_defaults(GTK_TABLE(commands), box, 1, 2, y, y+1);
	}

	++y;
    }
    
    gtk_widget_show_all(commands);

    return commands;
}
 
}
