/*
 *  $Id: createCommands.cc,v 1.4 2003-02-20 05:51:50 ueshiba Exp $
 */
#include "My1394Camera.h"

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
    const Ieee1394Camera::Feature	feature;	//!< カメラの機能
    const char* const			name;		//!< その名称
};
static const MyFeature	feature[] =
{
    {Ieee1394Camera::BRIGHTNESS,	"Brightness:"},
    {Ieee1394Camera::AUTO_EXPOSURE,	"Auto exposure:"},
    {Ieee1394Camera::SHARPNESS,		"Sharpness:"},
    {Ieee1394Camera::WHITE_BALANCE,	"White bal.(U/B):"},
    {Ieee1394Camera::WHITE_BALANCE,	"White bal.(V/R):"},
    {Ieee1394Camera::HUE,		"Hue:"},
    {Ieee1394Camera::SATURATION,	"Saturation:"},
    {Ieee1394Camera::GAMMA,		"Gamma:"},
    {Ieee1394Camera::SHUTTER,		"Shutter:"},
    {Ieee1394Camera::GAIN,		"Gain:"},
    {Ieee1394Camera::IRIS,		"Iris:"},
    {Ieee1394Camera::FOCUS,		"Focus:"},
    {Ieee1394Camera::TEMPERATURE,	"Temperature:"},
    {Ieee1394Camera::ZOOM,		"Zoom:"}
};
static const int		NFEATURES = sizeof(feature)/sizeof(feature[0]);

/*!
  カメラとその機能の2ツ組．3つのコールバック関数: CBturnOnOff(),
  CBsetAutoManual(), CBsetValue() の引数として渡される．
 */
struct CameraAndFeature
{
    Ieee1394Camera*		camera;		//!< カメラ
    Ieee1394Camera::Feature	feature;	//!< 操作したい機能
};
static CameraAndFeature		cameraAndFeature[NFEATURES];

/*!
  カメラと Bayer -> RGB 変換の種類の2ツ組．コールバック関数: CBbayerToRGB()
  の引数として渡される．
 */
struct CameraAndBayer
{
    My1394Camera*		camera;		//!< カメラ
    My1394Camera::Bayer		bayer;		//!< Bayer -> RGB 変換の種類
};
static CameraAndBayer		cameraAndBayer[3];

/************************************************************************
*  callback functions							*
************************************************************************/
//! キャプチャボタンがonの間定期的に呼ばれるidle用コールバック関数．
/*!
  カメラから画像を取り込んでcanvasに表示する．
  \param userdata	My1394Camera (IEEE1394カメラ)
  \return		TRUEを返す
*/
static gint
CBidle(gpointer userdata)
{
    My1394Camera*	camera = (My1394Camera*)userdata;
    camera->idle();
    return TRUE;
}

//! キャプチャボタンの状態が変更されると呼ばれるコールバック関数．
/*!
  timerを activate/deactivate する．
  \param toggle		キャプチャボタン
  \param userdata	My1394Camera (IEEE1394カメラ)
*/
static void
CBcontinuousShot(GtkWidget* toggle, gpointer userdata)
{
    static gint		idleTag;
    My1394Camera*	camera = (My1394Camera*)userdata;
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
  \param userdata	My1394Camera (IEEE1394カメラ)
*/
static void
CBtriggerMode(GtkWidget* toggle, gpointer userdata)
{
    My1394Camera*	camera = (My1394Camera*)userdata;
    if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle)))
	camera->turnOn(Ieee1394Camera::TRIGGER_MODE)
	    .setTriggerMode(Ieee1394Camera::Trigger_Mode0);
    else
	camera->turnOff(Ieee1394Camera::TRIGGER_MODE);
}

//! Bayer -> RGB変換ボタンがクリックされると呼ばれるコールバック関数．
/*!
  Bayer pattern -> RGB変換を設定する．
  \param userdata	CameraAndBayer (IEEE1394カメラと設定したい Bayer
			-> RGB 変換の種類の2ツ組)
*/
static void
CBbayerToRGB(GtkWidget*, gpointer userdata)
{
    CameraAndBayer*	cameraAndBayer = (CameraAndBayer*)userdata;
    cameraAndBayer->camera->setBayer(cameraAndBayer->bayer);
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
  \param userdata	My1394Camera (IEEE1394カメラ)
*/
static void
CBsetWhiteBalanceUB(GtkAdjustment* adj, gpointer userdata)
{
    My1394Camera*	camera = (My1394Camera*)userdata;
    u_int		ub, vr;
    camera->getWhiteBalance(ub, vr);
    ub = u_int(adj->value);
    camera->setWhiteBalance(ub, vr);
}

//! V/R値用 adjustment widget が動かされると呼ばれるコールバック関数．
/*!
  ホワイトバランスのV/R値を設定する．
  \param adj		設定値を与える adjuster
  \param userdata	My1394Camera (IEEE1394カメラ)
*/
static void
CBsetWhiteBalanceVR(GtkAdjustment* adj, gpointer userdata)
{
    My1394Camera*	camera = (My1394Camera*)userdata;
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
createCommands(My1394Camera& camera)
{
    GtkWidget*	commands = gtk_table_new(4, 3 + NFEATURES, FALSE);
    u_int	y = 0;

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
    if (camera.inquireFeatureFunction(Ieee1394Camera::TRIGGER_MODE) &
	Ieee1394Camera::Presence)
    {
      // カメラのtrigger modeをon/offするtoggle buttonを生成．
	toggle = gtk_toggle_button_new_with_label("Trigger mode");
      // コールバック関数の登録．
	gtk_signal_connect(GTK_OBJECT(toggle), "toggled",
			   GTK_SIGNAL_FUNC(CBtriggerMode), &camera);
      // カメラの現在のtrigger modeをtoggle buttonに反映．
	gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
	     (camera.isTurnedOn(Ieee1394Camera::TRIGGER_MODE) ? TRUE : FALSE));
	gtk_table_attach_defaults(GTK_TABLE(commands), toggle, 1, 2, y, y+1);
	++y;
    }
    
  // Bayer pattern -> RGB変換の種類を設定するradio buttonを生成．
    GtkWidget*	box = gtk_hbox_new(FALSE, 10);
    gtk_container_set_border_width(GTK_CONTAINER(box), 10);
    GtkWidget*	button = gtk_radio_button_new_with_label(NULL, "None");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(button), TRUE);
    cameraAndBayer[0].camera = &camera;
    cameraAndBayer[0].bayer  = My1394Camera::NONE;
    gtk_signal_connect(GTK_OBJECT(button), "clicked",
		       GTK_SIGNAL_FUNC(CBbayerToRGB), &cameraAndBayer[0]);
    gtk_box_pack_start(GTK_BOX(box), button, TRUE, TRUE, 0);
    button = gtk_radio_button_new_with_label(
		 gtk_radio_button_group(GTK_RADIO_BUTTON(button)), "RGGB");
    cameraAndBayer[1].camera = &camera;
    cameraAndBayer[1].bayer  = My1394Camera::RGGB;
    gtk_signal_connect(GTK_OBJECT(button), "clicked",
		       GTK_SIGNAL_FUNC(CBbayerToRGB), &cameraAndBayer[1]);
    gtk_box_pack_start(GTK_BOX(box), button, TRUE, TRUE, 0);
    button = gtk_radio_button_new_with_label(
		 gtk_radio_button_group(GTK_RADIO_BUTTON(button)), "BGGR");
    cameraAndBayer[2].camera = &camera;
    cameraAndBayer[2].bayer  = My1394Camera::BGGR;
    gtk_signal_connect(GTK_OBJECT(button), "clicked",
		       GTK_SIGNAL_FUNC(CBbayerToRGB), &cameraAndBayer[2]);
    gtk_box_pack_start(GTK_BOX(box), button, TRUE, TRUE, 0);
    gtk_table_attach_defaults(GTK_TABLE(commands), box, 1, 2, y, y+1);
    ++y;

    for (int i = 0; i < NFEATURES; ++i)
    {
	u_int	inq = camera.inquireFeatureFunction(feature[i].feature);
	if (inq & Ieee1394Camera::Presence)  // この機能が存在？
	{
	    u_int	x = 2;
	    
	    if (inq & Ieee1394Camera::OnOff)  // on/off操作が可能？
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

	    if (inq & Ieee1394Camera::Manual)  // manual操作が可能？
	    {
		if (inq & Ieee1394Camera::Auto)  // 自動設定が可能？
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
	      // この機能が取り得る値の範囲を調べる．
		u_int	min, max;
		camera.getMinMax(feature[i].feature, min, max);
		if (feature[i].feature == Ieee1394Camera::WHITE_BALANCE)
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
		    GtkWidget*	scale = gtk_hscale_new(GTK_ADJUSTMENT(adj));
		    gtk_scale_set_digits(GTK_SCALE(scale), 0);
		    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 30);
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
		    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 30);
		    gtk_table_attach_defaults(GTK_TABLE(commands), scale,
					      1, 2, y, y+1);
		}
		else
		{
		  // この機能の現在の値を調べる．
		    int		val = camera.getValue(feature[i].feature);
		  // この機能に値を与えるためのadjustment widgetを生成．
		    GtkObject*	adj = gtk_adjustment_new(val, min, max,
							 1.0, 1.0, 0.0);
		  // コールバック関数の登録．
		    cameraAndFeature[i].camera = &camera;
		    cameraAndFeature[i].feature = feature[i].feature;
		    gtk_signal_connect(GTK_OBJECT(adj), "value_changed",
				       GTK_SIGNAL_FUNC(CBsetValue),
				       (gpointer)&cameraAndFeature[i]);
		  // adjustmentを操作するためのscale widgetを生成．
		    GtkWidget*	scale = gtk_hscale_new(GTK_ADJUSTMENT(adj));
		    gtk_scale_set_digits(GTK_SCALE(scale), 0);
		    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 30);
		    gtk_table_attach_defaults(GTK_TABLE(commands), scale,
					      1, 2, y, y+1);
		}
	    }

	    ++y;
	}
    }

    gtk_widget_show_all(commands);

    return commands;
}
 
}
