/*
 *  $Id: createMenubar.cc,v 1.4 2002-12-18 04:34:08 ueshiba Exp $
 */
#include "My1394Camera.h"
#include "MyDialog.h"
#include <iomanip>

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
    const Ieee1394Camera::Format	format;		//!< 画像フォーマット
    const char* const			name;		//!< その名称
    
};
static const MyFormat	format[] =
{
    {Ieee1394Camera::YUV444_160x120,   "160x120-YUV(4:4:4)"},
    {Ieee1394Camera::YUV422_320x240,   "320x240-YUV(4:2:2)"},
    {Ieee1394Camera::YUV411_640x480,   "640x480-YUV(4:1:1)"},
    {Ieee1394Camera::YUV422_640x480,   "640x480-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_640x480,    "640x480-RGB"},
    {Ieee1394Camera::MONO8_640x480,    "640x480-Y(mono)"},
    {Ieee1394Camera::MONO16_640x480,   "640x480-Y(mono16)"},
    {Ieee1394Camera::YUV422_800x600,   "800x600-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_800x600,    "800x600-RGB"},
    {Ieee1394Camera::MONO8_800x600,    "800x600-Y(mono)"},
    {Ieee1394Camera::YUV422_1024x768,  "1024x768-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_1024x768,   "1024x768-RGB"},
    {Ieee1394Camera::MONO8_1024x768,   "1024x768-Y(mono)"},
    {Ieee1394Camera::MONO16_800x600,   "800x600-Y(mono16)"},
    {Ieee1394Camera::MONO16_1024x768,  "1024x768-Y(mono16)"},
    {Ieee1394Camera::YUV422_1280x960,  "1280x960-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_1280x960,   "1280x960-RGB"},
    {Ieee1394Camera::MONO8_1280x960,   "1280x960-Y(mono)"},
    {Ieee1394Camera::YUV422_1600x1200, "1600x1200-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_1600x1200,  "1600x1200-RGB"},
    {Ieee1394Camera::MONO8_1600x1200,  "1600x1200-Y(mono)"},
    {Ieee1394Camera::MONO16_1280x960,  "1280x960-Y(mono16)"},
    {Ieee1394Camera::MONO16_1600x1200, "1600x1200-Y(mono16)"},
    {Ieee1394Camera::Format_7_0,       "Format_7_0"},
    {Ieee1394Camera::Format_7_1,       "Format_7_1"},
    {Ieee1394Camera::Format_7_2,       "Format_7_2"},
    {Ieee1394Camera::Format_7_3,       "Format_7_3"},
    {Ieee1394Camera::Format_7_4,       "Format_7_4"},
    {Ieee1394Camera::Format_7_5,       "Format_7_5"},
    {Ieee1394Camera::Format_7_6,       "Format_7_6"},
    {Ieee1394Camera::Format_7_6,       "Format_7_7"}
};
static const int	NFORMATS = sizeof(format)/sizeof(format[0]);

/*!
  カメラがサポートするフレームレートとその名称．
*/
struct MyFrameRate
{
    const Ieee1394Camera::FrameRate	frameRate;	//!< フレームレート
    const char* const			name;		//!< その名称
};
static const MyFrameRate	frameRate[] =
{
    {Ieee1394Camera::FrameRate_1_875, "1.875fps"},
    {Ieee1394Camera::FrameRate_3_75,  "3.75fps"},
    {Ieee1394Camera::FrameRate_7_5,   "7.5fps"},
    {Ieee1394Camera::FrameRate_15,    "15fps"},
    {Ieee1394Camera::FrameRate_30,    "30fps"},
    {Ieee1394Camera::FrameRate_60,    "60fps"},
    {Ieee1394Camera::FrameRate_x,     "custom"}
};
static const int	NRATES=sizeof(frameRate)/sizeof(frameRate[0]);

/*!
  カメラ, 画像フォーマット, フレームレートの3ツ組．コールバック関数:
  CBmenuitem() の引数として渡される．
 */
struct FormatAndFrameRate
{
    My1394Camera*		camera;
    Ieee1394Camera::Format	format;
    Ieee1394Camera::FrameRate	frameRate;
};
static FormatAndFrameRate	fmtAndFRate[NFORMATS * NRATES];

/************************************************************************
*  static functions							*
************************************************************************/
static ostream&
operator <<(ostream& out, const Ieee1394Camera& camera)
{
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
	Ieee1394Camera::Feature	feature;
	const char*			name;
    } feature[] =
    {
	{Ieee1394Camera::BRIGHTNESS,	"BRIGHTNESS"},
	{Ieee1394Camera::AUTO_EXPOSURE, "AUTO_EXPOSURE"},
	{Ieee1394Camera::SHARPNESS,	"SHARPNESS"},
	{Ieee1394Camera::WHITE_BALANCE, "WHITE_BALANCE"},
	{Ieee1394Camera::HUE,		"HUE"},
	{Ieee1394Camera::SATURATION,	"SATURATION"},
	{Ieee1394Camera::GAMMA,		"GAMMA"},
	{Ieee1394Camera::SHUTTER,	"SHUTTER"},
	{Ieee1394Camera::GAIN,		"GAIN"},
	{Ieee1394Camera::IRIS,		"IRIS"},
	{Ieee1394Camera::FOCUS,		"FOCUS"},
	{Ieee1394Camera::TEMPERATURE,	"TEMPERATURE"},
	{Ieee1394Camera::ZOOM,		"ZOOM"},
	{Ieee1394Camera::PAN,		"PAN"},
	{Ieee1394Camera::TILT,		"TILT"}
    };
    const int	nfeatures = sizeof(feature) / sizeof(feature[0]);
		    
    for (int i = 0; i < nfeatures; ++i)
    {
	u_int	inq = camera.inquireFeatureFunction(feature[i].feature);
	if ((inq & Ieee1394Camera::Presence) &&
	    (inq & Ieee1394Camera::Manual))
	{
	    out << ' ' << feature[i].name << ' ';

	    if ((inq & Ieee1394Camera::Auto) && 
		camera.isAuto(feature[i].feature))
		out << -1;
	    else if (feature[i].feature == Ieee1394Camera::WHITE_BALANCE)
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
    if (fmtAndFRate->format >= Ieee1394Camera::Format_7_0)
    {
	MyDialog	dialog(fmtAndFRate->camera->
			       getFormat_7_Info(fmtAndFRate->format));
	u_int		u0, v0, width, height;
	dialog.getROI(u0, v0, width, height);
	fmtAndFRate->camera->setFormat_7_ROI(fmtAndFRate->format,
					     u0, v0, width, height);
    }
    fmtAndFRate->camera->setFormatAndFrameRate(fmtAndFRate->format,
					       fmtAndFRate->frameRate);
}

//! カメラの設定値を標準出力に書き出して終了するためのコールバック関数．
/*!
  \param userdata	My1394Camera (IEEE1394カメラ)
*/
static void
CBexit(GtkMenuItem*, gpointer userdata)
{
    My1394Camera*	camera = (My1394Camera*)userdata;
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
createMenubar(My1394Camera& camera)
{
    GtkWidget*	menubar	= gtk_menu_bar_new();

  // "File"メニューを生成．
    GtkWidget*	menu = gtk_menu_new();
    GtkWidget*	item = gtk_menu_item_new_with_label("Quit");
    gtk_signal_connect(GTK_OBJECT(item), "activate",
		       GTK_SIGNAL_FUNC(CBexit), &camera);
    gtk_menu_append(GTK_MENU(menu), item);
    item = gtk_menu_item_new_with_label("File");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), menu);
    gtk_menu_bar_append(GTK_MENU_BAR(menubar), item);

  // "Format"メニューを生成．
    menu = gtk_menu_new();
  // 現在指定されている画像フォーマットおよびフレームレートを調べる．
    Ieee1394Camera::Format	current_format = camera.getFormat();
    Ieee1394Camera::FrameRate	current_rate   = camera.getFrameRate();
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
