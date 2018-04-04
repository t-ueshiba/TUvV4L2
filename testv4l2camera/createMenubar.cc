/*
 *  $Id: createMenubar.cc,v 1.11 2012-08-29 19:35:49 ueshiba Exp $
 */
#include <list>
#include <boost/foreach.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include "MyV4L2Camera.h"
#include "MyDialog.h"

#define DEFAULT_IMAGE_FILE_NAME	"testv4l2camera.pbm"

namespace TU
{
/************************************************************************
*  local data								*
************************************************************************/
struct CameraAndFormat
{
    MyV4L2Camera*		camera;
    V4L2Camera::PixelFormat	pixelFormat;
    size_t			width, height;
    u_int			fps_n, fps_d;
};
static std::list<CameraAndFormat>	cameraAndFormats;

struct CameraAndFileSelection
{
    MyV4L2Camera*		camera;
    GtkWidget*			filesel;
};
static CameraAndFileSelection		cameraAndFileSelection;

/************************************************************************
*  callback functions							*
************************************************************************/
//! フォーマットとフレームレートを設定するためのコールバック関数．
/*!
  \param userdata	CameraAndFormat (カメラ, 設定すべきフォーマット,
			設定すべきフレームレートの3ツ組)
*/
static void
CBmenuitem(GtkMenuItem*, gpointer userdata)
{
    CameraAndFormat*	camAndFmt = (CameraAndFormat*)userdata;
    camAndFmt->camera->setFormat(camAndFmt->pixelFormat,
				 camAndFmt->width, camAndFmt->height,
				 camAndFmt->fps_n, camAndFmt->fps_d);
}

//! 選択されたファイルに画像をセーブするためのコールバック関数．
/*!
  \param userdata	MyV4L2Camera (V4L2カメラ)
*/
static void
CBfileSelectionOK(GtkWidget* filesel, gpointer userdata)
{
    CameraAndFileSelection*	camAndFSel = (CameraAndFileSelection*)userdata;
    std::ofstream		out(gtk_file_selection_get_filename(
					GTK_FILE_SELECTION(
					    camAndFSel->filesel)));
    if (out)
	camAndFSel->camera->save(out);
    gtk_widget_destroy(camAndFSel->filesel);
}

//! 画像をセーブするファイルを選択するdialogを表示するためのコールバック関数．
/*!
  \param userdata	MyV4L2Camera (V4L2カメラ)
*/
static void
CBsave(GtkMenuItem*, gpointer userdata)
{
    GtkWidget*		filesel = gtk_file_selection_new("Save image");
    gtk_signal_connect(GTK_OBJECT(filesel), "destroy",
		       GTK_SIGNAL_FUNC(gtk_main_quit), filesel);
    cameraAndFileSelection.camera  = (MyV4L2Camera*)userdata;
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
  \param userdata	MyV4L2Camera (V4L2カメラ)
*/
static void
CBexit(GtkMenuItem*, gpointer userdata)
{
    using namespace	std;
    
    MyV4L2Camera*	camera = (MyV4L2Camera*)userdata;
    camera->continuousShot(false);
    cout << *camera;
    gtk_exit(0);
}

//! カメラからの画像にROIを設定するdialogを表示するためのコールバック関数．
/*!
  \param userdata	MyV4L2Camera (V4L2カメラ)
*/
static void
CBsetROI(GtkMenuItem*, gpointer userdata)
{
    MyV4L2Camera*	camera = (MyV4L2Camera*)userdata;
    size_t		u0, v0, width, height;
    if (camera->getROI(u0, v0, width, height))
    {
	MyDialog	dialog(*camera);
	dialog.getROI(u0, v0, width, height);
	camera->setROI(u0, v0, width, height);
    }
}
    
/************************************************************************
*  global functions							*
************************************************************************/
//! メニューバーを生成する．
/*!
  V4L2カメラがサポートしている画像フォーマットとフレームレートを調べて
  メニュー項目を決定する．
  \param camera		V4L2カメラ
  \return		生成されたメニューバー
*/
GtkWidget*
createMenubar(MyV4L2Camera& camera)
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

  // このカメラがサポートする画素フォーマットに対応するメニュー項目を作る．
    BOOST_FOREACH (V4L2Camera::PixelFormat pixelFormat,
		   camera.availablePixelFormats())
    {
      // フレームサイズを指定するためのサブメニューを生成．
	GtkWidget*	sizeMenu = gtk_menu_new();

      // この画素フォーマットがサポートする各フレームサイズに対応するメニュー項目を作る．
	BOOST_FOREACH (const V4L2Camera::FrameSize& frameSize,
		       camera.availableFrameSizes(pixelFormat))
	{
	    std::ostringstream	s;
	    s << frameSize;
	    GtkWidget*
		sizeItem = gtk_menu_item_new_with_label(s.str().c_str());
	    gtk_menu_append(GTK_MENU(sizeMenu), sizeItem);
	    
	    cameraAndFormats.push_back(CameraAndFormat());
	    CameraAndFormat&	camAndFmt = cameraAndFormats.back();
	    camAndFmt.camera	  = &camera;
	    camAndFmt.pixelFormat = pixelFormat;
	    camAndFmt.width	  = frameSize.width.max;
	    camAndFmt.height	  = frameSize.height.max;
	    camAndFmt.fps_n	  = frameSize.frameRates.front().fps_n.min;
	    camAndFmt.fps_d	  = frameSize.frameRates.front().fps_d.max;
	    gtk_signal_connect(GTK_OBJECT(sizeItem), "activate",
			       GTK_SIGNAL_FUNC(CBmenuitem),
			       (gpointer)&camAndFmt);
	}
	
      // この画素フォーマットに対応するメニュー項目を作る．
	GtkWidget*	fmtItem = gtk_menu_item_new_with_label(
				      camera.getName(pixelFormat).c_str());
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(fmtItem), sizeMenu);
	gtk_menu_append(GTK_MENU(menu), fmtItem);
    }

    item = gtk_menu_item_new_with_label("Format");
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), menu);
    gtk_menu_bar_append(GTK_MENU_BAR(menubar), item);

  // "Set ROI"メニューを生成
    item = gtk_menu_item_new_with_label("Set ROI...");
    gtk_signal_connect(GTK_OBJECT(item), "activate",
		       GTK_SIGNAL_FUNC(CBsetROI), &camera);
    gtk_menu_bar_append(GTK_MENU_BAR(menubar), item);
    
    gtk_widget_show_all(menubar);
    
    return menubar;
}
 
}
