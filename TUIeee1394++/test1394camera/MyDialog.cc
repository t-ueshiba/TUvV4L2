/*
 *  $Id: MyDialog.cc,v 1.1 2002-12-18 04:34:08 ueshiba Exp $
 */
#include "MyDialog.h"

namespace TU
{
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
    u_int	step = *(u_int*)userdata;
    gfloat	val = gfloat(step * ((u_int(adj->value) + step/2) / step));
    gtk_adjustment_set_value(adj, val);
}
    
/************************************************************************
*  class MyDialog							*
************************************************************************/
//! ROIを指定するためのdialog windowを生成する
/*!
  \param fmt7info	ROIを指定したいFormat_7_xに関する情報．
*/
MyDialog::MyDialog(const Ieee1394Camera::Format_7_Info& fmt7info)
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
				0.0))
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
    
  // widgetを並べるためのtable．
    GtkWidget*	table = gtk_table_new(2, 5, FALSE);

  // 4つのadjustmentのためのlabelと操作用scaleを生成してtableにパック．
    GtkWidget*	label = gtk_label_new("u0:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 0, 1);
    GtkWidget*	scale = gtk_hscale_new(GTK_ADJUSTMENT(_u0));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 30);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 0, 1);
    
    label = gtk_label_new("v0:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 1, 2);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_v0));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 30);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 1, 2);
    
    label = gtk_label_new("width:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 2, 3);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_width));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 30);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 2, 3);
    
    label = gtk_label_new("height:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 3, 4);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_height));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 30);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 3, 4);

  // ROI設定用のボタンを生成してtableにパック．
    GtkWidget*	button = gtk_button_new_with_label("Set");
    gtk_signal_connect(GTK_OBJECT(button), "clicked",
		       GTK_SIGNAL_FUNC(gtk_main_quit), _dialog);
    gtk_table_attach_defaults(GTK_TABLE(table), button, 0, 2, 4, 5);

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
void
MyDialog::getROI(u_int& u0, u_int& v0, u_int& width, u_int& height)
{
    u0	   = u_int(GTK_ADJUSTMENT(_u0)->value);
    v0	   = u_int(GTK_ADJUSTMENT(_v0)->value);
    width  = u_int(GTK_ADJUSTMENT(_width)->value);
    height = u_int(GTK_ADJUSTMENT(_height)->value);
}
    
};
