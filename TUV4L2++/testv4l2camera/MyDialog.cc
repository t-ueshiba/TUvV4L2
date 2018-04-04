/*
 *  $Id: MyDialog.cc,v 1.5 2012-08-29 19:35:49 ueshiba Exp $
 */
#if HAVE_CONFIG_H
#  include <config.h>
#endif
#include "MyDialog.h"

namespace TU
{
/************************************************************************
*  class MyDialog							*
************************************************************************/
//! ROIを指定するためのdialog windowを生成する
/*!
  \param fmt7info	ROIを指定したいFormat_7_xに関する情報．
*/
MyDialog::MyDialog(const V4L2Camera& camera)
    :_dialog(gtk_dialog_new()), _u0(0), _v0(0), _width(0), _height(0)
{
    size_t	minU0, minV0, maxWidth, maxHeight;
    camera.getROILimits(minU0, minV0, maxWidth, maxHeight);
    size_t	u0, v0, width, height;
    camera.getROI(u0, v0, width, height);

    _u0	    = gtk_adjustment_new(u0, minU0, maxWidth  - 1, 1, 1, 0);
    _v0	    = gtk_adjustment_new(v0, minV0, maxHeight - 1, 1, 1, 0);
    _width  = gtk_adjustment_new(width,  0, maxWidth,	   1, 1, 0);
    _height = gtk_adjustment_new(height, 0, maxHeight,	   1, 1, 0);
    
  // dialogの属性を設定．
    gtk_window_set_modal(GTK_WINDOW(_dialog), TRUE);
    gtk_window_set_title(GTK_WINDOW(_dialog), "Setting ROI");
    gtk_window_set_policy(GTK_WINDOW(_dialog), FALSE, FALSE, TRUE);
    
  // widgetを並べるためのtable．
    GtkWidget*	table = gtk_table_new(2, 5, FALSE);

  // 4つのadjustmentのためのlabelと操作用scaleを生成してtableにパック．
    GtkWidget*	label = gtk_label_new("u0:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 0, 1);
    GtkWidget*	scale = gtk_hscale_new(GTK_ADJUSTMENT(_u0));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 0, 1);
    
    label = gtk_label_new("v0:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 1, 2);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_v0));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 1, 2);
    
    label = gtk_label_new("width:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 2, 3);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_width));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 2, 3);
    
    label = gtk_label_new("height:");
    gtk_table_attach_defaults(GTK_TABLE(table), label, 0, 1, 3, 4);
    scale = gtk_hscale_new(GTK_ADJUSTMENT(_height));
    gtk_scale_set_digits(GTK_SCALE(scale), 0);
    gtk_widget_set_usize(GTK_WIDGET(scale), 200, 50);
    gtk_table_attach_defaults(GTK_TABLE(table), scale, 1, 2, 3, 4);

  // ROI設定用のボタンを生成してtableにパック．
    GtkWidget*	button = gtk_button_new_with_label("Set");
    gtk_signal_connect(GTK_OBJECT(button), "clicked",
		       GTK_SIGNAL_FUNC(gtk_main_quit), _dialog);
    gtk_table_attach_defaults(GTK_TABLE(table), button, 0, 2, 5, 6);

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
MyDialog::getROI(size_t& u0, size_t& v0, size_t& width, size_t& height) const
{
    u0	   = size_t(GTK_ADJUSTMENT(_u0)->value);
    v0	   = size_t(GTK_ADJUSTMENT(_v0)->value);
    width  = size_t(GTK_ADJUSTMENT(_width)->value);
    height = size_t(GTK_ADJUSTMENT(_height)->value);
}
    
};
