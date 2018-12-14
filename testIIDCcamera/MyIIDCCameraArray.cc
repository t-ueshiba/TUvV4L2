/*
 * test1394camera: test program controlling an IIDC 1394-based Digital Camera
 * Copyright (C) 2006 VVV Working Group
 *   National Institute of Advanced Industrial Science and Technology (AIST)
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
 *  $Id: My1394CameraArray.cc 822 2012-08-27 02:39:41Z takase $
 */
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cassert>
#include "MyIIDCCameraArray.h"

namespace TU
{
GtkWidget*      createMenubar(MyIIDCCamera& camera, GtkWidget* showable);
GtkWidget*	createCommands(MyIIDCCamera& camera)			;

/************************************************************************
*  local data                                                           *
************************************************************************/
static const gchar*	columnTitles[2] = { "No.", "Camera ID" };

/************************************************************************
*  class MyIIDCCameraArray                                                   *
************************************************************************/
//! IEEE1394ボードを生成する
/*!
  \param port   カメラが接続されているポート．
*/
MyIIDCCameraArray::MyIIDCCameraArray()
    :_canvas(gtk_scrolled_window_new(NULL, NULL)),
     _list(gtk_clist_new_with_titles(2, const_cast<gchar**>(columnTitles))),
     _filesel(0),_speedPreference(0)
{
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(_canvas),
				   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_usize(GTK_WIDGET(_canvas), 250, 150);

    gtk_clist_set_selection_mode(GTK_CLIST(_list), GTK_SELECTION_SINGLE);
    gtk_clist_set_shadow_type(GTK_CLIST(_list), GTK_SHADOW_OUT);
    gtk_clist_column_titles_passive(GTK_CLIST(_list));
    gtk_clist_set_column_justification(GTK_CLIST(_list),0, GTK_JUSTIFY_CENTER);
    gtk_clist_set_column_justification(GTK_CLIST(_list),1, GTK_JUSTIFY_LEFT);
    gtk_clist_set_column_width(GTK_CLIST(_list), 0, 30);
    gtk_clist_set_column_width(GTK_CLIST(_list), 1, 220);
    gtk_container_add(GTK_CONTAINER(_canvas), _list);

  // カメラを探す
    scan();

    if (size() > 0)
	gtk_clist_select_row(GTK_CLIST(_list), 0, 1);
}


MyIIDCCameraArray::~MyIIDCCameraArray()
{
    clear();
}


GtkWidget*
MyIIDCCameraArray::view()
{
    const auto	selection = GTK_CLIST(_list)->selection;

    if (!selection)
	return nullptr;

    size_t	index = GPOINTER_TO_INT(selection->data);
    const auto	userdata = gtk_clist_get_row_data(GTK_CLIST(_list), index);
    GtkWidget*	subwindow;
    if (userdata)
    {
	subwindow = GTK_WIDGET(userdata);
    }
    else
    {
	auto&	camera = _cameras[index];
	    
      // カメラ画面を作成し，選択中のリスト項目に付加する．
	subwindow = gtk_window_new(GTK_WINDOW_TOPLEVEL);

	gchar title[30];
	g_snprintf(title, 30, "Camera - 0x016%lx", camera.globalUniqueId());
	gtk_window_set_title(GTK_WINDOW(subwindow), title);
	gtk_window_set_policy(GTK_WINDOW(subwindow), FALSE, FALSE, TRUE);
	gtk_signal_connect(GTK_OBJECT(subwindow), "destroy",
			   GTK_SIGNAL_FUNC(gtk_widget_hide_on_delete),
			   GTK_OBJECT(subwindow));
	gtk_signal_connect(GTK_OBJECT(subwindow), "delete_event",
			   GTK_SIGNAL_FUNC(gtk_widget_hide_on_delete),
			   GTK_OBJECT(subwindow));
	
	auto	table = gtk_table_new(2, 2, FALSE);
	auto	commands = createCommands(camera);
	camera.setCommands(commands, table);
	gtk_container_add(GTK_CONTAINER(subwindow), table);
	gtk_table_attach(GTK_TABLE(table), createMenubar(camera, subwindow),
			 0, 2, 0, 1, GTK_FILL, GTK_SHRINK, 0, 0);
	gtk_table_attach(GTK_TABLE(table), commands,
			 1, 2, 1, 2, GTK_SHRINK, GTK_SHRINK, 5, 0);
      // 1,2,1,2に配置: MyIIDCCamera::refreshCommands()
	gtk_table_attach(GTK_TABLE(table), camera.canvas(),
			 0, 1, 1, 2, GTK_SHRINK, GTK_SHRINK, 0, 0);

	gtk_clist_set_row_data(GTK_CLIST(_list), index, subwindow);
    }
    gtk_widget_show_all(GTK_WIDGET(subwindow));

    return GTK_WIDGET(subwindow);
}

void
MyIIDCCameraArray::clear()
{
    for (size_t i = 0; i < size(); i++)
    {
	// カメラ画面を破棄
	const auto	subwindow = gtk_clist_get_row_data(GTK_CLIST(_list), i);
	if (subwindow)
	    gtk_widget_destroy(GTK_WIDGET(subwindow));
    }
    gtk_clist_clear(GTK_CLIST(_list));

    _cameras.clear();
}

void
MyIIDCCameraArray::scan()
{
    clear();

    for (;;)
    {
	try
	{
	    emplace_back(0);
	}
	catch (std::exception& except)
	{
	    break;	// このノードはIIDCカメラではなかった．
	}
    }

    return;
}

void
MyIIDCCameraArray::up()
{
    const auto	selection = GTK_CLIST(_list)->selection;
    if (selection)
    {
	size_t	index = GPOINTER_TO_INT(selection->data);
	if ((size() > 1) && (index > 0))
	{
	    swap(index, index-1);
	    gtk_clist_select_row(GTK_CLIST(_list), index-1, 1);
	}
    }
}

void
MyIIDCCameraArray::down()
{
    const auto	selection = GTK_CLIST(_list)->selection;
    if (selection)
    {
	u_int index = GPOINTER_TO_INT(selection->data);
	if ((size() > 1) && (index + 1 < size()))
	{
	    swap(index, index+1);
	    gtk_clist_select_row(GTK_CLIST(_list), index+1, 1);
	}
    }
}

void
MyIIDCCameraArray::pushFileSelection(GtkWidget* filesel)
{
    _filesel = filesel;
}

GtkWidget*
MyIIDCCameraArray::popFileSelection()
{
    const auto	w = _filesel;
    _filesel = 0;		// ファイル選択画面はその度ごとに破棄する
    return w;
}

void
MyIIDCCameraArray::setSpeedPreference(GtkWidget* speedPreference)
{
    _speedPreference = speedPreference;
}

GtkWidget*
MyIIDCCameraArray::getSpeedPreference()	const
{
    return _speedPreference;
}

void
MyIIDCCameraArray::swap(size_t i, size_t j)
{
    if ((i + 1 > size()) || (j + 1 > size()))
	throw std::invalid_argument("Invalid index of list item!!");

    std::swap(_cameras[i], _cameras[j]);

    // 通し番号は変えず，GUIDとデータを入れ替え．
    gchar guid[20];
    g_snprintf(guid, 20, "0x%016lx", _cameras[i].globalUniqueId());
    gtk_clist_set_text(GTK_CLIST(_list), i, 1, guid);
    g_snprintf(guid, 20, "0x%016lx", _cameras[j].globalUniqueId());
    gtk_clist_set_text(GTK_CLIST(_list), j, 1, guid);

    const auto	iwindow = gtk_clist_get_row_data(GTK_CLIST(_list), i);
    const auto	jwindow = gtk_clist_get_row_data(GTK_CLIST(_list), j);
    gtk_clist_set_row_data(GTK_CLIST(_list), i, jwindow);
    gtk_clist_set_row_data(GTK_CLIST(_list), j, iwindow);

}

void
MyIIDCCameraArray::emplace_back(uint64_t uniqId)
{
  // カメラを追加
    _cameras.push_back(uniqId);

  // リストに新しいカメラの通し番号とGUIDを追加．
    const auto	index = size() - 1;
    gchar*	item[2];
    item[0] = new gchar[10];
    item[1] = new gchar[20];
    g_snprintf(item[0], 10, "%lu", index);
    g_snprintf(item[1], 20, "0x%016lx", _cameras.back().globalUniqueId());
    gtk_clist_append(GTK_CLIST(_list), item);
    delete[] item[0];
    delete[] item[1];

  // カメラ画面は初期化に時間がかかるので，表示するときに遅延作成する．
    GtkWidget*	window = nullptr;
    gtk_clist_set_row_data(GTK_CLIST(_list), index, window);
}

}
