/*
 *  $Id: main.cc,v 1.11 2012-08-29 19:35:49 ueshiba Exp $
 */
/*!
  \mainpage	testv4l2camera - program for testing an IIDC 1394-based Digital Camera
  \anchor	testv4l2camera

  \section copyright 著作権
  Copyright (C) 2003 Toshio UESHIBA
  National Institute of Advanced Industrial Science and Technology (AIST)

  Written by Toshio UESHIBA <t.ueshiba@aist.go.jp>

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software Foundation,
  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/
#include <stdlib.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include "MyV4L2Camera.h"

namespace TU
{
GtkWidget*	createMenubar(MyV4L2Camera& camera)			;
GtkWidget*	createCommands(MyV4L2Camera& camera)			;

/************************************************************************
*  static functions							*
************************************************************************/
//! 使用法を説明する
/*!
  \param s	コマンド名
*/
static void
usage(const char* s)
{
    using namespace	std;
    
    cerr << "\nControl an IEEE1394 digital camera.\n"
	 << endl;
    cerr << " Usage: " << s << " [-b] [uniqueID]"
         << endl;
    cerr << " arguments.\n"
         << "  -b:       IEEE1394b(800Mbps) mode\n"
         << "  uniqueID: camera unique-ID in hex format"
	 << " (i.e. 0x####, default: arbitrary)\n"
         << endl;
}

}
/************************************************************************
*  global functions							*
************************************************************************/
//! メイン関数
/*!
  "-p <port num>" でIEEE1394ポート(インターフェースカード)の番号を指定する．
  \param argc	引数の数(コマンド名を含む)
  \param argv   引数文字列の配列
*/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    gtk_init(&argc, &argv);	// GTK+ の初期化.

    const char*		dev = "/dev/video0";
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "d:")) != EOF; )
	switch (c)
	{
	  case 'd':
	    dev = optarg;
	    break;
	}
    
  // 本業を行う．
    try
    {
	MyV4L2Camera	camera(dev);			// カメラを開く．

	GtkWidget*	window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(window), "V4L2 camera controller");
	gtk_window_set_policy(GTK_WINDOW(window), FALSE, FALSE, TRUE);
	gtk_signal_connect(GTK_OBJECT(window), "destroy",
			   GTK_SIGNAL_FUNC(gtk_exit), NULL);
	gtk_signal_connect(GTK_OBJECT(window), "delete_event",
			   GTK_SIGNAL_FUNC(gtk_exit), NULL);

	GtkWidget*	table = gtk_table_new(2, 2, FALSE);
	gtk_container_add(GTK_CONTAINER(window), table);
	gtk_table_attach(GTK_TABLE(table), createMenubar(camera), 0, 2, 0, 1,
			 GTK_FILL, GTK_SHRINK, 0, 0);
	gtk_table_attach(GTK_TABLE(table), createCommands(camera), 1, 2, 1, 2,
			 GTK_SHRINK, GTK_SHRINK, 5, 10);
	gtk_table_attach(GTK_TABLE(table), camera.canvas(), 0, 1, 1, 2,
			 GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_widget_show_all(window);

	gtk_main();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
