/*
 *  $Id: main.cc,v 1.8 2009-05-13 01:15:01 ueshiba Exp $
 */
/*!
  \mainpage	test1394camera - program for testing an IIDC 1394-based Digital Camera
  \anchor	test1394camera

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

  \section functions 機能
  IEEE1394デジタルカメラのテストプログラム．1台のカメラに対して
  種々の設定を行ったり，撮影した画像ストリームをリアルタイムで
  X window上に表示したりできる．LINUX上で
  <a href="http://www.1394ta.com/">IEEE1394</a>デバイスおよび
  <a href="http://www.1394ta.com/Technology/Specifications/Descriptions/IIDC_Spec_v1_30.htm">IEEE1394デジタルカメラ</a>を使用するためのコントロールライブラリ:
  \ref libTUIeee1394 "libTUIeee1394++"を用いている．

  \section invocation コマンド呼び出しの形式
  \verbatim
  test1394camera [-b] [uniqueID]\endverbatim

    - [<tt>-b</tt>] IEEE1394b (FireWire 800)モードを使用
    - [<tt>uniqueID</tt>] カメラが複数ある場合に特定のカメラを指定するためのglobal
	uniqne IDを16進形式(0x####)で与える

  プログラム終了時に，カメラのglobal unique IDと設定値が標準出力に出力される．	
*/
#if HAVE_CONFIG_H
#  include <config.h>
#endif
#include <stdlib.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include "My1394Camera.h"

namespace TU
{
GtkWidget*	createMenubar(My1394Camera& camera)			;
GtkWidget*	createCommands(My1394Camera& camera)			;

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

  // IEEE1394ポート(インターフェースカード)の番号をコマンド行から読み込む．
    bool		i1394b = false;		// default: 非IEEE1394bモード
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "bh")) != EOF; )
	switch (c)
	{
	  case 'b':
	    i1394b = true;
	    break;
	  case 'h':
	    usage(argv[0]);
	    return 1;
	}
    extern int	optind;
    u_int64	uniqId = 0;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
  // 本業を行う．
    try
    {
	My1394Camera	camera(i1394b, uniqId);		// カメラを開く．

	GtkWidget*	window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(window), "IEEE1394 camera controller");
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
			 GTK_SHRINK, GTK_SHRINK, 5, 0);
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
