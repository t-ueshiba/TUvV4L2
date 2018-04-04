/*
 *  $Id: main.cc,v 1.11 2012-08-29 19:35:49 ueshiba Exp $
 */
/*!
  \mainpage	testIIDCcamera - program for testing an IIDC-based Digital Camera
  \anchor	testIIDCcamera

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
  IIDCデジタルカメラのテストプログラム．1台のカメラに対して
  種々の設定を行ったり，撮影した画像ストリームをリアルタイムで
  X window上に表示したりできる．LINUX上で
  <a href="http://www.1394ta.com/">IIDC</a>デバイスおよび
  <a href="http://www.1394ta.com/Technology/Specifications/Descriptions/IIDC_Spec_v1_30.htm">IIDCデジタルカメラ</a>を使用するためのコントロールライブラリ:
  \ref libTUIIDC "libTUIIDC++"を用いている．

  \section invocation コマンド呼び出しの形式
  \verbatim
  testIIDCcamera [uniqueID]\endverbatim

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
#include "MyIIDCCamera.h"

namespace TU
{
GtkWidget*	createMenubar(MyIIDCCamera& camera)			;
GtkWidget*	createCommands(MyIIDCCamera& camera)			;

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
    
    cerr << "\nControl an IIDC digital camera."
	 << endl;
    cerr << " Usage: " << s << " [options] [uniqueID]"
         << endl;
    cerr << "  options.\n"
	 << "   -1:       set FireWire speed to 100Mb/s\n"
	 << "   -2:       set FireWire speed to 200Mb/s\n"
	 << "   -4:       set FireWire speed to 400Mb/s (default)\n"
	 << "   -8:       set FireWire speed to 800Mb/s"
	 << endl;
    cerr << "  arguments.\n"
         << "   uniqueID: camera unique-ID in hex format"
	 << " (i.e. 0x####, default: arbitrary)\n"
         << endl;
}

}
/************************************************************************
*  global functions							*
************************************************************************/
//! メイン関数
/*!
  \param argc	引数の数(コマンド名を含む)
  \param argv   引数文字列の配列
*/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    gtk_init(&argc, &argv);	// GTK+ の初期化.

    IIDCCamera::Speed	speed = IIDCCamera::SPD_400M;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "1248h")) != EOF; )
	switch (c)
	{
	  case '1':
	    speed = IIDCCamera::SPD_100M;
	    break;
	  case '2':
	    speed = IIDCCamera::SPD_200M;
	    break;
	  case '4':
	    speed = IIDCCamera::SPD_400M;
	    break;
	  case '8':
	    speed = IIDCCamera::SPD_800M;
	    break;
	  case 'h':
	    usage(argv[0]);
	    return 1;
	}
    extern int	optind;
    uint64_t	uniqId = 0;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
  // 本業を行う．
    try
    {

	MyIIDCCamera	camera(uniqId);		// カメラを開く．

	camera.setSpeed(speed);
	
	const auto	window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(window), "IIDC camera controller");
	gtk_window_set_policy(GTK_WINDOW(window), FALSE, FALSE, TRUE);
	gtk_signal_connect(GTK_OBJECT(window), "destroy",
			   GTK_SIGNAL_FUNC(gtk_exit), NULL);
	gtk_signal_connect(GTK_OBJECT(window), "delete_event",
			   GTK_SIGNAL_FUNC(gtk_exit), NULL);

	const auto	table = gtk_table_new(2, 2, FALSE);
	gtk_container_add(GTK_CONTAINER(window), table);
	gtk_table_attach(GTK_TABLE(table), createMenubar(camera),
			 0, 2, 0, 1, GTK_FILL, GTK_SHRINK, 0, 0);
	gtk_table_attach(GTK_TABLE(table), createCommands(camera),
			 1, 2, 1, 2, GTK_SHRINK, GTK_SHRINK, 5, 0);
	gtk_table_attach(GTK_TABLE(table), camera.canvas(),
			 0, 1, 1, 2, GTK_SHRINK, GTK_SHRINK, 0, 0);
	gtk_widget_show_all(window);

	gtk_main();
    
	cout << camera;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
