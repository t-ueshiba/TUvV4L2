/*
 *  $Id: main.cc,v 1.2 2002-12-18 04:34:08 ueshiba Exp $
 */
/*!
  \mainpage	test1394camera
  \anchor	test1394camera
  IEEE1394デジタルカメラのテストプログラム．1台のカメラに対して
  種々の設定を行ったり，撮影した画像ストリームをリアルタイムで
  X window上に表示したりできる．LINUX上で
  <a href="http://www.1394ta.com/">IEEE1394</a>デバイスおよび
  <a href="http://www.1394ta.com/Technology/Specifications/Descriptions/IIDC_Spec_v1_30.htm">IEEE1394デジタルカメラ</a>を使用するためのコントロールライブラリ:
  \ref libTUIeee1394 "libTUIeee1394++"を用いている．
*/
#include <stdlib.h>
#include <unistd.h>
#include <stdexcept>
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
    cerr << "\nControl an IEEE1394 digital camera.\n"
	 << endl;
    cerr << " Usage: " << s << " [-p portnum] [uniqueID]"
         << endl;
    cerr << " arguments.\n"
         << "  -p portnum:  port number (i.e. board number) of IEEE1394 card"
	 << " (default: 0)\n"
         << "  uniqueID:    camera unique-ID in hex format"
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
    int			port_number = 0;	// default: 0番目のカード
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "p:h")) != EOF; )
	switch (c)
	{
	  case 'p':
	    port_number = atoi(optarg);
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
	Ieee1394Port	port(port_number);	// ポートを開く．
	My1394Camera	camera(port, uniqId);	// カメラを開く．

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
