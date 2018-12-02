/*
 * testIIDCcamera: test program controlling an IIDC 1394-based Digital Camera
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
 */
/*
 * NOTE: このファイルはXPM形式の画像データを含んでいる
 */
#include "MyIIDCCameraArray.h"
#include <iomanip>
#include <fstream>
#include <cstring>

namespace TU
{
/************************************************************************
*  local data                                                           *
************************************************************************/
/*!
  上矢印のアイコン画像．
*/
static const char * up_xpm[] = {
"16 16 20 1",
"       c None",
".      c #C0C0C0",
"+      c #E0E0E0",
"@      c #949494",
"#      c #D3D3D3",
"$      c #C8C8C8",
"%      c #A1A1A1",
"&      c #F7F7F7",
"*      c #D4D4D4",
"=      c #E5E5E5",
"-      c #F1F1F1",
";      c #BBBBBB",
">      c #CECECE",
",      c #F0F0F0",
"'      c #AEAEAE",
")      c #DADADA",
"!      c #F6F6F6",
"~      c #EAEAEA",
"{      c #EBEBEB",
"]      c #ECECEC",
"                ",
"       .+       ",
"      @#$       ",
"      %$&$      ",
"     %$*=-$     ",
"    @;>++,$     ",
"    '$)+++!$    ",
"   %$*++++=-$   ",
"  @;>++++++,$   ",
"  '$)+++++++!$  ",
" %$*++++++++=-$ ",
"@;$***+++=~~~!$ ",
"@''''$+++{]'''* ",
"    @$+++]'     ",
"    @;$$$+'     ",
"     @@@@@@     "};

/*!
  下矢印のアイコン画像．
*/
static const char * down_xpm[] = {
"16 16 20 1",
"       c None",
".      c #949494",
"+      c #AEAEAE",
"@      c #E0E0E0",
"#      c #C8C8C8",
"$      c #BBBBBB",
"%      c #ECECEC",
"&      c #D4D4D4",
"*      c #EBEBEB",
"=      c #F6F6F6",
"-      c #EAEAEA",
";      c #E5E5E5",
">      c #F1F1F1",
",      c #A1A1A1",
"'      c #DADADA",
")      c #F0F0F0",
"!      c #CECECE",
"~      c #F7F7F7",
"{      c #D3D3D3",
"]      c #C0C0C0",
"     ......     ",
"     +@###$.    ",
"     +%@@@#.    ",
" &+++%*@@@#++++.",
" #=---;@@@&&&#$.",
" #>;@@@@@@@@&#, ",
"  #=@@@@@@@'#+  ",
"   #)@@@@@@!$.  ",
"   #>;@@@@&#,   ",
"    #=@@@'#+    ",
"     #)@@!$.    ",
"     #>;&#,     ",
"      #~#,      ",
"       #{.      ",
"       @]       ",
"                "};


/*!
  終了のアイコン画像．
*/
static const char * exit_xpm[] = {
"32 32 11 1",
" 	c None",
".	c #000000",
"+	c #323232",
"@	c #4C4C4C",
"#	c #808080",
"$	c #C48B1B",
"%	c #DE9D1E",
"&	c #AB7917",
"*	c #FFFFFF",
"=	c #E6E6E6",
"-	c #B2B2B2",
"....................            ",
".+.+.+.+.+.+...+.+..            ",
"..@#@@#@@#@@#@@#@@+.            ",
".+$$#@@#@@#@@#@@#@..            ",
"..%&$&@@#@@#@@#@@#***=          ",
".+$%$$&&#@#@#@#@#*****=         ",
"..%$%&$$&&@#@#@#@****==         ",
".+$%$%%&$&&@#@@##****=-         ",
"..%%%$%$&$&####@@#*==-#         ",
".+%$%%$%$$&@@###@@@--#@--       ",
"..%%%$%%$&$##@#@##+.*****=--    ",
".+%%%%%$$$&#@#@##@.*****=**=-   ",
"..%%%%$%$&$####@##*****=-=**=-  ",
".+%%%%%%$$&##@#-#**=***=- -=*=- ",
"..%%%%%$$$&@#####**==**=-  -**# ",
".+%%%%%%$$$#-*****=.=**=-   -=# ",
"..%%%%%%$$$@#=====+.-**==-  -## ",
".+%%%%%%%$&##-##-#..-****-      ",
"..%%%%%%@+&#######+.-****-      ",
".+%%%%%%%.$@-##-#@..-****-      ",
"..%%%%%%$$$##-##-#+.***=**-     ",
".+%%%%%%$&&####-##..***==*-     ",
"..%%%%%%$&$-#-###-+.***@=*-     ",
".+%%%%%$$$&#-#-#-#.***@ =*--    ",
"..%%%$%&&$&-#-#-#-+**@  @=**--- ",
".+%$%%$%$&&#-##-##**=@   @=****-",
"..%%$&%$&$&-#-#-#-**@     =====@",
".+$%%$$&$&&#-#-#-**=@     -@@@@ ",
"..%&%$&&&&#-----***@            ",
".+$$&&&&-#-#--#=**-.#           ",
"..+.+.+.+.+.+.+.+.+.@####@@@@@# ",
"....................##- -####-  "};


/*!
  保存のアイコン画像．
*/
static const char * save_xpm[] = {
"32 32 24 1",
" 	c None",
".	c #8A86C8",
"+	c #CBCBCB",
"@	c #666666",
"#	c #999999",
"$	c #E0E0E0",
"%	c #000000",
"&	c #FFFFFF",
"*	c #7974C8",
"=	c #B0ADE0",
"-	c #333333",
";	c #100000",
">	c #655EC8",
",	c #AAAAAA",
"'	c #6D68C8",
")	c #4840AD",
"!	c #281EAD",
"~	c #BEBEBE",
"{	c #5B53BA",
"]	c #1C157A",
"^	c #D26F6F",
"/	c #EBABAB",
"(	c #D24646",
"_	c #949494",
"                                ",
"                    ..          ",
"                 +@#$..         ",
"            #+@%@$&&#*==.       ",
"        ##+#$+-;-+$&#>*==.      ",
"    >@#,+$$$$$#;##$$&#'*=*.     ",
" >)!>@#~+$+$+$+-#-+$$#.'*=.     ",
"!.>'{),~,+$$$$+-;##+$@>>.*..    ",
"!{>.{!##~++$+$$#-#-++$#>>'=.    ",
"!)>'>)@#+,+~$$$+---#+$@.>*.*    ",
"])>'.'-,#+~++++$#-++$$~#>'=>    ",
">!{>'{)#~#,+,+~+$+$~#~@#>**..   ",
".!)''.!@#~~~~+,~#~##@#.>>>>=.   ",
" ]!>.')-,##~####@@@.{>>>>>*.*   ",
" ]!{''{!##,#@@.{{.{>.>{.{{*.>   ",
" >!)>>{)!-@{{{{>>>>>{{{@#~{*..  ",
" .!){>>{{{{>>>>{{{{{.@$&&&#{..  ",
"  ]!)>>>>>>>{{{.#@~$&&&&^^@{=*  ",
"  ]!)>>{{{.#@@#$&&&&/^^((^@{>.  ",
"  >!){{##@$&&&&&/^^((((^^/&#{.. ",
"  .]){~$&&&//^^((((^^//&&&_@{.. ",
"   !!{#&/^^((((^^//&&&&$~$&]{>. ",
"   ]!)@^^((^^/&&&&&$_~$&$$@#{>> ",
"   >]){@^^/&&&$~_~$&&$$@#..>>){ ",
"   .!!{@&&&$~_&&&$$@@##.>>{{){{ ",
"    ]!)#$~$&&$$@@##.>>>{{)!]!)  ",
"    ]!){]$@@@#..>>>{))!!])>     ",
"    >!!{#]>{){){))!!!]>>        ",
"    .!!)){{))!!!!]]>            ",
"     ]!!!!]!]!]>.               ",
"     ]!]!]>..                   ",
"     ]]>.                       "};



/*!
  カメラのアイコン画像．
*/
static const char * camera_xpm[] = {
"32 32 43 1",
" 	c None",
".	c #C8EAC7",
"+	c #B2D7B1",
"@	c #79C678",
"#	c #5CB15A",
"$	c #4BD94B",
"%	c #5FF85F",
"&	c #2ECD2E",
"*	c #83A782",
"=	c #739472",
"-	c #699B69",
";	c #12B00E",
">	c #6A6680",
",	c #898999",
"'	c #4E4B5E",
")	c #B9B8CC",
"!	c #319631",
"~	c #737380",
"{	c #22A022",
"]	c #2E2E33",
"^	c #A2A1B2",
"/	c #D0CFE6",
"(	c #A1A1B2",
"_	c #658264",
":	c #000000",
"<	c #7C7A99",
"[	c #B6C6B6",
"}	c #8A8A99",
"|	c #E6E6FF",
"1	c #226820",
"2	c #4D534D",
"3	c #427041",
"4	c #333B33",
"5	c #B8B8CC",
"6	c #758775",
"7	c #FFB7D4",
"8	c #C83972",
"9	c #750B0B",
"0	c #A3B0A3",
"a	c #C9D2C9",
"b	c #849384",
"c	c #BEBEBE",
"d	c #779477",
"                         .+@+.  ",
"                 .++@#$@$$%%$+  ",
"        ..++@##@##&$$*==*-&$%$  ",
"   .++@#@+#;&&;**===*>>,,*;&$%. ",
"+@#++;&&;**==**>'''>>,,)),*;&$+ ",
"!;;*===**>'''>>,~~,,,~~,)>,!&$$ ",
"{!=']>'>>,,,,,~~,^^////~~,>*$&$ ",
"&=']]',,(,(,,~>>^^)))))//~>=;$;.",
"{_]:]>,)(,,,~>)<<)>''''))/'*-[[@",
"!_':]'>(,,,~},'')>>>']]'')|*+122",
"33_':]',,,,~>)'])>:::~~]')/(*24[",
"!1_]:]>(,,~~,}])>::::55~]')/*@[.",
"13_]:]'>,,~})]])>:::5|5~]')/*36;",
"33_'::]',~'>}]:)>:7:|5::>')/*&;&",
"1!__':]'~~'>>::)>:89::::>>)/*=$;",
"!3!_]:]>,~']}:::)>987:]5>>)^,*;;",
"!3!_]:]'>~'>}:::)>:::]~]>)^^>*=&",
";1!_'::]'~''}}::])>::]]>>)<'',*!",
"{!3;_]:]'~~'>}:]:])>>>>')<^'~>*;",
"!1!;_]:]>,~':>}]]:])))))'<)~>'*{",
"3!=&_]:]'>~'']>}]':]]]]')),~,'*=",
";!;_3'::]>,~''>>}]''~}((),~,(>}*",
"0;&;;_':]',~~'']>}'~}(()(~,(,'>*",
"0!{;&_]:]'~,~~'']]^^~~~~~~']]]'*",
" ;@a[_]:]>,~,~~~''~'']]]'']]:]'*",
" b)24a':]]',~,']]]'']]:]]]':]'=;",
" +422@_]]''''']]:]]]']]:]'__=={{",
"  #++*_]']]]]]']]:]'____==&;{60b",
"  $*c&$_]]::]'____==&;6db313b0  ",
"  bc!;33==__==&;d3313b0         ",
"  0&;1;6b3;1;;b0                ",
"   33330                        "};

  /*!
    カメラの転送速度の設定値と名前の対応表．
    （転送速度はカメラ個々の属性だが，運用上の利便性を考えてまとめて設定する）
   */
    static struct
    {
        IIDCCamera::Speed	data_rate;
        const char*		name;
    } speed[] =
    {
      {IIDCCamera::SPD_400M, ""},	// 0 for backward compatibility
      {IIDCCamera::SPD_100M, "S100"},	// 1
      {IIDCCamera::SPD_200M, "S200"},	// 2
      {IIDCCamera::SPD_400M, "S400"},	// 3 default
      {IIDCCamera::SPD_800M, "S800"},	// 4 requires 1394.b
      {IIDCCamera::SPD_1_6G, "S1600"},	// 5 requires 1394.b
      {IIDCCamera::SPD_3_2G, "S3200"}	// 6 requires 1394.b
    };
    const u_int	nspeeds = sizeof(speed) / sizeof(speed[0]);
    static const int B_MODE = 4;

/************************************************************************
*  callback functions                                                   *
************************************************************************/
//! リストの選択項目を上に1つ上げるコールバック関数．
/*!
  \param userdata       MyIIDCCameraArray (IIDCボード)
*/
static void
CBup(GtkWidget* widget, gpointer userdata)
{
    static_cast<MyIIDCCameraArray*>(userdata)->up();
}

//! リストの選択項目を下に1つ下げるコールバック関数．
/*!
  \param userdata       MyIIDCCameraArray (IIDCボード)
*/
static void
CBdown(GtkWidget* widget, gpointer userdata)
{
    static_cast<MyIIDCCameraArray*>(userdata)->down();
}

//! カメラ画面を表示するためのコールバック関数．
/*!
  \param userdata       MyIIDCCameraArray (IIDCボード)
*/
static void
CBview(GtkWidget* widget, gpointer userdata)
{
    static_cast<MyIIDCCameraArray*>(userdata)->view();
}

//! カメラの転送速度を変更するコールバック関数
static void
CBspeedSelection(GtkWidget* widget, gpointer userdata)
{
    using namespace     std;

#if GTK_CHECK_VERSION(2,4,0)	// GTK+2.4.0 or later
    const auto	selection = gtk_combo_box_get_active_text(
				GTK_COMBO_BOX(widget));
#else
    const auto	selection = gtk_entry_get_text(GTK_ENTRY(widget));
#endif
    auto	data_rate = IIDCCamera::SPD_400M;
    for (const auto& spd : speed)
        if (strcmp(spd.name, selection) == 0)
	{
	    data_rate = spd.data_rate;
	    break;
	}

    for (auto& camera : *static_cast<MyIIDCCameraArray*>(userdata))
	camera.setSpeed(data_rate);
}

//! 選択されたファイルに設定内容をセーブするためのコールバック関数．
/*!
  \param userdata       MyIIDCCameraArray (IIDCボード)
*/
static void
CBfileSelectionOK(GtkWidget* widget, gpointer userdata)
{
    const auto	cameras   = static_cast<MyIIDCCameraArray*>(userdata);
    if (cameras->size() > 0)
    {
	const auto	filesel = cameras->popFileSelection();
	std::ofstream	out(gtk_file_selection_get_filename(
				GTK_FILE_SELECTION(filesel)));
	if (out)
	{
	    YAML::Emitter	emitter;
	    emitter << YAML::BeginSeq;
	    for (const auto& camera : *cameras)
		emitter << camera;
	    emitter << YAML::EndSeq;

	    out << emitter.c_str() << std::endl;
	}
	gtk_widget_destroy(filesel);
    }
}


//! 設定をセーブするファイルを選択するdialogを表示するためのコールバック関数．
/*!
  \param userdata       MyIIDCCameraArray (IIDCボード)
*/
static void
CBsave(GtkMenuItem*, gpointer userdata)
{
    const auto	cameras = static_cast<MyIIDCCameraArray*>(userdata);
    if (cameras->size() > 0)
    {
	const auto	filesel = gtk_file_selection_new("Save Preference");
	gtk_signal_connect(GTK_OBJECT(filesel), "destroy",
			   GTK_SIGNAL_FUNC(gtk_main_quit), filesel);
	cameras->pushFileSelection(filesel);
	gtk_signal_connect(GTK_OBJECT(GTK_FILE_SELECTION(filesel)->ok_button),
			   "clicked", (GtkSignalFunc)CBfileSelectionOK, cameras);
	gtk_signal_connect_object(GTK_OBJECT(GTK_FILE_SELECTION(filesel)
					     ->cancel_button), "clicked",
				  (GtkSignalFunc)gtk_widget_destroy,
				  GTK_OBJECT(filesel));

	char	filename[] = "IIDCCamera.conf";
	gtk_file_selection_set_filename(GTK_FILE_SELECTION(filesel), filename);
	gtk_widget_show(filesel);
	gtk_main();
    }
}

//! アプリケーションを終了するためのコールバック関数．
/*!
  \param userdata       MyIIDCCameraArray (IIDCボード)
*/
static void
CBexit(gpointer userdata)
{
    static_cast<MyIIDCCameraArray*>(userdata)->clear();
    gtk_main_quit();
}


/************************************************************************
*  local functions                                                      *
************************************************************************/
//! 画像付きボタンを生成する．
/*
  アイコン画像とテキストラベルを貼り付けたボタンを生成する．
  \param window	背景色を取得するためのウィジェット
  \param xpmImage	アイコン画像データ(XPM)
  \param labelText	ラベル文字列
 */
static GtkWidget*
createIconBox(GtkWidget* window, gchar** xpmImage, gchar* labelText)
{
    GtkWidget*	icon_box = gtk_hbox_new(FALSE, 0);
    GtkWidget*	top_box = gtk_hbox_new(FALSE, 0);
    gtk_container_set_border_width(GTK_CONTAINER(icon_box), 2);
    gtk_box_pack_start(GTK_BOX(top_box), icon_box, TRUE, FALSE, 0);

    GdkBitmap*	mask;
    GdkColor	trans = (gtk_widget_get_style(window))->bg[GTK_STATE_NORMAL];
    GdkPixmap*	pixmap = gdk_pixmap_create_from_xpm_d(window->window, &mask,
						      &trans, xpmImage);
    GtkWidget*	pixmapwid = gtk_pixmap_new (pixmap, mask);
    GtkWidget*	label = gtk_label_new(labelText);

    gtk_box_pack_start(GTK_BOX(icon_box), pixmapwid, FALSE, FALSE, 3);
    gtk_box_pack_start(GTK_BOX(icon_box), label, FALSE, FALSE, 3);

    gtk_widget_show(pixmapwid);
    gtk_widget_show(label);
    return top_box;
}

/************************************************************************
*  global functions                                                     *
************************************************************************/
//! カメラのリストに対するコマンド群を生成する．
/*!
  カメラのリストを操作するコマンドを生成する．
  \param cameras          IIDCボード
  \return               生成されたコマンド群が貼りつけられたテーブル
*/
GtkWidget*
createCameraArrayCommands(MyIIDCCameraArray& cameras, GtkWidget* window,
			  IIDCCamera::Speed data_rate)
{
    const auto	commands = gtk_table_new(2, 2, FALSE);

  // 上矢印＆下矢印ボタン
    const auto	arrows = gtk_vbox_new(TRUE, 0);
    gtk_table_attach(GTK_TABLE(commands), arrows, 0, 1, 0, 2,
		     GTK_SHRINK, GTK_FILL, 5, 5);

  // ボタンの透明(背景)色を取得する．
    gtk_widget_show(window);
    const auto	trans = (gtk_widget_get_style(window))->bg[GTK_STATE_NORMAL];
    GdkBitmap*	mask;
    const auto	upPixmap = gdk_pixmap_create_from_xpm_d(window->window,
							&mask, &trans,
							(gchar**)up_xpm);
    const auto	up = gtk_button_new();
    gtk_container_add(GTK_CONTAINER(up), gtk_pixmap_new(upPixmap, mask));
    gtk_box_pack_start(GTK_BOX(arrows), up, FALSE, TRUE, 0);
    gtk_signal_connect(GTK_OBJECT(up), "clicked",
		       GTK_SIGNAL_FUNC(CBup), &cameras);

    const auto	downPixmap = gdk_pixmap_create_from_xpm_d(window->window,
							  &mask, &trans,
							  (gchar**)down_xpm);
    const auto	down = gtk_button_new();
    gtk_container_add(GTK_CONTAINER(down), gtk_pixmap_new(downPixmap, mask));
    gtk_box_pack_start(GTK_BOX(arrows), down, FALSE, TRUE, 0);
    gtk_signal_connect(GTK_OBJECT(down), "clicked",
		       GTK_SIGNAL_FUNC(CBdown), &cameras);

  // カメラ表示ボタン
    const auto	toolBox = gtk_vbox_new(TRUE, 0);
    gtk_container_set_border_width(GTK_CONTAINER(toolBox), 10);
    gtk_table_attach_defaults(GTK_TABLE(commands), toolBox, 1, 2, 0, 1);

    const auto	view = gtk_button_new();
    gtk_container_add(GTK_CONTAINER(view), 
		      createIconBox(window, (gchar**)camera_xpm, "View"));
    gtk_box_pack_start(GTK_BOX(toolBox), view, FALSE, TRUE, 5);
    gtk_signal_connect(GTK_OBJECT(view), "clicked",
		       GTK_SIGNAL_FUNC(CBview), &cameras);

  // 転送速度選択コンボボックス
    auto	dr = nspeeds;
#if GTK_CHECK_VERSION(2,4,0)	// GTK+2.4.0 or later
    GtkWidget*	speedbox = gtk_combo_box_new_text();
    for (u_int i = 0; i < nspeeds; i++)
    {
        gtk_combo_box_append_text(GTK_COMBO_BOX(speedbox), speed[i].name);
	if (data_rate == speed[i].data_rate)
	    dr = i;
    }
    gtk_combo_box_set_active(GTK_COMBO_BOX(speedbox), dr); // default speed
    gtk_signal_connect(GTK_OBJECT(speedbox), "changed",
		       GTK_SIGNAL_FUNC(CBspeedSelection), &cameras);
#else
    GList*	items = nullptr;
    for (int i = 0; i < nspeeds; i++)
    {
        items = g_list_append(items, (void*)speed[i].name);
	if (data_rate == speed[i].data_rate)
	    dr = i;
    }
    const auto	speedbox = gtk_combo_new();
    gtk_combo_set_popdown_strings(GTK_COMBO(speedbox), items);
    gtk_entry_set_editable(GTK_ENTRY(GTK_COMBO(speedbox)->entry), FALSE);
    gtk_entry_set_text(GTK_ENTRY(GTK_COMBO(speedbox)->entry), speed[dr].name);
    gtk_signal_connect(GTK_OBJECT(GTK_COMBO(speedbox)->entry), "changed",
		       GTK_SIGNAL_FUNC(CBspeedSelection), &cameras);
#endif
    gtk_box_pack_start(GTK_BOX(toolBox), speedbox, FALSE, TRUE, 5);
    cameras.setSpeedPreference(speedbox);

  // 保存ボタン＆終了ボタン
    const auto	fileBox = gtk_vbox_new(TRUE, 0);
    gtk_container_set_border_width(GTK_CONTAINER(fileBox), 10);
    gtk_table_attach_defaults(GTK_TABLE(commands), fileBox, 1, 2, 1, 2);

    const auto	savePreference = gtk_button_new();
    gtk_signal_connect(GTK_OBJECT(savePreference), "clicked",
                       GTK_SIGNAL_FUNC(CBsave), &cameras);
    gtk_container_add(GTK_CONTAINER(savePreference), 
		      createIconBox(window, (gchar**)save_xpm, "Save"));
    gtk_box_pack_start(GTK_BOX(fileBox), savePreference, FALSE, TRUE, 5);

    const auto	appExit = gtk_button_new();
    gtk_container_add(GTK_CONTAINER(appExit), 
		      createIconBox(window, (gchar**)exit_xpm, "Exit"));

    gtk_box_pack_start(GTK_BOX(fileBox), appExit, FALSE, TRUE, 5);
    gtk_signal_connect_object(GTK_OBJECT(appExit), "clicked",
			      GTK_SIGNAL_FUNC(CBexit), (GtkObject*)&cameras);

    return commands;
}

}
