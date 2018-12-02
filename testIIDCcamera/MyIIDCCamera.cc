/*
 * testIIDCcamera: test program controlling an IIDC-based Digital Camera
 * Copyright (C) 2003 Toshio UESHIBA
 *   National Institute of Advanced Industrial Science and Technology (AIST)
 *
 * Written by Toshio UESHIBA <t.ueshiba@aist.go.jp>
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
 *  $Id: MyIIDCCamera.cc,v 1.14 2012-08-29 19:35:49 ueshiba Exp $
 */
#include <sys/time.h>
#include <stdexcept>
#include <iostream>
#include "MyIIDCCamera.h"

namespace TU
{
GtkWidget*	createCommands(MyIIDCCamera& camera)			;

/************************************************************************
*  static functions							*
************************************************************************/
//! ループ10回に要した時間の平均をとることにより，フレームレートを測定する．
static void
countTime()
{
    static int			nframes = 0;
    static struct timeval	start;

    if (nframes == 10)
    {
	struct timeval	end;
	gettimeofday(&end, NULL);
	double	interval = (end.tv_sec  - start.tv_sec) +
	    (end.tv_usec - start.tv_usec) / 1.0e6;
	std::cerr << nframes / interval << " frames/sec" << std::endl;
	nframes = 0;
    }
    if (nframes++ == 0)
	gettimeofday(&start, NULL);
}

//! 再描画のためのコールバック関数．
/*!
  \param widget		DrawingArea widget
  \param event		イベント
  \param userdata	MyIIDCCamera (IIDCカメラ)
  \return		TRUEを返す
*/
static gboolean
CBexpose(GtkWidget* widget, GdkEventExpose* event, gpointer userdata)
{
    static_cast<MyIIDCCamera*>(userdata)->draw();
    return TRUE;
}

/************************************************************************
*  class MyIIDCCamera							*
************************************************************************/
//! IIDCカメラノードを生成する
/*!
  \param uniqId	個々のカメラ固有の64bit ID．同一のIIDC busに
		複数のカメラが接続されている場合，これによって
		同定を行う．
*/
MyIIDCCamera::MyIIDCCamera(uint64_t uniqId)
    :IIDCCamera(uniqId),
     _canvas(gtk_drawing_area_new()),
     _buf(),
     _rgb()
{
    gdk_rgb_init();
    gtk_signal_connect(GTK_OBJECT(_canvas), "expose_event",
		       GTK_SIGNAL_FUNC(CBexpose), (gpointer)this);
    
  // 現在のカメラのフォーマットに合わせてバッファの確保を行う．
    setFormatAndFrameRate(getFormat(), getFrameRate());
}

//! 画像フォーマットとフレームレートを指定する．
/*!
  さらに入力画像バッファとRGBバッファを確保し直し，canvasの大きさを変更する．
  \param format	設定したい画像フォーマット．
  \param rate	設定したいフレームレート．
  \return	このIIDCカメラオブジェクト．
*/
IIDCCamera&
MyIIDCCamera::setFormatAndFrameRate(Format format, FrameRate rate)
{
  // IIDCカメラに対して画像フォーマットとフレームレートを指定する．
    IIDCCamera::setFormatAndFrameRate(format, rate);

  // 指定したフォーマットに合わせて入力画像バッファとRGBバッファを再確保する．
    u_int	buffSize;
    switch (pixelFormat())
    {
      case YUV_444:
      case RGB_24:
	buffSize = width() * height() * 3;
	break;
      case YUV_422:
      case MONO_16:
      case SIGNED_MONO_16:
      case RAW_16:
	buffSize = width() * height() * 2;
	break;
      case YUV_411:
	buffSize = width() * height() * 3 / 2;
	break;
      case MONO_8:
      case RAW_8:
	buffSize = width() * height();
	break;
      default:
	throw std::invalid_argument("Unsupported camera format!!");
	break;
    }
    _buf.resize(buffSize);
    _rgb.resize(width() * height());

  // 指定したフォーマットに合わせてcanvasの大きさを変更する．
    gtk_drawing_area_size(GTK_DRAWING_AREA(_canvas), width(), height());
    gtk_widget_show(_canvas);
    
    return *this;
}

//! カメラから画像を読み込み，canvasに表示する．
/*!
  画像キャプチャボタンが押されている間は，idle関数として他にやることが
  ない時にくりかえし呼ばれる．
*/
void
MyIIDCCamera::idle()
{
  // フレームレートの測定．
    countTime();

  // カメラから画像データを読み込む．
    if (bayerTileMapping() != IIDCCamera::YYYY &&
	((pixelFormat() == MONO_8)  ||
	 (pixelFormat() == MONO_16) || (pixelFormat() == SIGNED_MONO_16)))
	snap().captureBayerRaw(_rgb.data());
    else
	snap().captureRaw(_buf.data());
    draw();			// canvasに表示する．
}

//! バッファ中の画像画像をcanvasに表示する．
/*!
  idle(), CBexpose()から呼ばれる．
*/
void
MyIIDCCamera::draw()
{
  // 必用なら YUV -> RGB の変換を行ってから画像を表示．
    switch (pixelFormat())
    {
      case YUV_444:
      {
	auto	p = reinterpret_cast<const YUV444*>(_buf.data());
	auto	q = _rgb.data();
	for (size_t y = 0; y < height(); ++y)
	{
	    std::copy(make_pixel_iterator(p), make_pixel_iterator(p + width()),
		      make_pixel_iterator(q));
	    p += width();
	    q += width();
	}
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(), GDK_RGB_DITHER_NONE,
			   reinterpret_cast<guchar*>(_rgb.data()), 3*width());
      }
	break;

      case YUV_422:
      {
	auto	p = reinterpret_cast<const YUV422*>(_buf.data());
	auto	q = _rgb.data();
	for (size_t y = 0; y < height(); ++y)
	{
	    std::copy(make_pixel_iterator(p), make_pixel_iterator(p + width()),
		      make_pixel_iterator(q));
	    p += width();
	    q += width();
	}
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(), GDK_RGB_DITHER_NONE,
			   reinterpret_cast<guchar*>(_rgb.data()), 3*width());
      }
	break;

      case YUV_411:
      {
	auto	p = reinterpret_cast<const YUV411*>(_buf.data());
	auto	q = _rgb.data();
	for (size_t y = 0; y < height(); ++y)
	{
	    std::copy(make_pixel_iterator(p),
		      make_pixel_iterator(p + width()/2),
		      make_pixel_iterator(q));
	    p += width()/2;
	    q += width();
	}
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(), GDK_RGB_DITHER_NONE,
			   reinterpret_cast<guchar*>(_rgb.data()), 3*width());
      }
	break;

      case RGB_24:
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(), GDK_RGB_DITHER_NONE,
			   _buf.data(), 3*width());
	break;

      case MONO_8:
      case RAW_8:
	if ((bayerTileMapping() != IIDCCamera::YYYY) &&
	    (pixelFormat() != RAW_8))
	    gdk_draw_rgb_image(_canvas->window,
			       _canvas->style
				      ->fg_gc[GTK_WIDGET_STATE(_canvas)],
			       0, 0, width(), height(), GDK_RGB_DITHER_NONE,
			       reinterpret_cast<guchar*>(_rgb.data()),
			       3*width());
	else
	    gdk_draw_gray_image(_canvas->window,
				_canvas->style
				       ->fg_gc[GTK_WIDGET_STATE(_canvas)],
				0, 0, width(), height(), GDK_RGB_DITHER_NONE,
				_buf.data(), width());
	break;

      case MONO_16:
      case SIGNED_MONO_16:
      case RAW_16:
	if ((bayerTileMapping() != IIDCCamera::YYYY) &&
	    (pixelFormat() != RAW_16))
	    gdk_draw_rgb_image(_canvas->window,
			       _canvas->style
				      ->fg_gc[GTK_WIDGET_STATE(_canvas)],
			       0, 0, width(), height(), GDK_RGB_DITHER_NONE,
			       reinterpret_cast<guchar*>(_rgb.data()),
			       3*width());
	else
	{
	    auto	p = reinterpret_cast<const u_short*>(_buf.data());
	    auto	q = _buf.data();
	    if (isLittleEndian())
		for (size_t y = 0; y < height(); ++y)
		    for (size_t x = 0; x < width(); ++x)
			*q++ = *p++;
	    else
		for (size_t y = 0; y < height(); ++y)
		    for (size_t x = 0; x < width(); ++x)
			*q++ = htons(*p++);
	    gdk_draw_gray_image(_canvas->window,
				_canvas->style
				       ->fg_gc[GTK_WIDGET_STATE(_canvas)],
				0, 0, width(), height(),
				GDK_RGB_DITHER_NONE, _buf.data(), width());
	}
	break;

      default:
	break;
    }
}
 
//! バッファ中の画像をsaveする．
/*!
  モノクロ画像はPGM形式で、カラー画像はPPM形式でsaveされる．
  \param out	画像を書き出す出力ストリーム．
  \return	outで指定した出力ストリーム．
*/
std::ostream&
MyIIDCCamera::save(std::ostream& out) const
{
    using namespace	std;
    
    switch (pixelFormat())
    {
      case YUV_444:
      case YUV_422:
      case YUV_411:
	out << "P6" << '\n' << width() << ' ' << height() << '\n' << 255
	    << endl;
	out.write(reinterpret_cast<const char*>(_rgb.data()),
		  3*width()*height());
	break;

      case RGB_24:
	out << "P6" << '\n' << width() << ' ' << height() << '\n' << 255
	    << endl;
	out.write(reinterpret_cast<const char*>(_buf.data()),
		  3*width()*height());
	break;

      case MONO_8:
      case MONO_16:
      case SIGNED_MONO_16:
      case RAW_8:
      case RAW_16:
	if ((bayerTileMapping() != IIDCCamera::YYYY) &&
	    (pixelFormat() != RAW_8) && (pixelFormat() != RAW_16))
	{
	    out << "P6" << '\n' << width() << ' ' << height() << '\n' << 255
		<< endl;
	    out.write(reinterpret_cast<const char*>(_rgb.data()),
		      3*width()*height());
	}
	else
	{
	    if ((pixelFormat() == MONO_16) ||
		(pixelFormat() == SIGNED_MONO_16) ||
		(pixelFormat() == RAW_16))
	    {
		auto	p = reinterpret_cast<const u_short*>(_buf.data());
		auto	q = _buf.data();
		if (isLittleEndian())
		    for (size_t y = 0; y < height(); ++y)
			for (size_t x = 0; x < width(); ++x)
			    *q++ = *p++;
		else
		    for (size_t y = 0; y < height(); ++y)
			for (size_t x = 0; x < width(); ++x)
			    *q++ = htons(*p++);
	    }
	    out << "P5" << '\n' << width() << ' ' << height() << '\n' << 255
		<< endl;
	    out.write(reinterpret_cast<const char*>(_buf.data()),
		      width()*height());
	}
	break;

      default:
	break;
    }
    
    return out;
}
 
//! コマンドボタンのコンテナとその親ウィジェットを記録する
/*!
  フォーマット変更のコールバック用にコマンド類とその親ウィジェットを記録する．
  カメラのフォーマットが変更されると，サポートされる機能も変わる可能性がある．
  \param parent 親ウィジェット
  \param commands コマンド類のコンテナウィジェット
 */
void
MyIIDCCamera::setCommands(GtkWidget* commands, GtkWidget* parent)
{
    _commands  = commands;
    _comParent = parent;
    return;
}

//! 現在のカメラの状態に応じてコマンドのコンテナウィジェットを更新する
/*!
  カメラを制御するためのコマンドボタンを一新する．
  カメラのフォーマットが変更されると，サポートされる機能も変わる可能性がある．
 */
void
MyIIDCCamera::refreshCommands()
{
    const auto	table = _comParent;
    const auto	dead  = _commands;
    assert(table != 0 && (dead != 0));
    _commands = createCommands(*this);
    gtk_table_attach(GTK_TABLE(table), _commands,
		     1, 2, 1, 2, GTK_SHRINK, GTK_SHRINK, 5, 0);
  // commandsはGtkTableの1,2,1,2に配置する
    gtk_widget_show_all(table);
    gtk_widget_destroy(dead);
    return;
}

}
