/*
 * testIIDCcamera: test program controlling an IIDC 1394-based Digital Camera
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
#if HAVE_CONFIG_H
#  include <config.h>
#endif
#include <sys/time.h>
#include <stdexcept>
#include <iostream>
#include "MyIIDCCamera.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
//! ループ10回に要した時間の平均をとることにより，フレームレートを測定する．
static void
countTime(int& nframes, struct timeval& start)
{
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
  \param userdata	MyIIDCCamera (IEEE1394カメラ)
  \return		TRUEを返す
*/
static gboolean
CBexpose(GtkWidget* widget, GdkEventExpose* event, gpointer userdata)
{
    MyIIDCCamera*	camera = (MyIIDCCamera*)userdata;
    camera->draw();
    return TRUE;
}

/************************************************************************
*  class MyIIDCCamera							*
************************************************************************/
//! IEEE1394カメラノードを生成する
/*!
  \param uniqId	個々のカメラ固有の64bit ID．同一のIEEE1394 busに
		複数のカメラが接続されている場合，これによって
		同定を行う．
  \param speed	データ転送速度
*/
MyIIDCCamera::MyIIDCCamera(u_int64_t uniqId, Speed speed)
    :IIDCCamera(IIDCCamera::Monocular, uniqId, speed),
     _canvas(gtk_drawing_area_new()),
     _buf(0),
     _rgb(0)
{
    gdk_rgb_init();
    gtk_signal_connect(GTK_OBJECT(_canvas), "expose_event",
		       GTK_SIGNAL_FUNC(CBexpose), (gpointer)this);
    
  // 現在のカメラのフォーマットに合わせてバッファの確保を行う．
    setFormatAndFrameRate(getFormat(), getFrameRate());
}

//! IEEE1394カメラオブジェクトを破壊する
MyIIDCCamera::~MyIIDCCamera()
{
    delete [] _rgb;
    delete [] _buf;
}

//! 画像フォーマットとフレームレートを指定する．
/*!
  さらに入力画像バッファとRGBバッファを確保し直し，canvasの大きさを変更する．
  \param format	設定したい画像フォーマット．
  \param rate	設定したいフレームレート．
  \return	このIEEE1394カメラオブジェクト．
*/
IIDCCamera&
MyIIDCCamera::setFormatAndFrameRate(Format format, FrameRate rate)
{
  // IEEE1394カメラに対して画像フォーマットとフレームレートを指定する．
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
    delete [] _rgb;
    delete [] _buf;
    _buf = new u_char[buffSize];
    _rgb = new RGB[width() * height()];

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
    static int			nframes = 0;
    static struct timeval	start;
    countTime(nframes, start);

  // IEEE1394Camera から画像データを読み込む．
    if (bayerTileMapping() != IIDCCamera::YYYY &&
	((pixelFormat() == MONO_8)  ||
	 (pixelFormat() == MONO_16) || (pixelFormat() == SIGNED_MONO_16)))
	snap().captureBayerRaw(_rgb);
    else
	snap().captureRaw(_buf);
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
	auto	p = reinterpret_cast<const YUV444*>(_buf);
	auto	q = _rgb;
	for (u_int y = 0; y < height(); ++y)
	{
	    std::copy(make_pixel_iterator(p), make_pixel_iterator(p + width()),
		      make_pixel_iterator(q));
	    p += width();
	    q += width();
	}
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(),
			   GDK_RGB_DITHER_NONE, (guchar*)_rgb, 3*width());
      }
	break;

      case YUV_422:
      {
	auto	p = reinterpret_cast<const YUV422*>(_buf);
	auto	q = _rgb;
	for (u_int y = 0; y < height(); ++y)
	{
	    std::copy(make_pixel_iterator(p), make_pixel_iterator(p + width()),
		      make_pixel_iterator(q));
	    p += width();
	    q += width();
	}
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(),
			   GDK_RGB_DITHER_NONE, (guchar*)_rgb, 3*width());
      }
	break;

      case YUV_411:
      {
	auto	p = reinterpret_cast<const YUV411*>(_buf);
	auto	q = _rgb;
	for (u_int y = 0; y < height(); ++y)
	{
	    std::copy(make_pixel_iterator(p),
		      make_pixel_iterator(p + width()/2),
		      make_pixel_iterator(q));
	    p += width()/2;
	    q += width();
	}
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(),
			   GDK_RGB_DITHER_NONE, (guchar*)_rgb, 3*width());
      }
	break;

      case RGB_24:
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(),
			   GDK_RGB_DITHER_NONE, (guchar*)_buf, 3*width());
	break;

      case MONO_8:
      case RAW_8:
	if ((bayerTileMapping() != IIDCCamera::YYYY) && (pixelFormat() != RAW_8))
	    gdk_draw_rgb_image(_canvas->window,
			       _canvas->style
				      ->fg_gc[GTK_WIDGET_STATE(_canvas)],
			       0, 0, width(), height(),
			       GDK_RGB_DITHER_NONE, (guchar*)_rgb, 3*width());
	else
	    gdk_draw_gray_image(_canvas->window,
				_canvas->style
				       ->fg_gc[GTK_WIDGET_STATE(_canvas)],
				0, 0, width(), height(),
				GDK_RGB_DITHER_NONE, (guchar*)_buf, width());
	break;

      case MONO_16:
      case SIGNED_MONO_16:
      case RAW_16:
	if ((bayerTileMapping() != IIDCCamera::YYYY) && (pixelFormat() != RAW_16))
	    gdk_draw_rgb_image(_canvas->window,
			       _canvas->style
				      ->fg_gc[GTK_WIDGET_STATE(_canvas)],
			       0, 0, width(), height(),
			       GDK_RGB_DITHER_NONE, (guchar*)_rgb, 3*width());
	else
	{
	    const u_short*	p = (u_short*)_buf;
	    u_char*		q = _buf;
	    if (isLittleEndian())
		for (u_int y = 0; y < height(); ++y)
		    for (u_int x = 0; x < width(); ++x)
			*q++ = *p++;
	    else
		for (u_int y = 0; y < height(); ++y)
		    for (u_int x = 0; x < width(); ++x)
			*q++ = htons(*p++);
	    gdk_draw_gray_image(_canvas->window,
				_canvas->style
				       ->fg_gc[GTK_WIDGET_STATE(_canvas)],
				0, 0, width(), height(),
				GDK_RGB_DITHER_NONE, (guchar*)_buf, width());
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
	out.write((const char*)_rgb, 3*width()*height());
	break;

      case RGB_24:
	out << "P6" << '\n' << width() << ' ' << height() << '\n' << 255
	    << endl;
	out.write((const char*)_buf, 3*width()*height());
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
	    out.write((const char*)_rgb, 3*width()*height());
	}
	else
	{
	    if ((pixelFormat() == MONO_16) ||
		(pixelFormat() == SIGNED_MONO_16) ||
		(pixelFormat() == RAW_16))
	    {
		const u_short*	p = (u_short*)_buf;
		u_char*		q = _buf;
		if (isLittleEndian())
		    for (u_int y = 0; y < height(); ++y)
			for (u_int x = 0; x < width(); ++x)
			    *q++ = *p++;
		else
		    for (u_int y = 0; y < height(); ++y)
			for (u_int x = 0; x < width(); ++x)
			    *q++ = htons(*p++);
	    }
	    out << "P5" << '\n' << width() << ' ' << height() << '\n' << 255
		<< endl;
	    out.write((const char*)_buf, width()*height());
	}
	break;

      default:
	break;
    }
    
    return out;
}
 
}
