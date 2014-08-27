/*
 *  $Id: MyV4L2Camera.cc,v 1.14 2012-08-29 19:35:49 ueshiba Exp $
 */
#include <sys/time.h>
#include <stdexcept>
#include <iostream>
#include "MyV4L2Camera.h"

namespace TU
{
/************************************************************************
*  YUV -> RGB conversion stuffs						*
************************************************************************/
static int	tbl_r [256];
static int	tbl_g0[256];
static int	tbl_g1[256];
static int	tbl_b [256];

inline int	flt2fix(float flt)	{return int(flt * (1 << 10));}
inline int	fix2int(int fix)	{return fix >> 10;}

//! YUV -> RGB 変換テーブルの初期化．
static void
initialize_tbl()
{
    for (int i = 0; i < 256; ++i)
    {
	tbl_r [i] = int(1.4022f * (i - 128));
	tbl_g0[i] = flt2fix(0.7144f * (i - 128));
	tbl_g1[i] = flt2fix(0.3457f * (i - 128));
	tbl_b [i] = int(1.7710f * (i - 128));
    }
}
    
//! YUV -> RGB 変換を行う．
inline MyRGB
yuv2rgb(u_char y, u_char u, u_char v)
{
    int		tmp;
    MyRGB	val;
    tmp   = y + tbl_r[v];
    val.r = (tmp > 255 ? 255 : tmp < 0 ? 0 : tmp);
    tmp   = y - fix2int(tbl_g0[v] + tbl_g1[u]);
    val.g = (tmp > 255 ? 255 : tmp < 0 ? 0 : tmp);
    tmp   = y + tbl_b[u];
    val.b = (tmp > 255 ? 255 : tmp < 0 ? 0 : tmp);
    
    return val;
}

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
  \param userdata	MyV4L2Camera (V4L2カメラ)
  \return		TRUEを返す
*/
static gboolean
CBexpose(GtkWidget* widget, GdkEventExpose* event, gpointer userdata)
{
    MyV4L2Camera*	camera = (MyV4L2Camera*)userdata;
    camera->draw();
    return TRUE;
}

/************************************************************************
*  class MyV4L2Camera							*
************************************************************************/
//! V4L2カメラノードを生成する
/*!
  \param dev	カメラのデバイス名
*/
MyV4L2Camera::MyV4L2Camera(const char* dev)
    :V4L2Camera(dev),
     _canvas(gtk_drawing_area_new()),
     _buf(0),
     _rgb(0)
{
    initialize_tbl();			// YUV -> RGB 変換テーブルの初期化．
    gdk_rgb_init();
    gtk_signal_connect(GTK_OBJECT(_canvas), "expose_event",
		       GTK_SIGNAL_FUNC(CBexpose), (gpointer)this);
    
  // 現在のカメラのフォーマットに合わせてバッファの確保を行う．
    u_int	fps_n, fps_d;
    getFrameRate(fps_n, fps_d);
    setFormat(pixelFormat(), width(), height(), fps_n, fps_d);
}

//! V4L2カメラオブジェクトを破壊する
MyV4L2Camera::~MyV4L2Camera()
{
    delete [] _rgb;
    delete [] _buf;
}

//! 画像フォーマットとフレームレートを指定する．
/*!
  さらに入力画像バッファとRGBバッファを確保し直し，canvasの大きさを変更する．
  \param format	設定したい画像フォーマット．
  \param rate	設定したいフレームレート．
  \return	このV4L2カメラオブジェクト．
*/
V4L2Camera&
MyV4L2Camera::setFormat(PixelFormat pixFmt,
			size_t w, size_t h, u_int fps_n, u_int fps_d)
{
  // V4L2カメラに対して画像フォーマットとフレームレートを指定する．
    V4L2Camera::setFormat(pixFmt, w, h, fps_n, fps_d);

  // 指定したフォーマットに合わせて入力画像バッファとRGBバッファを再確保する．
    size_t	buffSize;
    switch (pixelFormat())
    {
      case RGB32:
	buffSize = width() * height() * 4;
	break;
      case RGB24:
	buffSize = width() * height() * 3;
	break;
      case Y16:
      case YUYV:
      case UYVY:
	buffSize = width() * height() * 2;
	break;
      case GREY:
      case SBGGR8:
      case SGBRG8:
      case SGRBG8:
	buffSize = width() * height();
	break;
      default:
	throw std::invalid_argument("Unsupported pixel format!!");
	break;
    }
    delete [] _rgb;
    delete [] _buf;
    _buf = new u_char[buffSize];
    _rgb = new MyRGB[width() * height()];

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
MyV4L2Camera::idle()
{
  // フレームレートの測定．
    static int			nframes = 0;
    static struct timeval	start;
    countTime(nframes, start);

  // V4L2Camera から画像データを読み込む．
    if ((pixelFormat() == SBGGR8) ||
	(pixelFormat() == SGBRG8) || (pixelFormat() == SGRBG8))
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
MyV4L2Camera::draw()
{
  // 必用なら YUV -> RGB の変換を行ってから画像を表示．
    switch (pixelFormat())
    {
      case UYVY:
      {
	const u_char*	p = _buf;
	MyRGB*		q = _rgb;
	for (size_t y = 0; y < height(); ++y)
	    for (size_t x = 0; x < width(); x += 2)
	    {
		*q++ = yuv2rgb(p[1], p[0], p[2]);	// Y0, U, V
		*q++ = yuv2rgb(p[3], p[0], p[2]);	// Y1, U, V
		p += 4;
	    }
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(),
			   GDK_RGB_DITHER_NONE, (guchar*)_rgb, 3*width());
      }
	break;
      case YUYV:
      {
	const u_char*	p = _buf;
	MyRGB*		q = _rgb;
	for (size_t y = 0; y < height(); ++y)
	    for (size_t x = 0; x < width(); x += 2)
	    {
		*q++ = yuv2rgb(p[0], p[1], p[3]);	// Y0, U, V
		*q++ = yuv2rgb(p[2], p[1], p[3]);	// Y1, U, V
		p += 4;
	    }
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(),
			   GDK_RGB_DITHER_NONE, (guchar*)_rgb, 3*width());
      }
	break;
      case RGB24:
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(),
			   GDK_RGB_DITHER_NONE, (guchar*)_buf, 3*width());
	break;
      case SBGGR8:
      case SGBRG8:
      case SGRBG8:
	gdk_draw_rgb_image(_canvas->window,
			   _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			   0, 0, width(), height(),
			   GDK_RGB_DITHER_NONE, (guchar*)_rgb, 3*width());
	break;
      case RGB32:
	gdk_draw_rgb_32_image(_canvas->window,
			      _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			      0, 0, width(), height(),
			      GDK_RGB_DITHER_NONE, (guchar*)_buf, 4*width());
	break;
      case GREY:
	gdk_draw_gray_image(_canvas->window,
			    _canvas->style->fg_gc[GTK_WIDGET_STATE(_canvas)],
			    0, 0, width(), height(),
			    GDK_RGB_DITHER_NONE, (guchar*)_buf, width());
	break;
      case Y16:
      {
	const u_short*	p = (u_short*)_buf;
	u_char*		q = _buf;
	for (size_t y = 0; y < height(); ++y)
	    for (size_t x = 0; x < width(); ++x)
		*q++ = *p++;
	gdk_draw_gray_image(_canvas->window,
			    _canvas->style
			    ->fg_gc[GTK_WIDGET_STATE(_canvas)],
			    0, 0, width(), height(),
			    GDK_RGB_DITHER_NONE, (guchar*)_buf, width());
      }
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
MyV4L2Camera::save(std::ostream& out) const
{
    using namespace	std;
    
    switch (pixelFormat())
    {
      case UYVY:
      case YUYV:
	out << "P6" << '\n' << width() << ' ' << height() << '\n' << 255
	    << endl;
	out.write((const char*)_rgb, 3*width()*height());
	break;

      case RGB24:
      case SBGGR8:
      case SGBRG8:
      case SGRBG8:
	out << "P6" << '\n' << width() << ' ' << height() << '\n' << 255
	    << endl;
	out.write((const char*)_buf, 3*width()*height());
	break;

      case GREY:
      case Y16:
      {
	  out << "P5" << '\n' << width() << ' ' << height() << '\n' << 255
	      << endl;
	  out.write((const char*)_buf, width()*height());
      }
	break;
    }
    
    return out;
}
 
}
