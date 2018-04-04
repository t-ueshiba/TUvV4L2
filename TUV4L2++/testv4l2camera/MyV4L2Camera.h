/*
 *  $Id: MyV4L2Camera.h,v 1.8 2012-08-29 19:35:49 ueshiba Exp $
 */
#include <gtk/gtk.h>
#include "TU/V4L2++.h"

namespace TU
{
/*!
  RGB カラー画像の画素を表す構造体．
*/
struct MyRGB
{
    u_char	r;	//!< 赤
    u_char	g;	//!< 青
    u_char	b;	//!< 緑
};

/************************************************************************
*  class MyV4L2Camera							*
************************************************************************/
/*!
  Video for Linux .v2 に対応したカメラを表すクラス．さらに，GTK+ を用いた
  画像表示のための canvas (GTK+ の drawing area widget)，入力画像バッファ，
  RGB カラー画像，表示用のバッファを確保する機能を持つ．
*/
class MyV4L2Camera : public V4L2Camera
{
  public:
    MyV4L2Camera(const char* dev)					;
    ~MyV4L2Camera()							;

  //! 画像の表示領域となるキャンバスを返す．
    GtkWidget*		canvas()				const	;
    MyV4L2Camera&	setFormat(PixelFormat pixelFormat,
				  size_t width, size_t height,
				  u_int fps_n, u_int fps_d)		;
    MyV4L2Camera&	setROI(size_t u0, size_t v0,
			       size_t width, size_t height)		;
    void		idle()						;
    void		draw()						;
    std::ostream&	save(std::ostream& out)			const	;

  private:
    void		allocateBuffers()				;
    
  private:
    GtkWidget* const	_canvas;	// 画像を表示する領域
    u_char*		_buf;		// 入力画像用バッファ
    MyRGB*		_rgb;		// RGB カラー画像(表示用)バッファ
};

inline GtkWidget*
MyV4L2Camera::canvas() const
{
    return _canvas;
}

}
