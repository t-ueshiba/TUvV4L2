/*
 *  $Id: My1394Camera.h,v 1.3 2003-02-27 03:48:13 ueshiba Exp $
 */
#include <gtk/gtk.h>
#include "TU/Ieee1394++.h"

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
*  class My1394Camera							*
************************************************************************/
/*!
  IEEE1394デジタルカメラを表すクラス．さらに，GTK+ を用いた画像表示のための
  canvas (GTK+ の drawing area widget)，入力画像バッファ，RGB カラー画像
  表示用のバッファを確保する機能を持つ．
*/
class My1394Camera : public Ieee1394Camera
{
  public:
    My1394Camera(Ieee1394Port& port, u_int64 uniqId)			;
    ~My1394Camera()							;

  //! 画像の表示領域となるキャンバスを返す．
    GtkWidget*		canvas()				const	;
    Ieee1394Camera&	setFormatAndFrameRate(Format format,
					      FrameRate rate)		;
    void		idle()						;
    void		draw()						;
    std::ostream&	save(std::ostream& out)			const	;
    
  private:
    GtkWidget* const	_canvas;	// 画像を表示する領域
    u_char*		_buf;		// 入力画像用バッファ
    MyRGB*		_rgb;		// RGB カラー画像(表示用)バッファ
};

inline GtkWidget*
My1394Camera::canvas() const
{
    return _canvas;
}

}
