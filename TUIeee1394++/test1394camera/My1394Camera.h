/*
 *  $Id: My1394Camera.h,v 1.2 2003-02-20 05:51:50 ueshiba Exp $
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
  /*!
    Bayerパターン -> RGB 変換の種類．
  */
    enum Bayer
    {
	NONE,		//!< 変換しない
	RGGB,		//!< RGGB Bayer pattern から RGB 形式へ
	BGGR		//!< BGGR Bayer pattern から RGB 形式へ
    };
    
    My1394Camera(Ieee1394Port& port, u_int64 uniqId)	;
    ~My1394Camera()					;

  //! 画像の表示領域となるキャンバスを返す．
    GtkWidget*		canvas()		const	{return _canvas;}
    Ieee1394Camera&	setFormatAndFrameRate(Format format,
					      FrameRate rate)	;
    void		idle()					;
    void		draw()					;
    Bayer		getBayer()			const	;
    void		setBayer(Bayer bayer)			;
    std::ostream&	save(std::ostream& out)		const	;
    
  private:
    GtkWidget* const	_canvas;	// 画像を表示する領域
    u_char*		_buf;		// 入力画像用バッファ
    MyRGB*		_rgb;		// RGB カラー画像(表示用)バッファ
    Bayer		_bayer;		// Bayer -> RGB 変換の種類を示すflag
};
 
//! 現在設定されているBayerパターン -> RGB 変換の種類を返す．
/*!
  \return	変換の種類．
*/
inline My1394Camera::Bayer
My1394Camera::getBayer() const
{
    return _bayer;
}

//! Bayerパターン -> RGB 変換の種類を指定する．
/*!
  \param bayer	変換の種類．
*/
inline void
My1394Camera::setBayer(Bayer bayer)
{
    _bayer = bayer;
}

}
