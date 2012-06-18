/*
 *  $Id: V4L2++.h,v 1.1 2012-06-18 08:21:22 ueshiba Exp $
 */
/*!
  \mainpage	libTUV4L2++ - Video for Linux v.2デバイスを制御するC++ライブラリ
  \anchor	libTUV4L2
*/
#ifndef __TUV4L2PP_h

#include <asm/types.h>		// for videodev2.h
#include <linux/videodev2.h>
#include <vector>
#include <string>
#include <boost/iterator_adaptors.hpp>
#ifdef HAVE_LIBTUTOOLS__
#  include "TU/Image++.h"
#endif

/*!
  \namespace	TU
  \brief	本ライブラリで定義されたクラスおよび関数を収める名前空間
*/
namespace TU
{
/************************************************************************
*  class V4L2Camera							*
************************************************************************/
//! Video for Linux v.2 に対応するカメラを表すクラス
class V4L2Camera
{
  public:
    enum PixelFormat		//! 出力画像の画素の形式
    {
	BGR24	= V4L2_PIX_FMT_BGR24,	//!< 24 bits/pix, BGR-8-8-8
	RGB24	= V4L2_PIX_FMT_RGB24,	//!< 24 bits/pix, RGB-8-8-8
	BGR32	= V4L2_PIX_FMT_BGR32,	//!< 32 bits/pix, BGR-8-8-8-8
	RGB32	= V4L2_PIX_FMT_RGB32,	//!< 32 bits/pix, RGB-8-8-8-8

	GREY	= V4L2_PIX_FMT_GREY,	//!<  8 bits/pix, Greyscale
	Y16	= V4L2_PIX_FMT_Y16,	//!< 16 bits/pix, Greyscale

	UYVY	= V4L2_PIX_FMT_UYVY,	//!< 16 bits/pix, YUV 4:2:2

	SBGGR8	= V4L2_PIX_FMT_SBGGR8,	//!<  8 bits/pix, BGGR bayer pattern
	SGBRG8	= V4L2_PIX_FMT_SGBRG8,	//!<  8 bits/pix, GBRG bayer pattern
	SGRBG8	= V4L2_PIX_FMT_SGRBG8,	//!<  8 bits/pix, GRBG bayer pattern

	UNKNOWN_PIXEL_FORMAT = v4l2_fourcc('U', 'K', 'N', 'W')
    };

    enum Feature		//! カメラの属性
    {
	BRIGHTNESS			= V4L2_CID_BRIGHTNESS,
	BRIGHTNESS_AUTO			= V4L2_CID_AUTOBRIGHTNESS,
	CONTRAST			= V4L2_CID_CONTRAST,
	GAIN				= V4L2_CID_GAIN,
	SATURATION			= V4L2_CID_SATURATION,
	HUE				= V4L2_CID_HUE,
	HUE_AUTO			= V4L2_CID_HUE_AUTO,
	GAMMA				= V4L2_CID_GAMMA,
	SHARPNESS			= V4L2_CID_SHARPNESS,
	WHITE_BALANCE_TEMPERATURE	= V4L2_CID_WHITE_BALANCE_TEMPERATURE,
	WHITE_BALANCE_AUTO		= V4L2_CID_AUTO_WHITE_BALANCE,
	BACKLIGHT_COMPENSATION		= V4L2_CID_BACKLIGHT_COMPENSATION,
	POWER_LINE_FREQUENCY		= V4L2_CID_POWER_LINE_FREQUENCY,
	EXPOSURE_AUTO			= V4L2_CID_EXPOSURE_AUTO,
	EXPOSURE_AUTO_PRIORITY		= V4L2_CID_EXPOSURE_AUTO_PRIORITY,
	EXPOSURE_ABSOLUTE		= V4L2_CID_EXPOSURE_ABSOLUTE,
	FOCUS_ABSOLUTE			= V4L2_CID_FOCUS_ABSOLUTE,
	FOCUS_RELATIVE			= V4L2_CID_FOCUS_RELATIVE,
	FOCUS_AUTO			= V4L2_CID_FOCUS_AUTO,
	ZOOM_ABSOLUTE			= V4L2_CID_ZOOM_ABSOLUTE,
	ZOOM_RELATIVE			= V4L2_CID_ZOOM_RELATIVE,
	ZOOM_CONTINUOUS			= V4L2_CID_ZOOM_CONTINUOUS,
	PAN_ABSOLUTE			= V4L2_CID_PAN_ABSOLUTE,
	PAN_RELATIVE			= V4L2_CID_PAN_RELATIVE,
	PAN_RESET			= V4L2_CID_PAN_RESET,
	TILT_ABSOLUTE			= V4L2_CID_TILT_ABSOLUTE,
	TILT_RELATIVE			= V4L2_CID_TILT_RELATIVE,
	TILT_RESET			= V4L2_CID_TILT_RESET,

	UNKNOWN_FEATURE
    };

    template <class T>
    struct Range		//! 値の範囲
    {
	bool	involves(T val) const
		{
		    return (min <= val && val <= max &&
			    (val - min) % step == 0);
		}

	T	min;				//!< 最小値
	T	max;				//!< 最大値
	T	step;				//!< 増分ステップ
    };

    struct FrameRate		//! フレームレート
    {
	Range<u_int>		fps_n;		//!< 分子
	Range<u_int>		fps_d;		//!< 分母
    };
    typedef std::vector<FrameRate>::const_iterator	FrameRateIterator;

    struct FrameSize		//! 画像の大きさ
    {
	std::pair<FrameRateIterator, FrameRateIterator>
				availableFrameRates()		const	;
	
	Range<u_int>		width;		//!< 画像の幅
	Range<u_int>		height;		//!< 画像の高さ
	std::vector<FrameRate>	frameRates;	//!< フレーム間隔
    };
    typedef std::vector<FrameSize>::const_iterator	FrameSizeIterator;

    struct MenuItem		//! メニュー項目
    {
	int			index;		//!< メニュー項目の識別子
	std::string		name;		//!< メニュー項目名
    };
    typedef std::vector<MenuItem>::const_iterator	MenuItemIterator;
    
  private:
    struct Format		//! 画像フォーマット
    {
	PixelFormat		pixelFormat;	//!< 画素フォーマット
	std::string		name;		//!< 画素フォーマット名
	std::vector<FrameSize>	frameSizes;	//!< 画像の大きさ
    };

    struct Control		//! コントロール
    {
      public:
	Feature			feature;	//!< コントロールの識別子
	std::string		name;		//!< コントロール名
	v4l2_ctrl_type		type;		//!< 値の型
	Range<int>		range;		//!< 値の範囲
	int			def;		//!< デフォルト値
	u_int			flags;
	std::vector<MenuItem>	menuItems;
    };

    class Buffer	//! 受信用バッファ
    {
      public:
	Buffer()	:_p(0), _size(0)		{}
	~Buffer()					;

	void		map(int fd, u_int index)	;
	const void*	p()			const	{return _p;}
	u_int		size()			const	{return _size;}
	
      private:
	void*		_p;
	u_int		_size;
    };

    template <class S, class T>
    struct MemberIterator
	: public boost::iterator_adaptor<MemberIterator<S, T>,
		typename std::vector<T>::const_iterator, const S>
    {
	MemberIterator(typename std::vector<T>::const_iterator iter)
	    :boost::iterator_adaptor<MemberIterator<S, T>,
		typename std::vector<T>::const_iterator, const S>(iter)	{}
	const S&	dereference()				 const	;
    };

  public:
    typedef MemberIterator<PixelFormat, Format>		PixelFormatIterator;
    typedef MemberIterator<Feature, Control>		FeatureIterator;
    
  public:
    V4L2Camera(const char* deviceName)					;
    ~V4L2Camera()							;

  // Format stuffs.
    std::pair<PixelFormatIterator, PixelFormatIterator>
			availablePixelFormats()			const	;
    std::pair<FrameSizeIterator, FrameSizeIterator>
			availableFrameSizes(PixelFormat pixelFormat)
								const	;
    bool		isAvailable(PixelFormat pixelFormat)	const	;
    V4L2Camera&		setFormat(PixelFormat pixelFormat,
				  u_int width, u_int height,
				  u_int fps_n, u_int fps_d)		;
    void		getFrameRate(u_int& fps_n, u_int& fps_d) const	;
    u_int		width()					const	;
    u_int		height()				const	;
    PixelFormat		pixelFormat()				const	;
    std::ostream&	put(std::ostream& out,
			    PixelFormat pixelFormat)		const	;
  
  // Feature stuffs.
    std::pair<FeatureIterator, FeatureIterator>
			availableFeatures()			const	;
    std::pair<MenuItemIterator, MenuItemIterator>
			availableMenuItems(Feature feature)	const	;
    bool		isAvailable(Feature feature)		const	;
    V4L2Camera&		setValue(Feature feature, int value)		;
    int			getValue(Feature feature)		const	;
    void		getMinMaxStep(Feature feature, int& min,
				      int& max, int& step)	const	;
    int			getDefaultValue(Feature feature)	const	;
    const std::string&	getName(Feature feature)		const	;
    std::ostream&	put(std::ostream& out, Feature feature)	const	;
    
  // Capture stuffs.
    V4L2Camera&		continuousShot()				;
    V4L2Camera&		stopContinuousShot()				;
    V4L2Camera&		snap()						;
#ifdef HAVE_LIBTUTOOLS__
    template <class T> const V4L2Camera&
			operator >>(Image<T>& image)		const	;
    template <class T> const V4L2Camera&
			captureRGBImage(Image<T>& image)	const	;
    template <class T> const V4L2Camera&
			captureDirectly(Image<T>& image)	const	;
#endif
    const V4L2Camera&	captureRaw(void* image)			const	;
    const V4L2Camera&	captureBayerRaw(void* image)		const	;

  // Utility functions.
    static PixelFormat	uintToPixelFormat(u_int pixelFormat)		;
    static Feature	uintToFeature(u_int feature)			;
    
  private:
    void		enumerateFormats()				;
    void		enumerateControls()				;
    int			enumerateMenuItems(
			    const v4l2_queryctrl& ctrl,
			    std::vector<MenuItem>& menuItems)		;
    const Format&	pixelFormatToFormat(PixelFormat pixelFormat)
								const	;
    const Control&	featureToControl(Feature feature)	const	;

    void		mapBuffers(u_int n)				;
    void		unmapBuffers()					;
    u_int		requestBuffers(u_int n)				;
    void		enqueueBuffer(u_int index)		const	;
    u_int		dequeueBuffer()				const	;
    
    int			ioctl(int request, void* arg)		const	;
    int			ioctl(int id, v4l2_queryctrl& ctrl)	const	;

    friend std::ostream&
	operator <<(std::ostream& out, const Format& format)		;
    friend std::ostream&
	operator <<(std::ostream& out, const Control& control)		;
    
  private:
    const int			_fd;
    std::vector<Format>		_formats;
    std::vector<Control>	_controls;
    u_int			_width;
    u_int			_height;
    PixelFormat			_pixelFormat;
    std::vector<Buffer>		_buffers;
    u_int			_current;	// キューから取り出されている
};

//! このカメラで利用できる画素フォーマットの範囲を返す
/*!
  \return	画素フォーマット(#PixelFormat)を指す定数反復子のペア
*/
inline std::pair<V4L2Camera::PixelFormatIterator,
		 V4L2Camera::PixelFormatIterator>
V4L2Camera::availablePixelFormats() const
{
    return std::make_pair(_formats.begin(), _formats.end());
}
    
//! 指定した画素フォーマットのもとでこのカメラで利用できる画像サイズの範囲を返す
/*!
  \param pixelFormat	画素フォーマット
  \return		画像サイズ(#FrameSize)を指す定数反復子のペア
*/
inline std::pair<V4L2Camera::FrameSizeIterator,
		 V4L2Camera::FrameSizeIterator>
V4L2Camera::availableFrameSizes(PixelFormat pixelFormat) const
{
    const Format&	format = pixelFormatToFormat(pixelFormat);
    return std::make_pair(format.frameSizes.begin(), format.frameSizes.end());
}
    
//! 現在設定されている画像幅を返す
inline u_int
V4L2Camera::width() const
{
    return _width;
}

//! 現在設定されている画像高さを返す
inline u_int
V4L2Camera::height() const
{
    return _height;
}

//! 現在設定されている画素フォーマット(#PixelFormat)を返す
inline V4L2Camera::PixelFormat
V4L2Camera::pixelFormat() const
{
    return _pixelFormat;
}

//! 指定された画素フォーマットの内容を出力する．
/*
  \param out		出力ストリーム
  \param pixelFormat	画素フォーマット
  \return		outで指定した出力ストリーム
*/ 
inline std::ostream&
V4L2Camera::put(std::ostream& out, PixelFormat pixelFormat) const
{
    return out << pixelFormatToFormat(pixelFormat);
}

//! このカメラで利用できる属性の範囲を返す
/*!
  \return	属性(#Feature)を指す定数反復子のペア
*/
inline std::pair<V4L2Camera::FeatureIterator, V4L2Camera::FeatureIterator>
V4L2Camera::availableFeatures() const
{
    return std::make_pair(_controls.begin(), _controls.end());
}
    
//! この属性で利用できるメニュー項目の範囲を返す
/*!
  \return	メニュー項目(#MenuItem)を指す定数反復子のペア
*/
inline std::pair<V4L2Camera::MenuItemIterator, V4L2Camera::MenuItemIterator>
V4L2Camera::availableMenuItems(Feature feature) const
{
    const Control&	control = featureToControl(feature);
    return std::make_pair(control.menuItems.begin(), control.menuItems.end());
}

//! 指定された属性の内容を出力する
/*
  \param out		出力ストリーム
  \param feature	属性
  \return		outで指定した出力ストリーム
*/ 
inline std::ostream&
V4L2Camera::put(std::ostream& out, Feature feature) const
{
    return out << featureToControl(feature);
}

//! 指定された属性のデフォルト値を調べる
/*!
  \param feature	対象となる属性
  \return		デフォルト値
*/
inline int
V4L2Camera::getDefaultValue(Feature feature) const
{
    return featureToControl(feature).def;
}

inline const std::string&
V4L2Camera::getName(Feature feature) const
{
    return featureToControl(feature).name;
}

//! カメラから出力される最初の画像を保持する
/*!
  カメラからの画像出力は, continuousShot() によって行われる. 実際に画像データが
  受信されるまで, 本関数は呼び出し側に制御を返さない. 
  \return	このカメラオブジェクト
 */
inline V4L2Camera&
V4L2Camera::snap()
{
  //if (_current != ~0)			// 以前に受信したバッファがあれば...
    _current = dequeueBuffer();		// データが受信されるのを待つ
    enqueueBuffer(_current);	// キューに戻す
    return *this;
}

#ifdef HAVE_LIBTUTOOLS__
//! カメラから出力された画像を直接的に取り込む
/*!
  #operator >>() との違いは, 画像形式の変換を行わないことと, Image<T> 構造体
  の中のデータ領域へのポインタをV4L2入力バッファへのポインタに書き換えることに
  よって, 実際にはデータのコピーを行わないことである. テンプレートパラメータTは,
  格納先の画像の画素形式を表す. なお, 本関数を呼び出す前に snap() によって
  カメラからの画像を保持しておかなければならない. 
  \param image	画像データを格納する画像オブジェクト. 画像の幅と高さは, 
		現在カメラに設定されている画像サイズに合わせて自動的に
		設定される. 
  \return	このカメラオブジェクト
*/
template <class T> const V4L2Camera&
V4L2Camera::captureDirectly(Image<T>& image) const
{
    if (_current == ~0)
	throw std::runtime_error("V4L2Camera::captureDirectly(): no images snapped!!");
    image.resize((T*)_buffers[_current].p(), height(), width());

    return *this;
}
#endif

template <> inline const V4L2Camera::PixelFormat&
V4L2Camera::MemberIterator<V4L2Camera::PixelFormat,
			   V4L2Camera::Format>::dereference() const
{
    return base_reference()->pixelFormat;
}

template <> inline const V4L2Camera::Feature&
V4L2Camera::MemberIterator<V4L2Camera::Feature,
			   V4L2Camera::Control>::dereference() const
{
    return base_reference()->feature;
}

//! 指定した画像サイズのもとでこのカメラで利用できるフレームレートの範囲を返す
/*!
  \return		フレームレート(#FrameRate)を指す定数反復子のペア
*/
inline std::pair<V4L2Camera::FrameRateIterator,
		 V4L2Camera::FrameRateIterator>
V4L2Camera::FrameSize::availableFrameRates() const
{
    return std::make_pair(frameRates.begin(), frameRates.end());
}
    
/************************************************************************
*  global functions							*
************************************************************************/
template <class T>
std::ostream&	operator <<(std::ostream& out,
			    const typename V4L2Camera::Range<T>& range)	;
std::ostream&	operator <<(std::ostream& out,
			    const V4L2Camera::FrameSize& frameSize)	;
std::ostream&	operator <<(std::ostream& out,
			    const V4L2Camera::FrameRate& frameRate)	;
std::ostream&	operator <<(std::ostream& out,
			    const V4L2Camera::MenuItem& menuItem)	;
}
#endif //!__TUV4L2PP_h
