/*
 *  $Id: V4L2Camera.cc,v 1.6 2012-06-29 03:08:51 ueshiba Exp $
 */
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <iomanip>
#include <stdexcept>
#include <boost/foreach.hpp>
#include "TU/V4L2++.h"

#define XY_YZ(X, Y, Z)						\
{								\
    rgb->X = *buf;						\
    rgb->Y = (u_int(*(buf+1)) + u_int(*nxt)) >> 1;		\
    rgb->Z = *(nxt+1);						\
    ++rgb;							\
    ++buf;							\
    ++nxt;							\
}

#define YX_ZY(X, Y, Z)						\
{								\
    rgb->X = *(buf+1);						\
    rgb->Y = (u_int(*buf) + u_int(*(nxt+1))) >> 1;		\
    rgb->Z = *nxt;						\
    ++rgb;							\
    ++buf;							\
    ++nxt;							\
}

#define XYX_YZY_XYX(X, Y, Z)					\
{								\
    rgb->X = (u_int(*(prv-1)) + u_int(*(prv+1)) +		\
	       u_int(*(nxt-1)) + u_int(*(nxt+1))) >> 2;		\
    rgb->Y = (u_int(*(buf-1)) +u_int(*(buf+1)) +		\
	       u_int(*prv) + u_int(*nxt)) >> 2;			\
    rgb->Z = *buf;						\
    ++rgb;							\
    ++prv;							\
    ++buf;							\
    ++nxt;							\
}

#define yXy_ZYZ_yXy(X, Y, Z)					\
{								\
    rgb->X = (u_int(*prv) + u_int(*nxt)) >> 1;			\
    rgb->Y = *buf;						\
    rgb->Z = (u_int(*(buf-1)) + u_int(*(buf+1))) >> 1;		\
    ++rgb;							\
    ++prv;							\
    ++buf;							\
    ++nxt;							\
}

namespace TU
{
/************************************************************************
*  static data								*
************************************************************************/
static const int	CONTROL_IO_ERROR_RETRIES = 2;
static const int	NB_BUFFERS		 = 4;
static const struct
{
    V4L2Camera::Feature	feature;
    const char*		name;
} features[] =
{
    {V4L2Camera::BRIGHTNESS,			"BRIGHTNESS"},
    {V4L2Camera::BRIGHTNESS_AUTO,		"BRIGHTNESS_AUTO"},
    {V4L2Camera::CONTRAST,			"CONTRAST"},
    {V4L2Camera::GAIN,				"GAIN"},
    {V4L2Camera::SATURATION,			"SATURATION"},
    {V4L2Camera::HUE,				"HUE"},
    {V4L2Camera::HUE_AUTO,			"HUE_AUTO"},
    {V4L2Camera::GAMMA,				"GAMMA"},
    {V4L2Camera::SHARPNESS,			"SHARPNESS"},
    {V4L2Camera::WHITE_BALANCE_TEMPERATURE,	"WHITE_BALANCE_TEMPERATURE"},
    {V4L2Camera::WHITE_BALANCE_AUTO,		"WHITE_BALANCE_AUTO"},
    {V4L2Camera::BACKLIGHT_COMPENSATION,	"BACKLIGHT_COMPENSATION"},
    {V4L2Camera::POWER_LINE_FREQUENCY,		"POWER_LINE_FREQUENCY"},
    {V4L2Camera::EXPOSURE_AUTO,			"EXPOSURE_AUTO"},
    {V4L2Camera::EXPOSURE_AUTO_PRIORITY,	"EXPOSURE_AUTO_PRIORITY"},
    {V4L2Camera::EXPOSURE_ABSOLUTE,		"EXPOSURE_ABSOLUTE"},
    {V4L2Camera::FOCUS_ABSOLUTE,		"FOCUS_ABSOLUTE"},
    {V4L2Camera::FOCUS_RELATIVE,		"FOCUS_RELATIVE"},
    {V4L2Camera::FOCUS_AUTO,			"FOCUS_AUTO"},
    {V4L2Camera::ZOOM_ABSOLUTE,			"ZOOM_ABSOLUTE"},
    {V4L2Camera::ZOOM_RELATIVE,			"ZOOM_RELATIVE"},
    {V4L2Camera::ZOOM_CONTINUOUS,		"ZOOM_CONTINUOUS"},
    {V4L2Camera::PAN_ABSOLUTE,			"PAN_ABSOLUTE"},
    {V4L2Camera::PAN_RELATIVE,			"PAN_RELATIVE"},
    {V4L2Camera::PAN_RESET,			"PAN_RESET"},
    {V4L2Camera::TILT_ABSOLUTE,			"TILT_ABSOLUTE"},
    {V4L2Camera::TILT_RELATIVE,			"TILT_RELATIVE"},
    {V4L2Camera::TILT_RESET,			"TILT_RESET"}
};
static const int	NFEATURES = sizeof(features) / sizeof(features[0]);

/************************************************************************
*  static functions							*
************************************************************************/
template <class S, class T> static const S*
bayerRGGB2x2(const S* buf, T* rgb, int w)
{
    const S*	nxt = buf + w;			// next line
    while ((w -= 2) > 0)
    {
	XY_YZ(r, g, b)
	YX_ZY(r, g, b)
    }
    XY_YZ(r, g, b)
    --buf;
    --nxt;
    XY_YZ(r, g, b)
    
    return buf + 1;
}

template <class S, class T> static const S*
bayerRGGBOdd3x3(const S* buf, T* rgb, int w)
{
    const S	*nxt = buf + w;			// next line
    YX_ZY(b, g, r)				// 左端の画素は2x2で処理
    const S	*prv = buf - w;			// previous line
    while ((w -= 2) > 0)			// 奇数行中間の列を処理
    {
	XYX_YZY_XYX(r, g, b)
	yXy_ZYZ_yXy(r, g, b)
    }
    --buf;
    --nxt;
    YX_ZY(b, g, r)				// 右端の画素は2x2で処理
    
    return buf + 1;
}

template <class S, class T> static const S*
bayerRGGBEven3x3(const S* buf, T* rgb, int w)
{
    const S	*nxt = buf + w;			// next line
    XY_YZ(r, g, b)				// 左端の画素は2x2で処理
    const S	*prv = buf - w;			// previous line
    while ((w -= 2) > 0)			// 偶数行中間の列を処理
    {
	yXy_ZYZ_yXy(b, g, r)
	XYX_YZY_XYX(b, g, r)
    }
    --buf;
    --nxt;
    XY_YZ(r, g, b)				// 右端の画素は2x2で処理

    return buf + 1;
}

template <class S, class T> static const S*
bayerBGGR2x2(const S* buf, T* rgb, int w)
{
    const S*	nxt = buf + w;			// next line
    while ((w -= 2) > 0)
    {
	XY_YZ(b, g, r)
	YX_ZY(b, g, r)
    }
    XY_YZ(b, g, r)
    --buf;
    --nxt;
    XY_YZ(b, g, r)

    return buf + 1;
}

template <class S, class T> static const S*
bayerBGGROdd3x3(const S* buf, T* rgb, int w)
{
    const S	*nxt = buf + w;			// next line
    YX_ZY(r, g, b)				// 左端の画素は2x2で処理
    const S	*prv = buf - w;			// previous line
    while ((w -= 2) > 0)			// 奇数行中間の列を処理
    {
	XYX_YZY_XYX(b, g, r)
	yXy_ZYZ_yXy(b, g, r)
    }
    --buf;
    --nxt;
    YX_ZY(r, g, b)				// 右端の画素は2x2で処理

    return buf + 1;
}

template <class S, class T> static const S*
bayerBGGREven3x3(const S* buf, T* rgb, int w)
{
    const S	*nxt = buf + w;			// next line
    XY_YZ(b, g, r)				// 左端の画素は2x2で処理
    const S	*prv = buf - w;			// previous line
    while ((w -= 2) > 0)			// 偶数行中間の列を処理
    {
	yXy_ZYZ_yXy(r, g, b)
	XYX_YZY_XYX(r, g, b)
    }
    --buf;
    --nxt;
    XY_YZ(b, g, r)				// 右端の画素は2x2で処理

    return buf + 1;
}

template <class S, class T> static const S*
bayerGRBG2x2(const S* buf, T* rgb, int w)
{
    const S*	nxt = buf + w;			// next line
    while ((w -= 2) > 0)
    {
	YX_ZY(r, g, b)
	XY_YZ(r, g, b)
    }
    YX_ZY(r, g, b)
    --buf;
    --nxt;
    YX_ZY(r, g, b)

    return buf + 1;
}

template <class S, class T> static const S*
bayerGRBGOdd3x3(const S* buf, T* rgb, int w)
{
    const S	*nxt = buf + w;			// next line
    XY_YZ(b, g, r)				// 左端の画素は2x2で処理
    const S	*prv = buf - w;			// previous line
    while ((w -= 2) > 0)			// 奇数行中間の列を処理
    {
	yXy_ZYZ_yXy(r, g, b)
	XYX_YZY_XYX(r, g, b)
    }
    --buf;
    --nxt;
    XY_YZ(b, g, r)				// 右端の画素は2x2で処理

    return buf + 1;
}

template <class S, class T> static const S*
bayerGRBGEven3x3(const S* buf, T* rgb, int w)
{
    const S	*nxt = buf + w;			// next line
    YX_ZY(r, g, b)				// 左端の画素は2x2で処理
    const S	*prv = buf - w;			// previous line
    while ((w -= 2) > 0)			// 偶数行中間の列を処理
    {
	XYX_YZY_XYX(b, g, r)
	yXy_ZYZ_yXy(b, g, r)
    }
    --buf;
    --nxt;
    YX_ZY(r, g, b)				// 右端の画素は2x2で処理

    return buf + 1;
}

template <class S, class T> static const S*
bayerGBRG2x2(const S* buf, T* rgb, int w)
{
    const S*	nxt = buf + w;			// next line
    while ((w -= 2) > 0)
    {
	YX_ZY(b, g, r)
	XY_YZ(b, g, r)
    }
    YX_ZY(b, g, r)
    --buf;
    --nxt;
    YX_ZY(b, g, r)

    return buf + 1;
}

template <class S, class T> static const S*
bayerGBRGOdd3x3(const S* buf, T* rgb, int w)
{
    const S	*nxt = buf + w;			// next line
    XY_YZ(r, g, b)				// 左端の画素は2x2で処理
    const S	*prv = buf - w;			// previous line
    while ((w -= 2) > 0)			// 奇数行中間の列を処理
    {
	yXy_ZYZ_yXy(b, g, r)
	XYX_YZY_XYX(b, g, r)
    }
    --buf;
    --nxt;
    XY_YZ(r, g, b)				// 右端の画素は2x2で処理

    return buf + 1;
}

template <class S, class T> static const S*
bayerGBRGEven3x3(const S* buf, T* rgb, int w)
{
    const S	*nxt = buf + w;			// next line
    YX_ZY(b, g, r)				// 左端の画素は2x2で処理
    const S	*prv = buf - w;			// previous line
    while ((w -= 2) > 0)			// 偶数行中間の列を処理
    {
	XYX_YZY_XYX(r, g, b)
	yXy_ZYZ_yXy(r, g, b)
    }
    --buf;
    --nxt;
    YX_ZY(b, g, r)				// 右端の画素は2x2で処理

    return buf + 1;
}

static int
v4l2_get_fd(const char* dev)
{
    using namespace	std;
    
    int		fd = ::open(dev, O_RDWR);
    if (fd < 0)
	throw runtime_error(string("TU::v4l2_get_fd(): failed to open ")
			    + dev + "!! " + strerror(errno));
    return fd;
}

/************************************************************************
*  class V4L2Camera							*
************************************************************************/
/*
 *  public member functions
 */
//! Video for Linux v.2 カメラノードを生成する
/*!
  \param dev	デバイス名
*/
V4L2Camera::V4L2Camera(const char* dev)
    :_fd(v4l2_get_fd(dev)), _formats(), _controls(),
     _width(0), _height(0), _pixelFormat(UNKNOWN_PIXEL_FORMAT),
     _buffers(), _current(~0), _inContinuousShot(false)
{
    using namespace	std;

    enumerateFormats();		// 画素フォーマット，画像サイズ，フレームレート
    enumerateControls();	// カメラのコントロール=属性

  // このカメラのどの画素フォーマットも本ライブラリで未サポートならば例外を送出
    PixelFormatRange	pixelFormats = availablePixelFormats();
    if (pixelFormats.first == pixelFormats.second)
	throw runtime_error("V4L2Camera::V4L2Camera(): no available pixel formats!");
    
  // 画素フォーマットと画像サイズおよびフレームレートの現在値を取得
    v4l2_format	fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(VIDIOC_G_FMT, &fmt))
	throw runtime_error(string("V4L2Camera::V4L2Camera(): VIDIOC_G_FMT failed!! ") + strerror(errno));
    _width	 = fmt.fmt.pix.width;
    _height	 = fmt.fmt.pix.height;
    _pixelFormat = uintToPixelFormat(fmt.fmt.pix.pixelformat);
    u_int	fps_n, fps_d;
    getFrameRate(fps_n, fps_d);

  // カメラに現在セットされている画素フォーマットが本ライブラリで未サポートならば，
  // サポートされている1番目の画素フォーマットにセットし，画像サイズをそのフォー
  // マットにおける最大値にする．また，フレームレートをその画像サイズにおける最大値
  // にする．
    if (_pixelFormat == UNKNOWN_PIXEL_FORMAT)
    {
	_pixelFormat = *pixelFormats.first;
	const FrameSize&
	    frameSize = *availableFrameSizes(_pixelFormat).first;
	_width  = frameSize.width.max;
	_height = frameSize.height.max;
	const FrameRate&
	    frameRate = *frameSize.availableFrameRates().first;
	fps_n = frameRate.fps_n.min;
	fps_d = frameRate.fps_d.max;
    }
    
  // 画素フォーマット，画像サイズ，フレームレートをセット
    setFormat(_pixelFormat, _width, _height, fps_n, fps_d);
}

//! Video for Linux v.2カメラオブジェクトを破壊する
V4L2Camera::~V4L2Camera()
{
    stopContinuousShot();
    close(_fd);
}

/*
 *  Format stuffs.
 */
//! 指定された画素フォーマットがこのカメラでサポートされているか調べる
/*!
  \param pixelFormat	サポートの有無を調べたい画素フォーマット
  \return		サポートされていればtrue，そうでなければfalse
*/
bool
V4L2Camera::isAvailable(PixelFormat pixelFormat) const
{
    BOOST_FOREACH (const Format& format, _formats)
	if (format.pixelFormat != UNKNOWN_PIXEL_FORMAT &&
	    format.pixelFormat == pixelFormat)
	    return true;

    return false;
}

//! 画素フォーマット，画像サイズおよびフレームレートを設定する
/*!
  \param pixelFormat	設定したい画素フォーマット
  \param width		設定したい画像幅
  \param height		設定したい画像高さ
  \param fps_n		設定したいフレームレートの分子
  \param fps_d		設定したいフレームレートの分母
  \return		このカメラオブジェクト
*/
V4L2Camera&
V4L2Camera::setFormat(PixelFormat pixelFormat, u_int width, u_int height,
		      u_int fps_n, u_int fps_d)
{
    using namespace	std;

  // 指定された画素フォーマット，画像サイズ，フレーム間隔の組み合わせが有効かチェック
    BOOST_FOREACH (const FrameSize& frameSize,
		   availableFrameSizes(pixelFormat))
    {
	if (frameSize.width .involves(width) &&
	    frameSize.height.involves(height))
	{
	    BOOST_FOREACH (const FrameRate& frameRate,
			   frameSize.availableFrameRates())
	    {
		if (frameRate.fps_n.involves(fps_n) &&
		    frameRate.fps_d.involves(fps_d))
		    goto ok;
	    }
	}
    }
    
    throw invalid_argument("V4L2Camera::setFormat() illegal combination of pixel format, frame size and frame rate!! ");

  // 画素フォーマットと画像サイズを設定
  ok:
    const bool	cont = inContinuousShot();
    if (cont)			// 画像を出力中であれば...
	stopContinuousShot();	// 止める

    unmapBuffers();
    
    v4l2_format	fmt;
    fmt.type		    = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = width;
    fmt.fmt.pix.height      = height;
    fmt.fmt.pix.pixelformat = pixelFormat;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;
    if (ioctl(VIDIOC_S_FMT, &fmt))
	throw runtime_error(string("V4L2Camera::setFormat(): VIDIOC_S_FMT failed!! ") + strerror(errno));
    _width	 = fmt.fmt.pix.width;
    _height	 = fmt.fmt.pix.height;
    _pixelFormat = uintToPixelFormat(fmt.fmt.pix.pixelformat);

  // フレーム間隔を設定
    v4l2_streamparm	streamparm;
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    streamparm.parm.capture.timeperframe.numerator   = fps_n;
    streamparm.parm.capture.timeperframe.denominator = fps_d;
    if (ioctl(VIDIOC_S_PARM, &streamparm))
	throw runtime_error(string("V4L2Camera::setFormat(): VIDIOC_S_PARM failed!! ") + strerror(errno));

  // バッファをマップ
    mapBuffers(NB_BUFFERS);
    
    if (cont)			// 画像を出力中だったなら...
	continuousShot();	// 再び出力させる

    return *this;
}

//! 現在カメラに設定されているフレームレートを返す
/*!
  \param fps_n	設定されているフレームレートの分子が返される
  \param fps_d	設定されているフレームレートの分母が返される
*/
void
V4L2Camera::getFrameRate(u_int& fps_n, u_int& fps_d) const
{
    using namespace	std;
    
    v4l2_streamparm	streamparm;
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(VIDIOC_G_PARM, &streamparm))
	throw runtime_error(string("V4L2Camera::getFrameRate(): VIDIOC_G_PARM failed!! ") + strerror(errno));
    fps_n = streamparm.parm.capture.timeperframe.numerator;
    fps_d = streamparm.parm.capture.timeperframe.denominator;
}

/*
 *  Feature stuffs.
 */
//! 指定された属性がこのカメラでサポートされているか調べる
/*!
  \param feature	サポートの有無を調べたい属性
  \return		サポートされていればtrue，そうでなければfalse
*/
bool
V4L2Camera::isAvailable(Feature feature) const
{
    BOOST_FOREACH (const Control& control, _controls)
	if (control.feature != UNKNOWN_FEATURE && control.feature == feature)
	    return true;

    return false;
}
    
//! 指定された属性の値を設定する
/*!
  \param feature	値を設定したい属性
  \param value		設定する値
  \return		このカメラオブジェクト
*/
V4L2Camera&
V4L2Camera::setValue(Feature feature, int value)
{
    using namespace	std;
    
    const Control&	control = featureToControl(feature);
    
    if (control.flags & V4L2_CTRL_FLAG_READ_ONLY)
	throw runtime_error("V4L2Camera::setValue(): read only feature!! ");

    if (control.type == V4L2_CTRL_TYPE_BOOLEAN)
	value = (value ? 1 : 0);

    if (!control.range.involves(value))
	throw out_of_range("V4L2Camera::setValue(): out of range value!! ");

    v4l2_control	ctrl;
    ctrl.id    = control.feature;
    ctrl.value = value;
    if (ioctl(VIDIOC_S_CTRL, &ctrl))
	throw runtime_error(string("V4L2Camera::setValue(): VIDIOC_S_CTRL failed!! ") + strerror(errno));
    
    return *this;
}
    
//! 指定された属性の現在の値を調べる
/*!
  \param feature	対象となる属性
  \return		現在の値
*/
int
V4L2Camera::getValue(Feature feature) const
{
    using namespace	std;
    
    const Control&	control = featureToControl(feature);

    if (control.flags & V4L2_CTRL_FLAG_WRITE_ONLY)
	throw runtime_error("V4L2Camera::getValue(): write only feature!! ");

    v4l2_control	ctrl;
    ctrl.id = control.feature;
    if (ioctl(VIDIOC_G_CTRL, &ctrl))
	throw runtime_error(string("V4L2Camera::getValue(): VIDIOC_G_CTRL failed!! ") + strerror(errno));

    return ctrl.value;
}
    
//! 指定された属性がとり得る値の範囲と変化刻みを調べる
/*!
  \param feature	対象となる属性
  \param min		とり得る値の最小値が返される. 
  \param max		とり得る値の最大値が返される. 
  \param step		値の変化刻みが返される. 
*/
void
V4L2Camera::getMinMaxStep(Feature feature, int& min, int& max, int& step) const
{
    const Control&	control = featureToControl(feature);

    min  = control.range.min;
    max  = control.range.max;
    step = control.range.step;
}

/*
 *  Capture stuffs.
 */
//! カメラからの画像の連続的出力を開始する
/*!
  \return	このカメラオブジェクト
*/
V4L2Camera&
V4L2Camera::continuousShot()
{
    if (!_inContinuousShot)
    {
	using namespace	std;
    
	int	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (ioctl(VIDIOC_STREAMON, &type))
	    throw runtime_error(string("V4L2Camera::continuousShot(): VIDIOC_STREAMON failed!! ") + strerror(errno));

	_inContinuousShot = true;
    }
    
    return *this;
}

//! カメラからの画像の連続的出力を停止する
/*!
  \return	このカメラオブジェクト
*/
V4L2Camera&
V4L2Camera::stopContinuousShot()
{
    if (_inContinuousShot)
    {
	using namespace	std;

	int	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (ioctl(VIDIOC_STREAMOFF, &type))
	    throw runtime_error(string("V4L2Camera::stopContinuousShot(): VIDIOC_STREAMOFF failed!! ") + strerror(errno));
	unmapBuffers();
	mapBuffers(NB_BUFFERS);
	_inContinuousShot = false;
    }
    
    return *this;
}

#ifdef HAVE_LIBTUTOOLS__
//! IEEE1394カメラから出力された画像1枚分のデータを適当な形式に変換して取り込む
/*!
  テンプレートパラメータTは, 格納先の画像の画素形式を表す. なお, 本関数を
  呼び出す前に #snap() によってカメラからの画像を保持しておかなければならない.
  \param image	画像データを格納する画像オブジェクト. 画像の幅と高さは, 
		現在カメラに設定されている画像サイズに合わせて自動的に
		設定される. また, カメラに設定されたフォーマットの画素形式
		が画像のそれに一致しない場合は, 自動的に変換が行われる.
		サポートされている画素形式Tは, u_char, short, float, double,
		RGB, RGBA, BGR,	ABGR, YUV444, YUV422, YUYV422, YUV411
		のいずれかである.	また, サポートされている変換は以下のとおりであり,
		カメラの画素形式がこれ以外に設定されている場合はstd::domain_error
		例外が送出される.
		    -# #BGR24 -> T (YUV422, YUYV422, YUV411 を除く)
		    -# #RGB24 -> T (YUV422, YUYV422, YUV411 を除く) 
		    -# #BGR32 -> T (YUV422, YUYV422, YUV411 を除く) 
		    -# #RGB32 -> T (YUV422, YUYV422, YUV411 を除く) 
		    -# #GREY -> T
		    -# #Y16 -> T
		    -# #YUYV -> T
		    -# #UYVY -> T
  \return	このIEEE1394カメラオブジェクト
*/
template <class T> const V4L2Camera&
V4L2Camera::operator >>(Image<T>& image) const
{
    if (_current == ~0)
	throw std::runtime_error("TU::V4L2Camera::operator >>: no images snapped!!");
    const u_char* const	img = (u_char*)_buffers[_current].p();
    
  // Transfer image data from current buffer.
    image.resize(height(), width());
    switch (pixelFormat())
    {
      case BGR24:
      {
	const BGR*	src = (const BGR*)img;
	for (u_int v = 0; v < image.height(); ++v)
	    src = image[v].fill(src);
      }
	break;

      case RGB24:
      {
	const RGB*	src = (const RGB*)img;
	for (u_int v = 0; v < image.height(); ++v)
	    src = image[v].fill(src);
      }
	break;

      case BGR32:
      {
	const ABGR*	src = (const ABGR*)img;
	for (u_int v = 0; v < image.height(); ++v)
	    src = image[v].fill(src);
      }
	break;

      case RGB32:
      {
	const RGBA*	src = (const RGBA*)img;
	for (u_int v = 0; v < image.height(); ++v)
	    src = image[v].fill(src);
      }
	break;

      case GREY:
      {
	const u_char*	src = (const u_char*)img;
	for (u_int v = 0; v < image.height(); ++v)
	    src = image[v].fill(src);
      }
	break;

      case Y16:
      {
	const u_short*	src = (const u_short*)img;
	for (u_int v = 0; v < image.height(); ++v)
	    src = image[v].fill(src);
      }
	break;

      case YUYV:
      {
	const YUYV422*	src = (const YUYV422*)img;
	for (u_int v = 0; v < image.height(); ++v)
	    src = image[v].fill(src);
      }
        break;

      case UYVY:
      {
	const YUV422*	src = (const YUV422*)img;
	for (u_int v = 0; v < image.height(); ++v)
	    src = image[v].fill(src);
      }
	break;

      default:
	throw std::domain_error("V4L2Camera::operator >>(): unknown pixel format!!");
	break;
    }

    return *this;
}

//! カメラから出力された画像をRGB形式カラー画像として取り込む
/*!
  #operator >>() との違いは, #PixelFormat がbayerパターンである場合,
  RGB形式への変換を行うことである. テンプレートパラメータTは, 格納先の画像の
  画素形式を表す. なお, 本関数を呼び出す前に #snap() によってカメラからの
  画像を保持しておかなければならない.
  \param image	画像データを格納する画像オブジェクト. 画像の幅と高さは,
		現在カメラに設定されている画像サイズに合わせて自動的に
		設定される. サポートされている画素形式Tは RGB, RGBA,
		BGR, ABGR のいずれかである. また, サポートされている変換は
		以下のとおりであり, カメラの画素形式がこれ以外に設定されている
		場合はstd::domain_error例外が送出される.
		    -# #SBGGR8 -> T
		    -# #SGBRG8 -> T
		    -# #SGRBG8 -> T
  \return	このカメラオブジェクト
*/
template <class T> const V4L2Camera&
V4L2Camera::captureRGBImage(Image<T>& image) const
{
    if (_current == ~0)
	throw std::runtime_error("TU::V4L2Camera::captureRGBImage: no images snapped!!");
    const u_char* const	img = (const u_char*)_buffers[_current].p();
    
  // Transfer image data from current buffer.
    image.resize(height(), width());
    switch (pixelFormat())
    {

      case SBGGR8:
      {
	const u_char*	p = bayerBGGR2x2(img, &image[0][0], width());
	int		v = 1;
	while (v < image.height() - 1)	// 中間の行を処理
	{
	    p = bayerBGGROdd3x3 (p, &image[v++][0], width());
	    p = bayerBGGREven3x3(p, &image[v++][0], width());
	}
	bayerBGGR2x2(p - width(), &image[v][0], width());
      }
	break;

      case SGBRG8:
      {
	const u_char*	p = bayerGBRG2x2(img, &image[0][0], width());
	int		v = 1;
	while (v < image.height() - 1)	// 中間の行を処理
	{
	    p = bayerGBRGOdd3x3 (p, &image[v++][0], width());
	    p = bayerGBRGEven3x3(p, &image[v++][0], width());
	}
	bayerGBRG2x2(p - width(), &image[v][0], width());
      }
        break;

      case SGRBG8:
      {
	const u_char*	p = bayerGRBG2x2(img, &image[0][0], width());
	int		v = 1;
	while (v < image.height() - 1)	// 中間の行を処理
	{
	    p = bayerGRBGOdd3x3 (p, &image[v++][0], width());
	    p = bayerGRBGEven3x3(p, &image[v++][0], width());
	}
	bayerGRBG2x2(p - width(), &image[v][0], width());
      }
	break;

      default:
	*this >> image;
	break;
    }

    return *this;
}
#else
struct RGB
{
    u_char	r, g, b;
};
#endif	// HAVE_LIBTUTOOLS__

//! カメラから出力された画像1枚分のデータをなんら変換を行わずに取り込む
/*!
  本関数を呼び出す前に #snap() によってカメラからの画像を保持しておかなければ
  ならない.
  \param image	画像データの格納領域へのポインタ. #width(), #height() および
		#pixelFormat() を用いて画像のサイズと画素の形式を調べて
		画像1枚分の領域を確保しておくのは, ユーザの責任である.
  \return	このカメラオブジェクト
*/
const V4L2Camera&
V4L2Camera::captureRaw(void* image) const
{
    if (_current == ~0)
	throw std::runtime_error("V4L2Camera::captureRaw(): no images snapped!!");
    memcpy(image, _buffers[_current].p(), _buffers[_current].size());

    return *this;
}

//! カメラから出力されたBayerパターン画像1枚分のデータをRGB形式に変換して取り込む
/*!
  本関数を呼び出す前に #snap() によってカメラからの画像を保持しておかなければ
  ならない.
  \param image	画像データの格納領域へのポインタ. #width(), #height() および
		#pixelFormat() を用いて画像のサイズと画素の形式を調べて
		画像1枚分の領域を確保しておくのは, ユーザの責任である.
		画像データは, 各画素毎に R, G, B (各 1 byte)の順で格納され
		る. カメラの画素形式が #SBGGR8, #SGRBG8, #SGBRG8 以外に設定され
		ている場合はstd::domain_error例外が送出される.
  \return	このカメラオブジェクト
*/
const V4L2Camera&
V4L2Camera::captureBayerRaw(void* image) const
{
    if (_current == ~0)
	throw std::runtime_error("V4L2Camera::captureBayerRaw(): no images snapped!!");
    const u_char* const	img = (const u_char*)_buffers[_current].p();
    
  // Transfer image data from current buffer.
    switch (pixelFormat())
    {
      case SBGGR8:
      {
	RGB*		rgb = (RGB*)image;
	const u_char*	p = bayerBGGR2x2(img, rgb, width());
	rgb += width();
	for (int n = height(); (n -= 2) > 0; )	// 中間の行を処理
	{
	    p = bayerBGGROdd3x3 (p, rgb, width());
	    rgb += width();
	    p = bayerBGGREven3x3(p, rgb, width());
	    rgb += width();
	}
	bayerBGGR2x2(p - width(), rgb, width());
      }
	break;

      case SGRBG8:
      {
	RGB*		rgb = (RGB*)image;
	const u_char*	p = bayerGRBG2x2(img, rgb, width());
	rgb += width();
	for (int n = height(); (n -= 2) > 0; )	// 中間の行を処理
	{
	    p = bayerGRBGOdd3x3 (p, rgb, width());
	    rgb += width();
	    p = bayerGRBGEven3x3(p, rgb, width());
	    rgb += width();
	}
	bayerGRBG2x2(p - width(), rgb, width());
      }
	break;

      case SGBRG8:
      {
	RGB*		rgb = (RGB*)image;
	const u_char*	p = bayerGBRG2x2(img, rgb, width());
	rgb += width();
	for (int n = height(); (n -= 2) > 0; )	// 中間の行を処理
	{
	    p = bayerGBRGOdd3x3 (p, rgb, width());
	    rgb += width();
	    p = bayerGBRGEven3x3(p, rgb, width());
	    rgb += width();
	}
	bayerGBRG2x2(p - width(), rgb, width());
      }
        break;

      default:
	throw std::domain_error("V4L2Camera::captureBayerRaw(): must be bayer format!!");
	break;
    }

    return *this;
}

/*
 *  Utility functions
 */
//! unsinged intの値を同じビットパターンを持つ #PixelFormat に直す
/*!
  \param pixelFormat	#PixelFormat に直したいunsigned int値
  \return		#PixelFormat 型のenum値
 */
V4L2Camera::PixelFormat
V4L2Camera::uintToPixelFormat(u_int pixelFormat)
{
    switch (pixelFormat)
    {
      case BGR24:
	return BGR24;
      case RGB24:
	return RGB24;
      case BGR32:
	return BGR32;
      case RGB32:
	return RGB32;
      case GREY:
	return GREY;
      case Y16:
	return Y16;
      case YUYV:
	return YUYV;
      case UYVY:
	return UYVY;
      case SBGGR8:
	return SBGGR8;
      case SGBRG8:
	return SGBRG8;
      case SGRBG8:
	return SGRBG8;
    }
    
    return UNKNOWN_PIXEL_FORMAT;
}

//! unsinged intの値を同じビットパターンを持つ #Feature に直す
/*!
  \param feature	#Feature に直したいunsigned int値
  \return		#Feature 型のenum値
 */
V4L2Camera::Feature
V4L2Camera::uintToFeature(u_int feature)
{
    switch (feature)
    {
      case BRIGHTNESS:
	return BRIGHTNESS;
      case BRIGHTNESS_AUTO:
	return BRIGHTNESS_AUTO;
      case CONTRAST:
	return CONTRAST;
      case GAIN:
	return GAIN;
      case SATURATION:
	return SATURATION;
      case HUE:
	return HUE;
      case HUE_AUTO:
	return HUE_AUTO;
      case GAMMA:
	return GAMMA;
      case SHARPNESS:
	return SHARPNESS;
      case WHITE_BALANCE_TEMPERATURE:
	return WHITE_BALANCE_TEMPERATURE;
      case WHITE_BALANCE_AUTO:
	return WHITE_BALANCE_AUTO;
      case BACKLIGHT_COMPENSATION:
	return BACKLIGHT_COMPENSATION;
      case POWER_LINE_FREQUENCY:
	return POWER_LINE_FREQUENCY;
      case EXPOSURE_AUTO:
	return EXPOSURE_AUTO;
      case EXPOSURE_AUTO_PRIORITY:
	return EXPOSURE_AUTO_PRIORITY;
      case EXPOSURE_ABSOLUTE:
	return EXPOSURE_ABSOLUTE;
      case FOCUS_ABSOLUTE:
	return FOCUS_ABSOLUTE;
      case FOCUS_RELATIVE:
	return FOCUS_RELATIVE;
      case FOCUS_AUTO:
	return FOCUS_AUTO;
      case ZOOM_ABSOLUTE:
	return ZOOM_ABSOLUTE;
      case ZOOM_RELATIVE:
	return ZOOM_RELATIVE;
      case ZOOM_CONTINUOUS:
	return ZOOM_CONTINUOUS;
      case PAN_ABSOLUTE:
	return PAN_ABSOLUTE;
      case PAN_RELATIVE:
	return PAN_RELATIVE;
      case PAN_RESET:
	return PAN_RESET;
      case TILT_ABSOLUTE:
	return TILT_ABSOLUTE;
      case TILT_RELATIVE:
	return TILT_RELATIVE;
      case TILT_RESET:
	return TILT_RESET;
    }
    
    return UNKNOWN_FEATURE;
}
    
/*
 *  private member functions
 */
void
V4L2Camera::enumerateFormats()
{
  // このカメラがサポートする画素フォーマットを列挙
    v4l2_fmtdesc	fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    for (fmtdesc.index = 0; ioctl(VIDIOC_ENUM_FMT, &fmtdesc) == 0;
	 ++fmtdesc.index)
    {
	PixelFormat	pixelFormat = uintToPixelFormat(fmtdesc.pixelformat);
	if (pixelFormat == UNKNOWN_PIXEL_FORMAT)  // 未知のフォーマットならば...
	    continue;				  // スキップする
	
	_formats.push_back(Format());
	Format&	format = _formats.back();

	format.pixelFormat = pixelFormat;
	format.name	   = (char*)fmtdesc.description;

      // この画素フォーマットのもとでサポートされる画像サイズを列挙
	v4l2_frmsizeenum	fsize;
	memset(&fsize, 0, sizeof(fsize));
	fsize.type	   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fsize.pixel_format = fmtdesc.pixelformat;

	for (fsize.index = 0; ioctl(VIDIOC_ENUM_FRAMESIZES, &fsize) == 0;
	     ++fsize.index)
	{
	    format.frameSizes.push_back(FrameSize());
	    FrameSize&	frameSize = format.frameSizes.back();

	    if (fsize.type == V4L2_FRMSIZE_TYPE_DISCRETE)
	    {
		frameSize.width.min   = fsize.discrete.width;
		frameSize.width.max   = fsize.discrete.width;
		frameSize.width.step  = 1;
		frameSize.height.min  = fsize.discrete.height;
		frameSize.height.max  = fsize.discrete.height;
		frameSize.height.step = 1;
	    }
	    else
	    {
		frameSize.width.min  = fsize.stepwise.min_width;
		frameSize.width.max  = fsize.stepwise.max_width;
		frameSize.height.min = fsize.stepwise.min_height;
		frameSize.height.max = fsize.stepwise.max_height;

		if (fsize.type == V4L2_FRMSIZE_TYPE_CONTINUOUS)
		{
		    frameSize.width.step  = 1;
		    frameSize.height.step = 1;
		}
		else
		{
		    frameSize.width.step  = fsize.stepwise.step_width;
		    frameSize.height.step = fsize.stepwise.step_height;
		}
	    }

	  // この画素フォーマットと画像サイズのもとでサポートされる
	  // フレームレートを列挙
	    v4l2_frmivalenum	fival;
	    memset(&fival, 0, sizeof(fival));
	    fival.type	       = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	    fival.pixel_format = fsize.pixel_format;
	    fival.width	       = frameSize.width.max;
	    fival.height       = frameSize.height.max;
	    
	    for (fival.index = 0;
		 ioctl(VIDIOC_ENUM_FRAMEINTERVALS, &fival) == 0;
		 ++fival.index)
	    {
		frameSize.frameRates.push_back(FrameRate());
		FrameRate&	frameRate = frameSize.frameRates.back();

		if (fival.type == V4L2_FRMIVAL_TYPE_DISCRETE)
		{
		    frameRate.fps_n.min  = fival.discrete.numerator;
		    frameRate.fps_n.max  = fival.discrete.numerator;
		    frameRate.fps_d.min  = fival.discrete.denominator;
		    frameRate.fps_d.max  = fival.discrete.denominator;
		    frameRate.fps_n.step = 1;
		    frameRate.fps_d.step = 1;
		}
		else
		{
		    frameRate.fps_n.min = fival.stepwise.min.numerator;
		    frameRate.fps_n.max = fival.stepwise.max.numerator;
		    frameRate.fps_d.min = fival.stepwise.min.denominator;
		    frameRate.fps_d.max = fival.stepwise.max.denominator;

		    if (fival.type == V4L2_FRMIVAL_TYPE_CONTINUOUS)
		    {
			frameRate.fps_n.step = 1;
			frameRate.fps_d.step = 1;
		    }
		    else
		    {
			frameRate.fps_n.step = fival.stepwise.step.numerator;
			frameRate.fps_d.step = fival.stepwise.step.denominator;
		    }
		}
	    }
	}
    }
}
    
void
V4L2Camera::enumerateControls()
{
  // このカメラがサポートするコントロール(属性)を列挙
    v4l2_queryctrl	ctrl;
    memset(&ctrl, 0, sizeof(ctrl));

    for (int id = 0, ret; (ret = ioctl(id, ctrl)) == 0 || errno != EINVAL; )
	if (ret)		// ioctlがEINVALでないエラーを返したら...
	{
	    if (ctrl.id <= id)	// 次のctrl.idがセットされなかったら(v4l2のbug)
		++id;		// 自分で次のidに進めなければならない
	    else
		break;
	}
	else			// ioctlが正常に終了したら...
	{
	    if (ctrl.id == id)	// ctrl.idが更新されなかったら...(v4l2のbug)
		break;		// 列挙を中断
	    
	    id = ctrl.id;	// 次のidをセットする．

	    Feature	feature = uintToFeature(ctrl.id);
	    if (ctrl.flags & V4L2_CTRL_FLAG_DISABLED ||	// 無効化されているか
		feature == UNKNOWN_FEATURE)		// 未知の属性ならば...
		continue;				// スキップして次へ

	    _controls.push_back(Control());
	    Control&	control = _controls.back();
	    
	    control.feature = feature;
	    control.name    = (char*)ctrl.name;

	    switch (ctrl.type)
	    {
	      case V4L2_CTRL_TYPE_INTEGER:
	      case V4L2_CTRL_TYPE_BOOLEAN:
	      case V4L2_CTRL_TYPE_MENU:
		control.type = ctrl.type;
		if (control.type == V4L2_CTRL_TYPE_MENU)
		{
		    control.range.min = 0;
		    control.range.max
			= enumerateMenuItems(ctrl, control.menuItems);
		    control.range.step = 1;
		}
		else
		{
		    control.range.min  = ctrl.minimum;
		    control.range.max  = ctrl.maximum;
		    control.range.step = ctrl.step;
		}
		break;
	      default:
		_controls.pop_back();
		break;
	    }

	    control.def	  = ctrl.default_value;
	    control.flags = ctrl.flags;
	}
}
    
int
V4L2Camera::enumerateMenuItems(const v4l2_queryctrl& ctrl,
			       std::vector<MenuItem>& menuItems)
{
    v4l2_querymenu	menu;
    memset(&menu, 0, sizeof(menu));
    menu.id = ctrl.id;
    
    for (menu.index = ctrl.minimum; menu.index <= ctrl.maximum; ++menu.index)
    {
	if (ioctl(VIDIOC_QUERYMENU, &menu))
	    break;
	
	menuItems.push_back(MenuItem());
	MenuItem&	menuItem = menuItems.back();

	menuItem.index = menu.index;
	menuItem.name  = (char*)menu.name;
    }

    return menu.index - 1 - ctrl.minimum;
}

const V4L2Camera::Format&
V4L2Camera::pixelFormatToFormat(PixelFormat pixelFormat) const
{
    BOOST_FOREACH (const Format& format, _formats)
	if (format.pixelFormat != UNKNOWN_PIXEL_FORMAT &&
	    format.pixelFormat == pixelFormat)
	    return format;

    throw std::runtime_error("V4L2Camera::pixelFormatToFormat(): unknown pixel format!! ");

    return _formats[0];
}
    
const V4L2Camera::Control&
V4L2Camera::featureToControl(Feature feature) const
{
    using namespace	std;
    
    BOOST_FOREACH (const Control& control, _controls)
	if (control.feature != UNKNOWN_FEATURE && control.feature == feature)
	{
	    if (control.flags & V4L2_CTRL_FLAG_DISABLED)
		throw runtime_error("V4L2Camera::featureToControl(): disabled feature!! ");
	    if (control.flags & V4L2_CTRL_FLAG_INACTIVE)
		throw runtime_error("V4L2Camera::featureToControl(): inactive feature!! ");
	    return control;
	}
    
    throw runtime_error("V4L2Camera::featureToControl(): unknown feature!! ");

    return _controls[0];
}

//! 指定した個数の受信用のバッファを確保し，メモリをマップしてキューに入れる
/*!
  \param n	バッファの個数
 */
void
V4L2Camera::mapBuffers(u_int n)
{
    n = requestBuffers(n);	// 指定された個数だけバッファを確保
    if (n < 2)			// 充分な個数を確保できなかったら...
	throw std::runtime_error("V4L2Camera::mapBuffer(): failed to allocate sufficient number of buffers!!");	// 脱出する

    _buffers.resize(n);
    for (u_int i = 0; i < _buffers.size(); ++i)	// 確保された個数のバッファに
    {
	_buffers[i].map(_fd, i);		// メモリをマップして
	enqueueBuffer(i);			// キューに入れる
    }

    usleep(100000);
}

//! すべての受信用バッファを解放する
void
V4L2Camera::unmapBuffers()
{
    for (u_int i = 0; i < _buffers.size(); ++i)
	_buffers[i].unmap();
    requestBuffers(0);	// 確保するバッファ数を0にすることによってキューをクリア
    _current = ~0;	// データが残っていないことを示す
}

//! 指定した個数の受信用バッファを確保するように要求する
/*!
  \param n	バッファの個数
  \return	実際に確保されたバッファの個数
*/
u_int
V4L2Camera::requestBuffers(u_int n)
{
    using namespace	std;
    
    v4l2_requestbuffers	reqbuf;
    memset(&reqbuf, 0, sizeof(reqbuf));
    reqbuf.count  = n;
    reqbuf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(VIDIOC_REQBUFS, &reqbuf))
	throw runtime_error(string("V4L2Camera::requestBuffer(): VIDIOC_REQBUFS failed!! ") + strerror(errno));
#ifdef _DEBUG
    cerr << "VIDIOC_REQBUFS(" << reqbuf.count << ")" << endl;
#endif
    return reqbuf.count;
}

//! 指定したバッファをqueueに入れる
/*!
  \param index	バッファのindex
 */
void
V4L2Camera::enqueueBuffer(u_int index) const
{
    using namespace	std;
    
    v4l2_buffer	buf;
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = index;
    if (ioctl(VIDIOC_QBUF, &buf))
	throw runtime_error(string("TU::V4L2Camera::requeueBuffer(): ioctl(VIDIOC_QBUF) failed!! ") + strerror(errno));
#ifdef _DEBUG
    cerr << "VIDIOC_QBUF(" << buf.index << ")" << endl;
#endif
}

//! データを受信したバッファをキューから取り出す
/*!
  実際にデータが受信されるまで, 本関数は呼び出し側に制御を返さない. 
  \return	データを受信したバッファのindex
 */
u_int
V4L2Camera::dequeueBuffer() const
{
    using namespace	std;
    
    v4l2_buffer	buf;
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(VIDIOC_DQBUF, &buf))
	throw runtime_error(string("TU::V4L2Camera::waitBuffer(): ioctl(VIDIOC_DQBUF) failed!! ") + strerror(errno));
#ifdef _DEBUG
    cerr << "VIDIOC_DQBUF(" << buf.index << ")" << endl;
#endif
    return buf.index;
}

int
V4L2Camera::ioctl(int request, void* arg) const
{
    int	ret;

    do
    {
	ret = ::ioctl(_fd, request, arg);
    } while (ret == -1 && errno == EINTR);

    return ret;
}

//! 指定されたコントロールIDに対応するコントロールの情報を取得する．
/*!
  V4L2_CTRL_FLAG_NEXT_CTRL フラグを立てて VIDIO_QUERYCTRL を行っている．
  ctrl.id に指定された ID より大きな最小の ID が返されるはずであるが，
  そうならないbuggyなv4l2の実装が存在する．
  \param id	コントロールID
  \param ctrl	コントロールの情報が返される
  \return	正常に情報が取得されたら0，エラーが生じたら非零
*/
int
V4L2Camera::ioctl(int id, v4l2_queryctrl& ctrl) const
{
    int	ret;
    
    for (int n = 0; n < CONTROL_IO_ERROR_RETRIES; ++n)
    {
	ctrl.id = id | V4L2_CTRL_FLAG_NEXT_CTRL;

      // ioctlが成功するか，I/Oエラー以外のエラーが生じたら脱出する．
	if ((ret = ::ioctl(_fd, VIDIOC_QUERYCTRL, &ctrl)) == 0 ||
	    (errno != EIO && errno != EPIPE && errno != ETIMEDOUT))
	    break;
    }

    return ret;
}
    
/************************************************************************
*  class V4L2Camera::Buffer						*
************************************************************************/
void
V4L2Camera::Buffer::map(int fd, u_int index)
{
    using namespace	std;

  // バッファの大きさとオフセットを調べる．
    v4l2_buffer	buf;
    memset(&buf, 0, sizeof(buf));
    buf.index  = index;
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (::ioctl(fd, VIDIOC_QUERYBUF, &buf))
	throw runtime_error(string("V4L2Camera::Buffer::Buffer(): VIDIOC_QUERYBUF failed!! ") + strerror(errno));
	
  // 得られた大きさとオフセットをもとにメモリ領域をバッファにマップする．
    if ((_p = ::mmap(0, buf.length, PROT_READ | PROT_WRITE,
		     MAP_SHARED, fd, buf.m.offset)) == MAP_FAILED)
    {
	_p = 0;
	throw runtime_error(string("V4L2Camera::Buffer::Buffer(): mmap failed!! ") + strerror(errno));
    }
    _size = buf.length;
#ifdef _DEBUG
    cerr << "Buffer::map(): VIDIOC_QUERYBUF(" << index << ") & mmap" << endl;
#endif
}

void
V4L2Camera::Buffer::unmap()
{
    if (_p)
    {
	::munmap(_p, _size);
	_p = 0;
	_size = 0;
#ifdef _DEBUG
	std::cerr << "Buffer::unmap(): munmap" << std::endl;
#endif
    }
}

/************************************************************************
*  friend functions							*
************************************************************************/
std::ostream&
operator <<(std::ostream& out, const V4L2Camera::Format& format)
{
    out << format.name;
    char	fourcc[5];
    fourcc[0] =	 format.pixelFormat	   & 0xff;
    fourcc[1] = (format.pixelFormat >>  8) & 0xff;
    fourcc[2] = (format.pixelFormat >> 16) & 0xff;
    fourcc[3] = (format.pixelFormat >> 24) & 0xff;
    fourcc[4] = '\0';
    out << " [id:" << fourcc << ']' << std::endl;
    
    BOOST_FOREACH (const V4L2Camera::FrameSize& frameSize, format.frameSizes)
	out << "  " << frameSize << std::endl;

    return out ;
}
    
std::ostream&
operator <<(std::ostream& out, const V4L2Camera::Control& control)
{
    using namespace	std;
    
    out << control.name << " [id:";
    if (control.feature == V4L2Camera::UNKNOWN_FEATURE)
	out << "UNKNOWN";
    else
	out << "0x" << hex << control.feature << dec;
    out << ']' << endl;
    out <<   "  type:    "
	<< (control.type == V4L2_CTRL_TYPE_INTEGER    ? "INT" :
	    control.type == V4L2_CTRL_TYPE_BOOLEAN    ? "BOOL" :
	    control.type == V4L2_CTRL_TYPE_MENU	      ? "MENU" :
	    control.type == V4L2_CTRL_TYPE_BUTTON     ? "BUTTON" :
	    control.type == V4L2_CTRL_TYPE_INTEGER64  ? "INT64" :
	    control.type == V4L2_CTRL_TYPE_CTRL_CLASS ? "CLASS" :
	    control.type == V4L2_CTRL_TYPE_STRING     ? "STRING" : "UNKNOWN")
	<< "\n  range:   " << control.range
	<< "\n  default: " << control.def
	<< "\n  flags:  ";
    if (control.flags & V4L2_CTRL_FLAG_DISABLED)
	out << " DISABLED";
    if (control.flags & V4L2_CTRL_FLAG_GRABBED)
	out << " GRABBED";
    if (control.flags & V4L2_CTRL_FLAG_READ_ONLY)
	out << " R/O";
    if (control.flags & V4L2_CTRL_FLAG_UPDATE)
	out << " UPDATE";
    if (control.flags & V4L2_CTRL_FLAG_INACTIVE)
	out << " INACTIVE";
    if (control.flags & V4L2_CTRL_FLAG_SLIDER)
	out << " SLIDER";
    if (control.flags & V4L2_CTRL_FLAG_WRITE_ONLY)
	out << " W/O";
    out << endl;
    if (control.type == V4L2_CTRL_TYPE_MENU)
	BOOST_FOREACH (const V4L2Camera::MenuItem& menuItem,
		       control.menuItems)
	    out << "    " << menuItem << endl;

    return out;
}

/************************************************************************
*  global functions							*
************************************************************************/
//! 値の範囲を出力ストリームに出力する
/*
  \param out	出力ストリーム
  \param range	値の範囲
  \return	outで指定した出力ストリーム
*/ 
template <class T> std::ostream&
operator <<(std::ostream& out, const V4L2Camera::Range<T>& range)
{
    if (range.min == range.max)
	return out << range.min;
    else
	return out << '['
		   << range.min << ',' << range.max << ':' << range.step
		   << ']';
}
    
//! 画像サイズを出力ストリームに出力する
/*
  \param out		出力ストリーム
  \param frameSize	画像サイズ
  \return		outで指定した出力ストリーム
*/ 
std::ostream&
operator <<(std::ostream& out, const V4L2Camera::FrameSize& frameSize)
{
    out << frameSize.width << 'x' << frameSize.height << ':';

    BOOST_FOREACH (const V4L2Camera::FrameRate& frameRate,
		   frameSize.availableFrameRates())
	out << ' ' << frameRate;
    return out;
}
    
//! フレームレートを出力ストリームに出力する
/*
  \param out		出力ストリーム
  \param frameRate	フレームレート
  \return		outで指定した出力ストリーム
*/ 
std::ostream&
operator <<(std::ostream& out, const V4L2Camera::FrameRate& frameRate)
{
    return out << frameRate.fps_n << '/' << frameRate.fps_d;
}

//! メニュー項目を出力ストリームに出力する
/*
  \param out		出力ストリーム
  \param menuItem	メニュー項目
  \return		outで指定した出力ストリーム
*/ 
std::ostream&
operator <<(std::ostream& out, const V4L2Camera::MenuItem& menuItem)
{
    return out << menuItem.index << ": " << menuItem.name;
}

//! 現在のカメラの設定をストリームに書き出す
/*!
  \param out		出力ストリーム
  \param camera		対象となるカメラ
  \return		outで指定した出力ストリーム
*/
std::ostream&
operator <<(std::ostream& out, const V4L2Camera& camera)
{
  // 画素フォーマットと画像サイズを書き出す．
    V4L2Camera::PixelFormat	pixelFormat = camera.pixelFormat();
    char			fourcc[5];
    fourcc[0] =	 pixelFormat	    & 0xff;
    fourcc[1] = (pixelFormat >>  8) & 0xff;
    fourcc[2] = (pixelFormat >> 16) & 0xff;
    fourcc[3] = (pixelFormat >> 24) & 0xff;
    fourcc[4] = '\0';
    out << fourcc << ' ' << camera.width() << 'x' << camera.height();

  // フレームレートを書き出す．
    u_int	fps_n, fps_d;
    camera.getFrameRate(fps_n, fps_d);
    out << ' ' << fps_n << '/' << fps_d;

  // 各カメラ属性の値を書き出す．
    BOOST_FOREACH (V4L2Camera::Feature feature, camera.availableFeatures())
	for (u_int i = 0; i < NFEATURES; ++i)
	    if (feature == features[i].feature)
	    {
		out << ' ' << features[i].name
		    << ' ' << camera.getValue(feature);
		break;
	    }
    
    return out << std::endl;
}

//! ストリームから読み込んだ設定をカメラにセットする
/*!
  \param in		入力ストリーム
  \param camera		対象となるカメラ
  \return		inで指定した入力ストリーム
*/
std::istream&
operator >>(std::istream& in, V4L2Camera& camera)
{
  // 画素フォーマット，画像サイズ，フレームレートを読み込んでカメラに設定する．
    std::string	s;
    in >> s;				// 画素フォーマット
    V4L2Camera::PixelFormat	pixelFormat
	= V4L2Camera::uintToPixelFormat( s[0]	     | (s[1] <<  8) |
					(s[2] << 16) | (s[3] << 24));
    char	c;
    u_int	w, h;
    in >> w >> c >> h;			// 画像の幅と高さ
    u_int	fps_n, fps_d;
    in >> fps_n >> c >> fps_d;		// フレームレートの分子と分母
    camera.setFormat(pixelFormat, w, h, fps_n, fps_d);
    
  // 各カメラ属性を読み込んでカメラに設定する．
    while (in >> s)
	for (u_int i = 0; i < NFEATURES; ++i)
	    if (s == features[i].name)
	    {
		int	val;
		in >> val;
		camera.setValue(features[i].feature, val);
		break;
	    }
    
    return in;
}

#ifdef HAVE_LIBTUTOOLS__
/************************************************************************
*  instantiations							*
************************************************************************/
template const V4L2Camera&
V4L2Camera::operator >>(Image<u_char>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<short>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<float>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<double>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<RGB>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<RGBA>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<BGR>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<ABGR>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<YUV444>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<YUV422>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<YUYV422>& image)	const	;
template const V4L2Camera&
V4L2Camera::operator >>(Image<YUV411>& image)	const	;
template const V4L2Camera&
V4L2Camera::captureRGBImage(Image<RGB>& image)	const	;
template const V4L2Camera&
V4L2Camera::captureRGBImage(Image<RGBA>& image)	const	;
template const V4L2Camera&
V4L2Camera::captureRGBImage(Image<BGR>& image)	const	;
template const V4L2Camera&
V4L2Camera::captureRGBImage(Image<ABGR>& image)	const	;
#endif
}
