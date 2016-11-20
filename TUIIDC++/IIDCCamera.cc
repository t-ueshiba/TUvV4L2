/*
 *  $Id$: IIDCCamera.cc 1777 2015-02-06 10:54:01Z ueshiba $
 */
#include "FireWireNode_.h"
#include "USBNode_.h"
#include <libraw1394/csr.h>
#include <algorithm>
#include <cctype>	// for isspace()
#include <cstring>
#include <iostream>

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static uint32_t
triggerModeValue(IIDCCamera::TriggerMode mode)
{
    uint32_t	val = 0;
    for (u_int n = mode; n >>= 1; )
	++val;

    return 15 - val;
}

static inline u_int
triggerModeInq(uint32_t val)
{
    return 0x1u << (15 - val);
}

static inline uint64_t
cycletime_to_cycle(uint32_t cycletime)
{
    const uint64_t	sec	 = (cycletime & 0xfe000000) >> 25;
    const uint64_t	cycle	 = (cycletime & 0x01fff000) >> 12;
  //const uint64_t	subcycle = (cycletime & 0x00000fff);

    return cycle + 8000*sec;
}

static std::istream&
skipws(std::istream& in)
{
    for (char c; in.get(c); )
	if (!isspace(c) || c == '\n')
	{
	    in.putback(c);
	    break;
	}
    return in;
}
    
/************************************************************************
*  local constants							*
************************************************************************/
// Offsets concerning with GPIO pins.
static constexpr uint32_t	Advanced_Feature_Quadlet_Offset	= 0x480;
static constexpr uint32_t	PIO_Control_Quadlet_Offset	= 0x484;
static constexpr uint32_t	SIO_Control_Quadlet_Offset	= 0x488;
static constexpr uint32_t	Strobe_Output_Quadlet_Offset	= 0x48c;

// Offsets concerning with image capturing.
static constexpr uint32_t	Cur_V_Frm_Rate			= 0x600;
static constexpr uint32_t	Cur_V_Mode			= 0x604;
static constexpr uint32_t	Cur_V_Format			= 0x608;
static constexpr uint32_t	ISO_Channel			= 0x60c;
static constexpr uint32_t	Camera_Power			= 0x610;
static constexpr uint32_t	ISO_EN				= 0x614;
static constexpr uint32_t	Memory_Save			= 0x618;
static constexpr uint32_t	One_Shot			= 0x61c;
static constexpr uint32_t	Mem_Save_Ch			= 0x620;
static constexpr uint32_t	Cur_Mem_Ch			= 0x624;
static constexpr uint32_t	Software_Trigger		= 0x62c;

// Video Mode CSR for Format_7.
static constexpr uint32_t	MAX_IMAGE_SIZE_INQ		= 0x000;
static constexpr uint32_t	UNIT_SIZE_INQ			= 0x004;
static constexpr uint32_t	IMAGE_POSITION			= 0x008;
static constexpr uint32_t	IMAGE_SIZE			= 0x00c;
static constexpr uint32_t	COLOR_CODING_ID			= 0x010;
static constexpr uint32_t	COLOR_CODING_INQ		= 0x014;
static constexpr uint32_t	PIXEL_NUMBER_INQ		= 0x034;
static constexpr uint32_t	TOTAL_BYTES_HI_INQ		= 0x038;
static constexpr uint32_t	TOTAL_BYTES_LO_INQ		= 0x03c;
static constexpr uint32_t	PACKET_PARA_INQ			= 0x040;
static constexpr uint32_t	BYTE_PER_PACKET			= 0x044;
static constexpr uint32_t	PACKET_PER_FRAME_INQ		= 0x048;
static constexpr uint32_t	UNIT_POSITION_INQ		= 0x04c;
static constexpr uint32_t	VALUE_SETTING			= 0x07c;

static constexpr uint64_t	PointGrey_Feature_ID	= 0x00b09d000004ull;

// Offsets of access control registers (PointGrey specific)
static constexpr uint32_t	BAYER_TILE_MAPPING		= 0x040;
static constexpr uint32_t	IMAGE_DATA_FORMAT		= 0x048;
static constexpr uint32_t	PIO_DIRECTION			= 0x1f8;
static constexpr uint32_t	FRAME_INFO			= 0x2f8;
static constexpr uint32_t	OUTPUT_VOLTAGE_ENABLE		= 0x9d0;

// Bit fields of feature status/control registers.
static constexpr uint32_t	Abs_Control_bit			= 0x1u << 30;
static constexpr uint32_t	One_Push_bit			= 0x1u << 28;
static constexpr uint32_t	ON_OFF_bit			= 0x1u << 25;
static constexpr uint32_t	A_M_Mode_bit			= 0x1u << 24;

// NOTE: Two buffers are not enough under kernel-2.4.6 (2001.8.24).
static constexpr u_int		NBUFFERS			= 2;

/************************************************************************
*  union AbsValue							*
************************************************************************/
union AbsValue
{
    uint32_t	i;
    float	f;
};
    
/************************************************************************
*  class IIDCCamera							*
************************************************************************/
//! IIDCカメラノードを生成する
/*!
  カメラデバイスと結びつけて使用するには initialize() する必要がある
*/
IIDCCamera::IIDCCamera()
    :_node(nullptr), _cmdRegBase(0), _acRegBase(0), _w(0), _h(0), _p(MONO_8),
     _img(nullptr), _img_size(0), _bayer(YYYY), _littleEndian(false)
{
}

//! IIDCカメラノードを生成する
/*!
  \param uniqId		個々のカメラ固有の64bit ID. 同一のバスに
			複数のカメラが接続されている場合, これによって
			同定を行う. 0が与えられると, まだ IIDCCamera
			オブジェクトを割り当てられていないカメラのうち, 
			一番最初にみつかったものがこのオブジェクトと結び
			つけられる. オブジェクト生成後は, globalUniqueId()
			によってこの値を知ることができる.
  \param type		カメラのタイプ
*/
IIDCCamera::IIDCCamera(uint64_t uniqId, Type type)
    :IIDCCamera()
{
    initialize(uniqId, type);
}

//! IIDCカメラオブジェクトを破壊する
/*!
  画像データ出力中であった場合は, それを停止する. 
*/
IIDCCamera::~IIDCCamera()
{
}

//! 移動コンストラクタ
IIDCCamera::IIDCCamera(IIDCCamera&& camera)
    :_node(std::move(camera._node)),
     _cmdRegBase(camera._cmdRegBase), _acRegBase(camera._acRegBase),
     _w(camera._w), _h(camera._h), _p(camera._p),
     _img(camera._img), _img_size(camera._img_size),
     _bayer(camera._bayer), _littleEndian(camera._littleEndian)
{
}

//! 移動代入演算子
IIDCCamera&
IIDCCamera::operator =(IIDCCamera&& camera)
{
    _node	  = std::move(camera._node);
    _cmdRegBase	  = camera._cmdRegBase;
    _acRegBase	  = camera._acRegBase;
    _w		  = camera._w;
    _h		  = camera._h;
    _p		  = camera._p;
    _img	  = camera._img;
    _img_size	  = camera._img_size;
    _bayer	  = camera._bayer;
    _littleEndian = camera._littleEndian;

    return *this;
}
    
//! IIDCカメラのglobal unique IDを返す
/*!
  \return	このIIDCカメラのglobal unique ID
*/
uint64_t
IIDCCamera::globalUniqueId() const
{
    return _node->globalUniqueId();
}

//! IIDCカメラを初期化して使用可能な状態にする
/*!
  \param uniqId		個々のカメラ固有の64bit ID. 同一のバスに
			複数のカメラが接続されている場合, これによって
			同定を行う. 0が与えられると, まだ IIDCCamera
			オブジェクトを割り当てられていないカメラのうち, 
			一番最初にみつかったものがこのオブジェクトと結び
			つけられる. オブジェクト生成後は, globalUniqueId()
			によってこの値を知ることができる.
  \param type		カメラのタイプ
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::initialize(uint64_t uniqId, Type type)
{
    try
    {
	_node.reset(new FireWireNode(type, uniqId));
    }
    catch (const std::exception& err)
    {
	_node.reset(new USBNode(type, uniqId));
    }
    
    _cmdRegBase  = _node->commandRegisterBase();
    const auto	inq = inquireBasicFunction();
    _acRegBase   = (inq & Advanced_Feature_Inq ?
		    CSR_REGISTER_BASE +
		    4*readQuadletFromRegister(Advanced_Feature_Quadlet_Offset) :
		    0);
    _img	  = nullptr;
    _bayer	  = YYYY;
    _littleEndian = false;

  // Map video1394 buffer according to current format and frame rate.
    setFormatAndFrameRate(getFormat(), getFrameRate());

  // Get Bayer pattern supported by this camera.
    if (unlockAdvancedFeature(PointGrey_Feature_ID, 10))
    {
	switch (readQuadletFromACRegister(BAYER_TILE_MAPPING))
	{
	  case RGGB:
	    _bayer = RGGB;
	    break;
	  case BGGR:
	    _bayer = BGGR;
	    break;
	  case GRBG:
	    _bayer = GRBG;
	    break;
	  case GBRG:
	    _bayer = GBRG;
	    break;
	  default:
	    break;
	}

	u_int	y16_data_format = readQuadletFromACRegister(IMAGE_DATA_FORMAT);
	if ((y16_data_format & 0x80000001) == 0x80000001)
	    _littleEndian = true;
    }
    return *this;
}
    
//! IIDCカメラの使用を終了して待機状態にする
/*!
  再びカメラデバイスと結びつけて使用するには initialize() する必要がある
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::terminate()
{
    _node	  = nullptr;
    _cmdRegBase	  = 0;
    _acRegBase	  = 0;
    _w		  = 0;
    _h		  = 0;
    _p		  = MONO_8;
    _img	  = nullptr;
    _img_size	  = 0;
    _bayer	  = YYYY;
    _littleEndian = false;

    return *this;
}
    
//! IIDCカメラの設定を工場出荷時に戻す
/*!
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::reset()
{
#if 1
    constexpr uint32_t	INITIALIZE = 0x0;
    constexpr quadlet_t	Reset = 0x1u << 31;

    writeQuadletToRegister(INITIALIZE, Reset);
    while (readQuadletFromRegister(INITIALIZE) & Reset)
	;
#else
    writeQuadletToRegister(Memory_Save, 0xdeafbeef);
#endif    
    return *this;
}
    
//! IIDCカメラの電源をon/offする
/*!
  \param onOff	trueならばon, falseならばoff
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setPower(bool enable)
{
    checkAvailability(Cam_Power_Cntl_Inq);
    writeQuadletToRegister(Camera_Power, (enable ? 0x1u << 31 : 0x0));
    return *this;
}

//! IIDCカメラのデータ転送速度を設定する
/*!
  \param speed	データ転送速度
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setSpeed(Speed speed)
{
    uint32_t	quad;
    if (inquireBasicFunction() & I1394b_mode_Capability)
    {
	quad = ((0x1u << 15) | speed);
    }
    else
    {
	switch (speed)
	{
	  case SPD_100M:
	  case SPD_200M:
	  case SPD_400M:
	    quad = (speed << 24);
	    break;
	  default:
	    throw std::runtime_error("IIDCCaera::setSpeed: specified speed is not supported by non-IEEE1394b cameras!!");
	    break;
	}
    }

    const auto	cont = inContinuousShot();
    continuousShot(false);

    writeQuadletToRegister(ISO_Channel,
			   (readQuadletFromRegister(ISO_Channel)
			    & 0xf0003f00) | quad);

    continuousShot(cont);

    return *this;
}
    
//! IIDCカメラのデータ転送速度を返す
/*!
  \return	データ転送速度
*/
IIDCCamera::Speed
IIDCCamera::getSpeed() const
{
    const auto	quad  = readQuadletFromRegister(ISO_Channel);
    const auto	speed = (quad & (0x1u << 15) ? quad & 0x7
					     : (quad >> 24) & 0x3);
    
    switch (speed)
    {
      case SPD_100M:
	return SPD_100M;
      case SPD_200M:
	return SPD_200M;
      case SPD_400M:
	return SPD_400M;
      case SPD_800M:
	return SPD_800M;
      case SPD_1_6G:
	return SPD_1_6G;
      case SPD_3_2G:
	return SPD_3_2G;
    }

    throw std::runtime_error("Unknown speed!!");

    return SPD_100M;
}

//! 指定された画像フォーマットにおいてサポートされているフレームレートを調べる
/*!
  \param format	対象となるフォーマット
  \return	サポートされているフレームレートを #FrameRate 型の列挙値
		のorとして返す. 指定されたフォーマット自体がこのカメラでサ
		ポートされていなければ, 0が返される.
*/
uint32_t
IIDCCamera::inquireFrameRate(Format format) const
{
    const auto	quad = inquireFrameRate_or_Format_7_Offset(format);

    switch (format)
    {
      case Format_7_0:
      case Format_7_1:
      case Format_7_2:
      case Format_7_3:
      case Format_7_4:
      case Format_7_5:
      case Format_7_6:
      case Format_7_7:
	if (quad != 0)
	    return FrameRate_x;
	break;
      default:
	break;
    }

    return quad;
}

//! 画像フォーマットとフレームレートを設定する
/*!
  画像データを出力中であった場合はそれを停止して設定を行うが, それが
  完了すれば出力を再開する.
  \param format	設定したい画像フォーマット
  \param rate	設定したいフレームレート
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setFormatAndFrameRate(Format format, FrameRate rate)
{
#ifdef DEBUG
    using namespace	std;
    
    cerr << "*** BEGIN [setFormatAndFrameRate] ***" << endl;
#endif
    checkAvailability(format, rate);
    const u_int	fmt  = (u_int(format) - u_int(YUV444_160x120)) / 0x20,
		mode = (u_int(format) - u_int(YUV444_160x120)) % 0x20 / 4;
    const auto	cont = inContinuousShot();
    
    continuousShot(false);
    
    u_int	rt = 0;
    for (u_int bit = FrameRate_1_875; bit != rate; bit >>= 1)
	++rt;
#ifdef DEBUG
    cerr << "  rt = " << dec << rt << endl;
#endif
    writeQuadletToRegister(Cur_V_Frm_Rate, rt   << 29);
    writeQuadletToRegister(Cur_V_Mode,	   mode << 29);
    writeQuadletToRegister(Cur_V_Format,   fmt  << 29);

  // Change buffer size and listen to the channel.
    u_int	packet_size = 0;
    switch (format)
    {
      case YUV444_160x120:
	_w = 160;
	_h = 120;
	_p = YUV_444;
	packet_size = 480 * sizeof(quadlet_t);
	break;
      case YUV422_320x240:
	_w = 320;
	_h = 240;
	_p = YUV_422;
	packet_size = 1280 * sizeof(quadlet_t);
	break;
      case YUV411_640x480:
	_w = 640;
	_h = 480;
	_p = YUV_411;
	packet_size = 3840 * sizeof(quadlet_t);
	break;
      case YUV422_640x480:
	_w = 640;
	_h = 480;
	_p = YUV_422;
	packet_size = 5120 * sizeof(quadlet_t);
	break;
      case RGB24_640x480:
	_w = 640;
	_h = 480;
	_p = RGB_24;
	packet_size = 7680 * sizeof(quadlet_t);
	break;
      case MONO8_640x480:
	_w = 640;
	_h = 480;
	_p = MONO_8;
	packet_size = 2560 * sizeof(quadlet_t);
	break;
      case MONO16_640x480:
	_w = 640;
	_h = 480;
	_p = MONO_16;
	packet_size = 5120 * sizeof(quadlet_t);
	break;
      case YUV422_800x600:
	_w = 800;
	_h = 600;
	_p = YUV_422;
	packet_size = 8000 * sizeof(quadlet_t);
	break;
      case RGB24_800x600:
	_w = 800;
	_h = 600;
	_p = RGB_24;
	packet_size = 12000 * sizeof(quadlet_t);
	break;
      case MONO8_800x600:
	_w = 800;
	_h = 600;
	_p = MONO_8;
	packet_size = 4000 * sizeof(quadlet_t);
	break;
      case YUV422_1024x768:
	_w = 1024;
	_h = 768;
	_p = YUV_422;
	packet_size = 12288 * sizeof(quadlet_t);
	break;
      case RGB24_1024x768:
	_w = 1024;
	_h = 768;
	_p = RGB_24;
	packet_size = 18432 * sizeof(quadlet_t);
	break;
      case MONO8_1024x768:
	_w = 1024;
	_h = 768;
	_p = MONO_8;
	packet_size = 6144 * sizeof(quadlet_t);
	break;
      case MONO16_800x600:
	_w = 800;
	_h = 600;
	_p = MONO_16;
	packet_size = 8000 * sizeof(quadlet_t);
	break;
      case MONO16_1024x768:
	_w = 1024;
	_h = 768;
	_p = MONO_16;
	packet_size = 12288 * sizeof(quadlet_t);
	break;
      case YUV422_1280x960:
	_w = 1280;
	_h = 960;
	_p = YUV_422;
	packet_size = 20480 * sizeof(quadlet_t);
	break;
      case RGB24_1280x960:
	_w = 1280;
	_h = 960;
	_p = RGB_24;
	packet_size = 30720 * sizeof(quadlet_t);
	break;
      case MONO8_1280x960:
	_w = 1280;
	_h = 960;
	_p = MONO_8;
	packet_size = 10240 * sizeof(quadlet_t);
	break;
      case YUV422_1600x1200:
	_w = 1600;
	_h = 1200;
	_p = YUV_422;
	packet_size = 32000 * sizeof(quadlet_t);
	break;
      case RGB24_1600x1200:
	_w = 1600;
	_h = 1200;
	_p = RGB_24;
	packet_size = 48000 * sizeof(quadlet_t);
	break;
      case MONO8_1600x1200:
	_w = 1600;
	_h = 1200;
	_p = MONO_8;
	packet_size = 16000 * sizeof(quadlet_t);
	break;
      case MONO16_1280x960:
	_w = 1280;
	_h = 960;
	_p = MONO_16;
	packet_size = 20480 * sizeof(quadlet_t);
	break;
      case MONO16_1600x1200:
	_w = 1600;
	_h = 1200;
	_p = MONO_16;
	packet_size = 32000 * sizeof(quadlet_t);
	break;
      case MONO8_640x480x2:
	_w = 640;
	_h = 480;
	_p = YUV_422;
	packet_size = 5120 * sizeof(quadlet_t);
	break;
      case Format_7_0:
      case Format_7_1:
      case Format_7_2:
      case Format_7_3:
      case Format_7_4:
      case Format_7_5:
      case Format_7_6:
      case Format_7_7:
      {
	const auto	fmt7info = getFormat_7_Info(format);
	_w	    = fmt7info.width;
	_h	    = fmt7info.height;
	_p	    = fmt7info.pixelFormat;
	packet_size = setFormat_7_PacketSize(format);
	rt = 7;		// frameRateによってpacket_sizeを変えないようにする.
      }
	break;
      default:
	throw std::invalid_argument("IIDCCamera::setFormat: not implemented format!!");
	break;
    }
    packet_size >>= (7 - rt);	// frameRateによってpacket_sizeを変える.
    _img_size = _w * _h;
    switch (_p)
    {
      case YUV_444:
      case RGB_24:
	_img_size *= 3;
	break;
      case YUV_422:
      case MONO_16:
      case SIGNED_MONO_16:
      case RAW_16:
	_img_size *= 2;
	break;
      case RGB_48:
      case SIGNED_RGB_48:
	_img_size *= 6;
	break;
      case YUV_411:
	(_img_size *= 3) /= 2;
	break;
      default:
	break;
    }
#ifdef DEBUG
    cerr << "  packetsize = " << packet_size << ", _img_size = " << _img_size
	 << endl;
#endif
    const u_char ch = _node->mapListenBuffer(packet_size, _img_size, NBUFFERS);

  // map時に割り当てられたチャンネル番号をカメラに設定する.
    auto	 quad = readQuadletFromRegister(ISO_Channel);
    (quad &= 0x0fffc0ff) |= ((ch << 28) | (ch << 8));
    writeQuadletToRegister(ISO_Channel, quad);

    continuousShot(cont);
    
#ifdef DEBUG
    cerr << "*** END [setFormatAndFrameRate] ***" << endl;
#endif
    return *this;
}

//! 現在カメラに設定されている画像フォーマットを返す
/*!
  \return	設定されている画像フォーマット
*/
IIDCCamera::Format
IIDCCamera::getFormat() const
{
    return
	uintToFormat(YUV444_160x120 +
		     ((readQuadletFromRegister(Cur_V_Mode)  >>29) & 0x7)*4 + 
		     ((readQuadletFromRegister(Cur_V_Format)>>29) & 0x7)*0x20);
}

//! 現在カメラに設定されているフレームレートを返す
/*!
  \return	設定されているフレームレート
*/
IIDCCamera::FrameRate
IIDCCamera::getFrameRate() const
{
    switch (getFormat())
    {
      case Format_7_0:
      case Format_7_1:
      case Format_7_2:
      case Format_7_3:
      case Format_7_4:
      case Format_7_5:
      case Format_7_6:
      case Format_7_7:
	return FrameRate_x;
      default:
	break;
    }
    return
	uintToFrameRate(0x1u << 
			(31 - ((readQuadletFromRegister(Cur_V_Frm_Rate) >> 29)
			       & 0x7)));
}

//! 指定されたFormat_7タイプのフォーマットの内容を返す
/*!
  ただし, 注目領域(ROI)の幅または高さが0の場合(カメラ電源投入直後など)は, 
  幅と高さをそれぞれの最小単位に設定し, その情報を返す. 
  \param format7 対象となるフォーマット(#Format_7_0 - #Format_7_7 のいずれか)
  \return	 指定されたフォーマットの内容
 */
IIDCCamera::Format_7_Info
IIDCCamera::getFormat_7_Info(Format format7)
{
    const auto		base = getFormat_7_BaseAddr(format7);
    quadlet_t		quad;
    Format_7_Info	fmt7info;

    quad = _node->readQuadlet(base + MAX_IMAGE_SIZE_INQ);
    fmt7info.maxWidth  = ((quad >> 16) & 0xffff);
    fmt7info.maxHeight = (quad & 0xffff);

    quad = _node->readQuadlet(base + UNIT_SIZE_INQ);
    fmt7info.unitWidth  = ((quad >> 16) & 0xffff);
    fmt7info.unitHeight = (quad & 0xffff);

    quad = _node->readQuadlet(base + UNIT_POSITION_INQ);
    fmt7info.unitU0 = ((quad >> 16) & 0xffff);
    if (fmt7info.unitU0 == 0)
	fmt7info.unitU0 = fmt7info.unitWidth;
    fmt7info.unitV0 = (quad & 0xffff);
    if (fmt7info.unitV0 == 0)
	fmt7info.unitV0 = fmt7info.unitHeight;

    quad = _node->readQuadlet(base + IMAGE_POSITION);
    fmt7info.u0 = ((quad >> 16) & 0xffff);
    fmt7info.v0 = (quad & 0xffff);

    quad = _node->readQuadlet(base + IMAGE_SIZE);
    fmt7info.width  = ((quad >> 16) & 0xffff);
    fmt7info.height = (quad & 0xffff);
    if (fmt7info.width == 0 || fmt7info.height == 0)
    {
	fmt7info.width  = fmt7info.unitWidth;
	fmt7info.height = fmt7info.unitHeight;
	_node->writeQuadlet(base + IMAGE_SIZE,
			    ((fmt7info.width << 16) & 0xffff0000) |
			    (fmt7info.height & 0xffff));
    }

    quad = _node->readQuadlet(base + COLOR_CODING_ID);
    const u_int colorCodingID = ((quad >> 24) & 0xff);
    if (colorCodingID > 31)
	throw std::runtime_error("IIDCCamera::getFormat_7_Info: Sorry, unsupported COLOR_CODING_ID!!");
    fmt7info.pixelFormat = uintToPixelFormat(0x1u << (31 - colorCodingID));

    quad = _node->readQuadlet(base + COLOR_CODING_INQ);
    fmt7info.availablePixelFormats = quad;
#ifdef DEBUG
    using namespace	std;
    
    cerr <<   "max size:    " << fmt7info.maxWidth  << 'x' << fmt7info.maxHeight
	 << "\nunit size:   " << fmt7info.unitWidth << 'x' << fmt7info.unitHeight
	 << "\norigin unit: " << fmt7info.unitU0 << 'x' << fmt7info.unitV0
	 << "\norigin:      " << fmt7info.u0	 << ',' << fmt7info.v0
	 << "\nsize:        " << fmt7info.width	 << 'x' << fmt7info.height
	 << endl;
#endif
    return fmt7info;
}

//! 指定されたFormat_7タイプのフォーマットについて, 注目領域(Region Of Interest)を設定する
/*!
  実際には, u0, v0, width, heightは, 予め決められた最小単位
  (Format_7_Info::unitU0, Format_7_Info::unitV0, Format_7_Info::unitWidth,
  Format_7_Info::unitHeight)の倍数となるように調整される. そこで, 注目領域は, 
  (u0, v0)と(u0 + width - 1, v0 + height - 1)を対角線とする矩形を含む
  最小領域となるように設定される. また, たとえwidthもしくはheightに0を指定
  しても, ROIの幅もしくは高さは, それぞれ Format_7_Info::unitWidth,
  Format_7_Info::unitHeight 以上に設定される.
  \param format7 対象となるフォーマット(#Format_7_0 - #Format_7_7 のいずれか)
  \param u0	 注目領域の左上隅の横座標
  \param v0	 注目領域の左上隅の縦座標
  \param width	 注目領域の幅
  \param height	 注目領域の高さ
  \return	 このIIDCカメラオブジェクト
 */
IIDCCamera&
IIDCCamera::setFormat_7_ROI(Format format7, u_int u0, u_int v0,
				u_int width, u_int height)
{
    const auto	fmt7info = getFormat_7_Info(format7);

    const u_int	u1 = u0 + width;		    // 右端の希望値
    u0 = fmt7info.unitU0 * (u0 / fmt7info.unitU0);  // 左端を最小単位の倍数に
    while (u0 > fmt7info.maxWidth - fmt7info.unitWidth)
	u0 -= fmt7info.unitU0;	// 右に最小幅の余白ができるよう左端を修正
    width = fmt7info.unitWidth	// 右端希望値を含むよう幅を最小単位の倍数に
	  * (u1 - u0 > 0 ? (u1 - u0 - 1) / fmt7info.unitWidth + 1 : 1);
    while (u0 + width > fmt7info.maxWidth)
	width -= fmt7info.unitWidth;	// 最大幅に収まるよう幅を修正

    const u_int	v1 = v0 + height;		    // 下端の希望値
    v0 = fmt7info.unitV0 * (v0 / fmt7info.unitV0);  // 上端を最小単位の倍数に
    while (v0 > fmt7info.maxWidth - fmt7info.unitWidth)
	v0 -= fmt7info.unitV0;	 // 下に最小高さの余白ができるよう上端を修正
    height = fmt7info.unitHeight // 下端希望値を含むよう高さを最小単位の倍数に
	   * (v1 - v0 > 0 ? (v1 - v0 - 1) / fmt7info.unitHeight + 1 : 1);
    while (v0 + height > fmt7info.maxHeight)
	height -= fmt7info.unitHeight;	// 最大高さに収まるよう高さを修正

  // 画像出力中はROIを変更できないので, もしそうであれば停止する.
    const auto	cont = inContinuousShot();
    continuousShot(false);

  // ROIを指定する.
    const auto	base = getFormat_7_BaseAddr(format7);
    _node->writeQuadlet(base + IMAGE_POSITION,
			((u0 << 16) & 0xffff0000) | (v0 & 0xffff));
    _node->writeQuadlet(base + IMAGE_SIZE,
			((width << 16) & 0xffff0000) | (height & 0xffff));
    if (getFormat() == format7)
	setFormatAndFrameRate(format7, FrameRate_x);
    
    continuousShot(cont);
    
    return *this;
}

//! 指定されたFormat_7タイプのフォーマットについて, 画素形式を設定する
/*!
  \param format7	対象となるフォーマット(#Format_7_0 - #Format_7_7 の
			いずれか)
  \param pixelFormat	画素形式
  \return		このIIDCカメラオブジェクト
 */
IIDCCamera&
IIDCCamera::setFormat_7_PixelFormat(Format format7,
					PixelFormat pixelFormat)
{
    const auto	fmt7info = getFormat_7_Info(format7);
    if (!(pixelFormat & fmt7info.availablePixelFormats))
	throw std::invalid_argument("IIDCCamera::setFormat_7_pixelFormat: unsupported pixel format!!");

  // 画像出力中はpixel formatを変更できないので, もしそうであれば停止する.
    const bool	cont = inContinuousShot();
    continuousShot(false);

  // pixel formatを指定する.
    const auto	base = getFormat_7_BaseAddr(format7);
    u_int	colorCodingID = 0;
    while ((0x1u << (31 - colorCodingID)) != pixelFormat)
	++colorCodingID;
    _node->writeQuadlet(base + COLOR_CODING_ID, (colorCodingID << 24));
    if (getFormat() == format7)
	setFormatAndFrameRate(format7, FrameRate_x);
    
    continuousShot(cont);
    
    return *this;
}

//! 指定された属性においてカメラがサポートする機能を返す
/*!
  \param feature	対象となる属性
  \return		サポートされている機能を #FeatureFunction 型の列挙値
			のorとして返す
 */
uint32_t
IIDCCamera::inquireFeatureFunction(Feature feature) const
{
    u_int	n   = (u_int(feature) - 0x800) >> 2;
    quadlet_t	inq = 0;
    if (n < 32)		// FEATURE_HI
    {
	constexpr uint32_t	Feature_Hi_Inq	= 0x404;
	inq = readQuadletFromRegister(Feature_Hi_Inq) & (0x1u << (31 - n));
    }
    else		// FEATURE_LO
    {
	constexpr uint32_t	Feature_Lo_Inq	= 0x408;
	n -= 32;
	inq = readQuadletFromRegister(Feature_Lo_Inq) & (0x1u << (31 - n));
    }
    if (inq == 0)	// Check presence of feature.
	return 0;

    return readQuadletFromRegister(feature - 0x300);
}

//! 指定された属性を1回だけ自動設定する.
/*!
  本関数を呼ぶと, 指定した属性の自動設定が直ちに開始される. 自動設定が終了
  したかどうかは, #inOnePushOperation() で知ることができる.
  \param feature	自動設定したい属性
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::onePush(Feature feature)
{
    checkAvailability(feature, Presence | One_Push);
    writeQuadletToRegister(feature,
			   readQuadletFromRegister(feature) | One_Push_bit);
    return *this;
}

//! 指定された属性をon/offする
/*!
  \param feature	on/offしたい属性
  \param enable		trueならばon, falseならばoff
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setActive(Feature feature, bool enable)
{
    if (isActive(feature) != enable)
    {
	checkAvailability(feature, Presence | OnOff);
	const auto	quad = readQuadletFromRegister(feature);
	writeQuadletToRegister(feature, (enable ? quad |  ON_OFF_bit
						: quad & ~ON_OFF_bit));
    }
    return *this;
}

//! 指定された属性を絶対値/相対値操作モードにする
/*!
  \param feature	絶対値/相対値操作モードにしたい属性
  \param enable		trueならば絶対値操作モード, falseならば相対値操作モード
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setAbsControl(Feature feature, bool enable)
{
    if (isAbsControl(feature) != enable)
    {
	checkAvailability(feature, Presence | Abs_Control);
	const auto	quad = readQuadletFromRegister(feature);
	writeQuadletToRegister(feature, (enable ? quad |  Abs_Control_bit
						: quad & ~Abs_Control_bit));
    }
    return *this;
}

//! 指定された属性を自動/手動設定モードにする
/*!
  自動設定にすると, この属性の値は環境の変化に追従して継続的に自動的に調整
  される.
  \param feature	自動/手動設定モードにしたい属性
  \param enable		trueならば自動設定モード, falseならば手動設定モード
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setAuto(Feature feature, bool enable)
{
    if (isAuto(feature) != enable)
    {
	checkAvailability(feature, Presence | Auto);
	const auto	quad = readQuadletFromRegister(feature);
	writeQuadletToRegister(feature, (enable ? quad |  A_M_Mode_bit
						: quad & ~A_M_Mode_bit));
    }
    return *this;
}

//! 指定された属性の値を設定する
/*!
  #WHITE_BALANCE, #TRIGGER_MODE の値を設定することはできない. 代わりに
  setWhiteBalance(), setTriggerMode(), setTriggerPolarity() を用いること.
  \param feature	値を設定したい属性
  \param value		設定する値
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setValue(Feature feature, u_int value)
{
    if (feature == WHITE_BALANCE || feature == TRIGGER_MODE)
	throw std::invalid_argument("TU::IIDCCamera::setValue: cannot set WHITE_BALANCE/TRIGGER_MODE value using this method!!");
    checkAvailability(feature, Presence | Manual);
    if (feature == TEMPERATURE)
	writeQuadletToRegister(TEMPERATURE,
			       (readQuadletFromRegister(TEMPERATURE) &
				0xff000fff) | ((value & 0xfff) << 12));
    else
	writeQuadletToRegister(feature,
			       (readQuadletFromRegister(feature) &
				0xfffff000) | (value & 0xfff));
    return *this;
}

//! 指定された属性の値を絶対値で設定する
/*!
  \param feature	値を設定したい属性
  \param value		設定する値
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setValue(Feature feature, float value)
{
    if (feature == WHITE_BALANCE || feature == TRIGGER_MODE)
	throw std::invalid_argument("TU::IIDCCamera::setValue: cannot set WHITE_BALANCE/TRIGGER_MODE value using this method!!");
    checkAvailability(feature, Presence | Abs_Control | Manual);
    AbsValue	val;
    val.f = value;
    writeQuadletToRegister(getAbsValueOffset(feature) + 8, val.i);
    return *this;
}
    
//! 指定された属性が1回だけの自動設定の最中であるか調べる
/*!
  \param feature	対象となる属性
  \return		#onePush() を行った属性値の自動設定が継続中であれば
			trueを, 終了していればfalseを返す.
*/
bool
IIDCCamera::inOnePushOperation(Feature feature) const
{
    checkAvailability(feature, Presence);	// One_Push を調べる必要なし
    return readQuadletFromRegister(feature) & One_Push_bit;
}

//! 指定された属性がonになっているか調べる
/*!
  \param feature	対象となる属性
  \return		onになっていればtrueを, そうでなければfalseを返す.
*/
bool
IIDCCamera::isActive(Feature feature) const
{
    checkAvailability(feature, Presence);	// OnOff を調べる必要なし
    return readQuadletFromRegister(feature) & ON_OFF_bit;
}

//! 指定された属性が絶対値操作モードになっているか調べる
/*!
  \param feature	対象となる属性
  \return		絶対値操作モードになっていればtrueを,
			そうでなければfalseを返す.
*/
bool
IIDCCamera::isAbsControl(Feature feature) const
{
    checkAvailability(feature, Presence);	// Abs_Control を調べる必要なし
    return readQuadletFromRegister(feature) & Abs_Control_bit;
}

//! 指定された属性が自動設定モードになっているか調べる
/*!
  \param feature	対象となる属性
  \return		自動設定モードになっていればtrueを, そうでなければ
			falseを返す.
*/
bool
IIDCCamera::isAuto(Feature feature) const
{
    checkAvailability(feature, Presence);	// Auto を調べる必要なし
    return readQuadletFromRegister(feature) & A_M_Mode_bit;
}

//! 指定された属性がとり得る値の範囲を調べる
/*!
  \param feature	対象となる属性
  \param min		とり得る値の最小値が返される. 
  \param max		とり得る値の最大値が返される. 
*/
void
IIDCCamera::getMinMax(Feature feature, u_int& min, u_int& max) const
{
    const auto	quad = checkAvailability(feature, Presence);
    min = (quad >> 12) & 0xfff;
    max = quad & 0xfff;
}

//! 指定された属性がとり得る値の範囲を絶対値で調べる
/*!
  \param feature	対象となる属性
  \param min		とり得る値の最小値が返される. 
  \param max		とり得る値の最大値が返される. 
*/
void
IIDCCamera::getMinMax(Feature feature, float& min, float& max) const
{
    checkAvailability(feature, Presence | Abs_Control);
    AbsValue		val;
    const uint32_t	offset = getAbsValueOffset(feature);
    val.i = readQuadletFromRegister(offset);
    min = val.f;
    val.i = readQuadletFromRegister(offset + 4);
    max = val.f;
}
    
//! 指定された属性の現在の値を調べる
/*!
  feature = #TEMPERATURE の場合は, #setValue() で設定した目標値ではなく, 
  実際値が返される. 目標値を得るには, #getAimedTemperature() を用いる.
  また, #WHITE_BALANCE, #TRIGGER_MODE の値を知ることはできない. 代わり
  に #getWhiteBalance(), #getTriggerMode(), #getTriggerPolarity() を用いる
  こと.
  \param feature	対象となる属性
  \return		現在の値
*/
template <> u_int
IIDCCamera::getValue<u_int>(Feature feature) const
{
    if (feature == WHITE_BALANCE || feature == TRIGGER_MODE)
	throw std::invalid_argument("TU::IIDCCamera::getValue: cannot get WHITE_BALANCE/TRIGGER_MODE value using this method!!");
    checkAvailability(feature, Presence | ReadOut);
    return readQuadletFromRegister(feature) & 0xfff;	// 12bit value.
}

//! 指定された属性の現在の値を絶対値で調べる
/*!
  \param feature	対象となる属性
  \return		現在の値
*/
template <> float
IIDCCamera::getValue<float>(Feature feature) const
{
    checkAvailability(feature, Presence | Abs_Control | ReadOut);
    AbsValue	val;
    val.i = readQuadletFromRegister(getAbsValueOffset(feature) + 8);
    return val.f;
}

//! ホワイトバランスの値を設定する
/*!
  \param ub		設定するU/B値
  \param vr		設定するV/R値
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setWhiteBalance(u_int ub, u_int vr)
{
    checkAvailability(WHITE_BALANCE, Presence | Manual);
    writeQuadletToRegister(WHITE_BALANCE,
			   (readQuadletFromRegister(WHITE_BALANCE) &
			    0xff000000) | ((ub & 0xfff) << 12) | (vr & 0xfff));
    return *this;
}

//! ホワイトバランスの値を絶対値で設定する
/*!
  \param ub		設定するU/B値
  \param vr		設定するV/R値
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setWhiteBalance(float ub, float vr)
{
    checkAvailability(WHITE_BALANCE, Presence | Abs_Control | Manual);
    AbsValue	fub, fvr;
    fub.f = ub;
    fvr.f = vr;
    writeQuadletToRegister(getAbsValueOffset(WHITE_BALANCE) +  8, fub.i);
    writeQuadletToRegister(getAbsValueOffset(WHITE_BALANCE) + 20, fvr.i);
    return *this;
}

//! ホワイトバランスの値を調べる
/*!
  \param ub		U/B値が返される
  \param vr		V/R値が返される
*/
void
IIDCCamera::getWhiteBalance(u_int &ub, u_int& vr) const
{
    checkAvailability(WHITE_BALANCE, Presence | ReadOut);
    const auto	quad = readQuadletFromRegister(WHITE_BALANCE);
    ub = (quad >> 12) & 0xfff;
    vr = quad & 0xfff;
}

//! ホワイトバランスの値を絶対値で調べる
/*!
  \param ub		U/B値が返される
  \param vr		V/R値が返される
*/
void
IIDCCamera::getWhiteBalance(float& ub, float& vr) const
{
    checkAvailability(WHITE_BALANCE, Presence | Abs_Control | ReadOut);
    AbsValue	val;
    val.i = readQuadletFromRegister(getAbsValueOffset(WHITE_BALANCE) +  8);
    ub = val.f;
    val.i = readQuadletFromRegister(getAbsValueOffset(WHITE_BALANCE) + 20);
    vr = val.f;
}
    
//! 色温度の目標値を調べる
/*!
  色温度の実際値を知るには, 代わりに #getValue() を用いよ.
  \return	設定されている目標値が返される.
*/
u_int
IIDCCamera::getAimedTemperature() const
{
    checkAvailability(TEMPERATURE, Presence | ReadOut);
    return (readQuadletFromRegister(TEMPERATURE) >> 12) & 0xfff;
}

//! トリガモードを設定する
/*!
  実際にカメラが外部トリガによって駆動されるためには, この関数でモード設定
  を行った後に #turnOn(#TRIGGER_MODE) を行わなければならない.
  \param mode	設定したいトリガモード
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setTriggerMode(TriggerMode mode)
{
    checkAvailability(TRIGGER_MODE, mode);
    writeQuadletToRegister(TRIGGER_MODE,
			   (readQuadletFromRegister(TRIGGER_MODE) & ~0xf0000) |
			   triggerModeValue(mode) << 16);
    return *this;
}

//! 現在設定されているトリガモードを調べる
/*!
  \return	現在設定されているトリガモード
*/
IIDCCamera::TriggerMode
IIDCCamera::getTriggerMode() const
{
    checkAvailability(TRIGGER_MODE, Presence | ReadOut);
    return uintToTriggerMode(
	       triggerModeInq(
		   (readQuadletFromRegister(TRIGGER_MODE) >> 16) & 0xf));
}

//! トリガ信号の極性を設定する
/*!
  \param highActive	trueならばhighでトリガon, falseならばlowでトリガon
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setTriggerPolarity(bool highActive)
{
    constexpr quadlet_t	Polarity_Inq    = 0x1u << 25;
    constexpr quadlet_t	HighActiveInput = 0x1u << 24;
    
    checkAvailability(TRIGGER_MODE, Presence | Polarity_Inq);
    const auto	quad = readQuadletFromRegister(TRIGGER_MODE);
    writeQuadletToRegister(TRIGGER_MODE,
			   (highActive ?
			    quad | HighActiveInput : quad & ~HighActiveInput));
    return *this;
}

//! 現在設定されているトリガ信号の極性を調べる
/*!
  \return	trueならばhighでトリガon, falseならばlowでトリガon
*/
bool
IIDCCamera::getTriggerPolarity() const
{
    checkAvailability(TRIGGER_MODE, Presence | ReadOut);
    return readQuadletFromRegister(TRIGGER_MODE) & (0x1u << 24);
}

//! ソフトウェアトリガ信号をセットする
/*!
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setSoftwareTrigger()
{
    writeQuadletToRegister(Software_Trigger, 0x1u << 31);
    return *this;
}

//! ソフトウェアトリガ信号をリセットする
/*!
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::resetSoftwareTrigger()
{
    writeQuadletToRegister(Software_Trigger, 0);
    return *this;
}

//! 指定されたストロボにおいてカメラがサポートする機能を返す
/*!
  \param strobe		対象となるストロボ
  \return		サポートされている機能を #StrobeFunction 型の列挙値
			のorとして返す
 */
uint32_t
IIDCCamera::inquireStrobeFunction(Strobe strobe) const
{
    return readQuadletFromRegister(getStrobeOffset(strobe));
}

//! 指定されたストロボをon/offする
/*!
  \param strobe		on/offしたいストロボ
  \param enable		trueならばon, falseならばoff
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setActive(Strobe strobe, bool enable)
{
    if (unlockAdvancedFeature(PointGrey_Feature_ID, 10))
    {
	const auto	val = readQuadletFromACRegister(PIO_DIRECTION);
	const uint32_t	bit = 0x1u << (31 - ((strobe - Strobe_0) >> 2));

	writeQuadletToACRegister(PIO_DIRECTION,
				 (enable ? val | bit : val & ~bit));
    }
    
    checkAvailability(strobe, Strobe_Presence | Strobe_OnOff);
    const auto	offset = getStrobeOffset(strobe);
    const auto	val    = readQuadletFromRegister(offset + 0x100);
    writeQuadletToRegister(offset + 0x100,
			   (enable ? val | ON_OFF_bit : val & ~ON_OFF_bit));
    return *this;
}

//! ストロボ信号の極性を設定する
/*!
  \param strobe		極性を設定したいストロボ
  \param highActive	trueならばストロボonでhigh, falseならばストロボonでlow
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setPolarity(Strobe strobe, bool highActive)
{
    constexpr quadlet_t	POLARITY = 0x1u << 24;

    checkAvailability(strobe, Strobe_Presence | Strobe_Polarity);
    const auto	offset = getStrobeOffset(strobe);
    const auto	val    = readQuadletFromRegister(offset + 0x100);
    writeQuadletToRegister(offset + 0x100,
			   (highActive ? val | POLARITY : val & ~POLARITY));
    return *this;
}
    
//! 露光が開始されてからストロボ信号がonになるまでの遅延時間を設定する
/*!
  \param strobe		遅延時間を設定したいストロボ
  \param delay		遅延時間
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setDelay(Strobe strobe, u_int delay)
{
    checkAvailability(strobe, Strobe_Presence);
    const auto	offset = getStrobeOffset(strobe);
    const auto	val    = readQuadletFromRegister(offset + 0x100);
    writeQuadletToRegister(offset + 0x100,
			   (val & 0xff000fff) | ((delay & 0xfff) << 12));
    return *this;
}

//! ストロボ信号がonになる継続時間を設定する
/*!
  \param strobe		継続時間を設定したいストロボ
  \param duration	継続時間
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setDuration(Strobe strobe, u_int duration)
{
    checkAvailability(strobe, Strobe_Presence);
    const auto	offset = getStrobeOffset(strobe);
    const auto	val    = readQuadletFromRegister(offset + 0x100);
    writeQuadletToRegister(offset + 0x100,
			   (val & ~0xfff) | (duration & 0xfff));
    return *this;
}

//! 指定されたストロボがonになっているか調べる
/*!
  \param strobe		対象となるストロボ
  \return		onになっていればtrueを, そうでなければfalseを返す.
*/
bool
IIDCCamera::isStrobeActive(Strobe strobe) const
{
    checkAvailability(strobe, Strobe_Presence);
    const auto	offset = getStrobeOffset(strobe);
    return readQuadletFromRegister(offset + 0x100) & ON_OFF_bit;
}

//! 現在設定されているストロボ信号の極性を調べる
/*!
  \param strobe		対象となるストロボ
  \return		trueならばストロボonでhigh, falseならばストロボonでlow
*/
bool
IIDCCamera::getPolarity(Strobe strobe) const
{
    constexpr quadlet_t	POLARITY = 0x1u << 24;

    checkAvailability(strobe, Strobe_Presence);
    const auto	offset = getStrobeOffset(strobe);
    return readQuadletFromRegister(offset + 0x100) & POLARITY;
}

//! 指定されたストロボの遅延時間および継続時間がとり得る値の範囲を調べる
/*!
  \param strobe		対象となるストロボ
  \param min		とり得る値の最小値が返される. 
  \param max		とり得る値の最大値が返される. 
*/
void
IIDCCamera::getMinMax(Strobe strobe, u_int& min, u_int& max) const
{
    checkAvailability(strobe, Strobe_Presence);
    const auto	val = inquireStrobeFunction(strobe);
    min = (val >> 12) & 0xfff;
    max = val & 0xfff;
}

//! 指定されたストロボに設定されている遅延時間を調べる
/*!
  \param strobe		対象となるストロボ
  \return		遅延時間
*/
u_int
IIDCCamera::getDelay(Strobe strobe) const
{
    checkAvailability(strobe, Strobe_Presence);
    const auto	offset = getStrobeOffset(strobe);
    return (readQuadletFromRegister(offset + 0x100) >> 12) & 0xfff;
}

//! 指定されたストロボに設定されている継続時間を調べる
/*!
  \param strobe		対象となるストロボ
  \return		継続時間
*/
u_int
IIDCCamera::getDuration(Strobe strobe) const
{
    checkAvailability(strobe, Strobe_Presence);
    const auto	offset = getStrobeOffset(strobe);
    return readQuadletFromRegister(offset + 0x100) & 0xfff;
}

//! カメラのGPIOピンからのDC3.3V出力をon/offする
/*!
  \param enable		trueならばon, falseならばoff
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::setOutputVoltage(bool enable)
{
    if (unlockAdvancedFeature(PointGrey_Feature_ID, 10))
    {
	const auto	val = readQuadletFromACRegister(OUTPUT_VOLTAGE_ENABLE);

	if (val & Presence)
	    writeQuadletToACRegister(OUTPUT_VOLTAGE_ENABLE,
				     (enable ? val | 0x1u : val & ~0x1u));
    }
    
    return *this;
}
    
//! カメラからの画像の連続的出力を開始/終了する
/*!
  #TRIGGER_MODE がonであれば, 撮影のタイミングは外部トリガ信号によって制御さ
  れる.
  \param enable	trueならば出力を開始、falseならば終了
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::continuousShot(bool enable)
{
    if (enable != inContinuousShot())
    {
	if (enable)
	    writeQuadletToRegister(ISO_EN, 0x1u << 31);
	else
	{
	    writeQuadletToRegister(ISO_EN, 0x0);
	  //flushListenBuffer();
	    _img = nullptr;
	  // 再び continuoutShot() した時に captureRaw() で使用するので, 
	  // _img_size の値は0にせずに保持する.
	}
    }
    
    return *this;
}

//! カメラから画像が出力中であるか調べる
/*!
  \return	画像出力中であればtrueを, そうでなければfalseを返す. 
*/
bool
IIDCCamera::inContinuousShot() const
{
    return readQuadletFromRegister(ISO_EN) & (0x1u << 31);
}

//! 画像を1枚だけ撮影してそれを出力する
/*!
  画像を連続的に出力中であれば, それを停止した後にあらためて1枚だけ撮影する. 
  #TRIGGER_MODE がonであれば, 撮影のタイミングは外部トリガ信号によって制御さ
  れる.
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::oneShot()
{
    checkAvailability(One_Shot_Inq);
    continuousShot(false);
    writeQuadletToRegister(One_Shot, 0x1u << 31);
    return *this;
}

//! 画像を指定された枚数だけ撮影してそれを出力する
/*!
  画像を連続的に出力中であれば, それを停止した後にあらためて撮影を開始する. 
  #TRIGGER_MODE がonであれば, 撮影のタイミングは外部トリガ信号によって制御さ
  れる.
  \param nframes	撮影したい枚数
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::multiShot(u_short nframes)
{
    checkAvailability(Multi_Shot_Inq);
    continuousShot(false);
    writeQuadletToRegister(One_Shot, (0x1u << 30) | (nframes & 0xffff));
    return *this;
}

//! 現在のカメラの設定を指定されたメモリチャンネルに記憶する
/*!
  IIDCカメラの一部には, カメラに設定した画像フォーマットや属性値などを
  カメラ内部のメモリチャンネルに記憶できるものがある. 
  \param mem_ch		値を記憶するメモリチャンネル番号. 0以上の値をとり, 
			最大値は #getMemoryChannelMax() で調べられる. 
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::saveConfig(u_int mem_ch)
{
    const u_int	max = getMemoryChannelMax();
    if (mem_ch == 0 || mem_ch > max)
	throw std::invalid_argument("TU::IIDCCamera::saveConfig: invalid memory channel!!");
    writeQuadletToRegister(Mem_Save_Ch, mem_ch << 28);
    writeQuadletToRegister(Memory_Save, 0x1u << 31);
    while ((readQuadletFromRegister(Memory_Save) & (0x1u << 31)) != 0)
	;
    return *this;
}

//! 指定されたメモリチャンネルに記憶された値をカメラに設定する
/*!
  IIDCカメラの一部には, カメラに設定した画像フォーマットや属性値などを
  カメラ内部のメモリチャンネルに記憶できるものがある.
  \param mem_ch		設定したい値を記憶しているメモリチャンネル番号. 0以上
			の値をとり, 最大値は #getMemoryChannelMax() で調べら
			れる.
  \return		このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::restoreConfig(u_int mem_ch)
{
    const u_int	max = getMemoryChannelMax();
    if (mem_ch > max)
	throw std::invalid_argument("TU::IIDCCamera::restoreConfig: invalid memory channel!!");
    writeQuadletToRegister(Cur_Mem_Ch, mem_ch << 28);
    return *this;
}

//! メモリチャンネルの最大値を調べる
/*!
  IIDCカメラの一部には, カメラに設定した画像フォーマットや属性値などを
  カメラ内部のメモリチャンネルに記憶できるものがある. 
  \return 	メモリチャンネル番号の最大値
*/
u_int
IIDCCamera::getMemoryChannelMax() const
{
    return (inquireBasicFunction() & 0xf);
}

/*
 *  Capture stuffs.
 */
//! IIDCカメラから出力された画像1枚分のデータを適当な形式に変換して取り込む
/*!
  テンプレートパラメータTは, 格納先の画像の画素形式を表す. なお, 本関数を
  呼び出す前に #snap() によってカメラからの画像を保持しておかなければならない.
  \param image	画像データを格納する画像オブジェクト. 画像の幅と高さは, 
		現在カメラに設定されている画像サイズに合わせて自動的に
		設定される. また, カメラに設定されたフォーマットの画素形式
		が画像のそれに一致しない場合は, 自動的に変換が行われる.
		サポートされている画素形式Tは, u_char, short, float, double,
		RGB, RGBA, BGR,	ABGR, YUV444, YUV422, YUV411 のいずれかである.
		また, サポートされている変換は以下のとおりであり, カメラの
		画素形式がこれ以外に設定されている場合はstd::domain_error
		例外が送出される.
		    -# #YUV_444 -> T
		    -# #YUV_422 -> T
		    -# #YUV_411 -> T
		    -# #RGB_24 -> T (YUV444, YUV422, YUV411 を除く) 
		    -# #MONO_8 -> T
		    -# #MONO_16 -> T
  \return	このIIDCカメラオブジェクト
*/
template <class T> const IIDCCamera&
IIDCCamera::operator >>(Image<T>& image) const
{
    if (!_img)
	throw std::runtime_error("TU::IIDCCamera::operator >>: no images snapped!!");
  // Transfer image data from current buffer.
    image.resize(height(), width());
    switch (pixelFormat())
    {
      case YUV_444:
      {
	auto	src = static_cast<const YUV444*>(_img);
	for (auto& line : image)
	{
	    std::copy_n(make_pixel_iterator(src), line.size(),
			make_pixel_iterator(line.begin()));
	    src += line.size();
	}
      }
	break;
      case YUV_422:
      {
	auto	src = static_cast<const YUV422*>(_img);
	for (auto& line : image)
	{
	    std::copy_n(make_pixel_iterator(src), line.size(),
			make_pixel_iterator(line.begin()));
	    src += line.size();
	}
      }
	break;
      case YUV_411:
      {
	auto	src = static_cast<const YUV411*>(_img);
	for (auto& line : image)
	{
	    std::copy_n(make_pixel_iterator(src), line.size(),
			make_pixel_iterator(line.begin()));
	    src += line.size();
	}
      }
	break;
      case RGB_24:
      {
	auto	src = static_cast<const RGB*>(_img);
	for (auto& line : image)
	{
	    std::copy_n(make_pixel_iterator(src), line.size(),
			make_pixel_iterator(line.begin()));
	    src += line.size();
	}
      }
	break;
      case MONO_8:
      case RAW_8:
      {
	auto	src = static_cast<const u_char*>(_img);
	for (auto& line : image)
	{
	    std::copy_n(make_pixel_iterator(src), line.size(),
			make_pixel_iterator(line.begin()));
	    src += line.size();
	}
      }
	break;
      case MONO_16:
      case RAW_16:
	if (_littleEndian)
	{
	    auto	src = static_cast<const u_short*>(_img);
	    for (auto& line : image)
	    {
		std::copy_n(make_pixel_iterator(src), line.size(),
			    make_pixel_iterator(line.begin()));
		src += line.size();
	    }
	}
	else
	{
	    auto	src = static_cast<const Mono16*>(_img);
	    for (auto& line : image)
	    {
		std::copy_n(make_pixel_iterator(src), line.size(),
			    make_pixel_iterator(line.begin()));
		src += line.size();
	    }
	}
	break;
      case SIGNED_MONO_16:
	if (_littleEndian)
	{
	    auto	src = static_cast<const short*>(_img);
	    for (auto& line : image)
	    {
		std::copy_n(make_pixel_iterator(src), line.size(),
			    make_pixel_iterator(line.begin()));
		src += line.size();
	    }
	}
	else
	{
	    auto	src = static_cast<const Mono16*>(_img);
	    for (auto& line : image)
	    {
		std::copy_n(make_pixel_iterator(src), line.size(),
			    make_pixel_iterator(line.begin()));
		src += line.size();
	    }
	}
	break;
      default:
	throw std::domain_error("TU::IIDCCamera::operator >>: unknown pixel format!!");
	break;
    }

    return *this;
}

//! IIDCカメラから出力された画像をRGB形式カラー画像として取り込む
/*!
  #operator >>() との違いは, カメラが #Bayer パターンをサポートしたカメラで
  ある場合, #Bayer パターンからRGB形式への変換を行うことである.
  テンプレートパラメータTは, 格納先の画像の画素形式を表す. なお, 本関数を
  呼び出す前に #snap() によってカメラからの画像を保持しておかなければならない.
  \param image	画像データを格納する画像オブジェクト. 画像の幅と高さは,
		現在カメラに設定されている画像サイズに合わせて自動的に
		設定される. サポートされている画素形式Tは RGB, RGBA,
		BGR, ABGR のいずれかである.
  \return	このIIDCカメラオブジェクト
*/
template <class T> const IIDCCamera&
IIDCCamera::captureRGBImage(Image<T>& image) const
{
    if (!_img)
	throw std::runtime_error("TU::IIDCCamera::captureRGBImage: no images snapped!!");
  // Transfer image data from current buffer.
    image.resize(height(), width());
    switch (pixelFormat())
    {
      case MONO_8:
      {
	auto	p = static_cast<const u_char*>(_img);
	  
	switch (_bayer)
	{
	  case RGGB:
	    bayerDecodeRGGB(make_flat_row_iterator(p, width()),
			    make_flat_row_iterator(p + height()*width(),
						   width()),
			    image.begin());
	    break;
	  case BGGR:
	    bayerDecodeBGGR(make_flat_row_iterator(p, width()),
			    make_flat_row_iterator(p + height()*width(),
						   width()),
			    image.begin());
	    break;
	  case GRBG:
	    bayerDecodeGRBG(make_flat_row_iterator(p, width()),
			    make_flat_row_iterator(p + height()*width(),
						   width()),
			    image.begin());
	    break;
	  case GBRG:
	    bayerDecodeGBRG(make_flat_row_iterator(p, width()),
			    make_flat_row_iterator(p + height()*width(),
						   width()),
			    image.begin());
	    break;
	  default:
	    *this >> image;
	    break;
	}
      }
	break;
	
      case MONO_16:
	if (_littleEndian)
	{
	    auto	p = static_cast<const u_short*>(_img);
	  
	    switch (_bayer)
	    {
	      case RGGB:
		bayerDecodeRGGB(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
						       width()),
				image.begin());
		break;
	      case BGGR:
		bayerDecodeBGGR(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
						       width()),
				image.begin());
		break;
	      case GRBG:
		bayerDecodeGRBG(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
						       width()),
				image.begin());
		break;
	      case GBRG:
		bayerDecodeGBRG(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
						       width()),
				image.begin());
		break;
	      default:
		*this >> image;
		break;
	    }
	}
	else
	{
	    auto	p = static_cast<const Mono16*>(_img);

	    switch (_bayer)
	    {
	      case RGGB:
		bayerDecodeRGGB(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
							  width()),
				   image.begin());
		break;
	      case BGGR:
		bayerDecodeBGGR(make_flat_row_iterator(p, width()),
				   make_flat_row_iterator(p + height()*width(),
							  width()),
				   image.begin());
		break;
	      case GRBG:
		bayerDecodeGRBG(make_flat_row_iterator(p, width()),
				   make_flat_row_iterator(p + height()*width(),
							  width()),
				   image.begin());
		break;
	      case GBRG:
		bayerDecodeGBRG(make_flat_row_iterator(p, width()),
				   make_flat_row_iterator(p + height()*width(),
							  width()),
				   image.begin());
		break;
	      default:
		*this >> image;
		break;
	    }
	}
	break;

      default:
	*this >> image;
	break;
    }

    return *this;
}

//! IIDCカメラから出力された画像1枚分のデータをなんら変換を行わずに取り込む
/*!
  本関数を呼び出す前に #snap() によってカメラからの画像を保持しておかなければ
  ならない.
  \param image	画像データの格納領域へのポインタ. #width(), #height() および
		#pixelFormat() を用いて画像のサイズと画素の形式を調べて
		画像1枚分の領域を確保しておくのは, ユーザの責任である.
  \return	このIIDCカメラオブジェクト
*/
const IIDCCamera&
IIDCCamera::captureRaw(void* image) const
{
    if (!_img)
	throw std::runtime_error("TU::IIDCCamera::captureRaw: no images snapped!!");
  // Transfer image data from current buffer.
    memcpy(image, _img, _img_size);

    return *this;
}

//! IIDCカメラから出力されたBayerパターン画像1枚分のデータをRGB形式に変換して取り込む
/*!
  本関数を呼び出す前に #snap() によってカメラからの画像を保持しておかなければ
  ならない.
  \param image	画像データの格納領域へのポインタ. #width(), #height() および
		#pixelFormat() を用いて画像のサイズと画素の形式を調べて
		画像1枚分の領域を確保しておくのは, ユーザの責任である.
		画像データは, 各画素毎に R, G, B (各 1 byte)の順で格納され
		る. カメラの画素形式が #MONO_8 または #MONO_16 以外に設定され
		ている場合はstd::domain_error例外が送出される.
  \return	このIIDCカメラオブジェクト
*/
const IIDCCamera&
IIDCCamera::captureBayerRaw(void* image) const
{
    if (!_img)
	throw std::runtime_error("TU::IIDCCamera::captureBayerRaw: no images snapped!!");

    auto	rgb = static_cast<RGB*>(image);

  // Transfer image data from current buffer.
    switch (pixelFormat())
    {
      case MONO_8:
      {
	auto	p = static_cast<const u_char*>(_img);
	  
	switch (_bayer)
	{
	  case RGGB:
	    bayerDecodeRGGB(make_flat_row_iterator(p, width()),
			    make_flat_row_iterator(p + height()*width(),
						   width()),
			    make_flat_row_iterator(rgb, width()));
	    break;
	  case BGGR:
	    bayerDecodeBGGR(make_flat_row_iterator(p, width()),
			    make_flat_row_iterator(p + height()*width(),
						   width()),
			    make_flat_row_iterator(rgb, width()));
	    break;

	  case GRBG:
	    bayerDecodeGRBG(make_flat_row_iterator(p, width()),
			    make_flat_row_iterator(p + height()*width(),
						   width()),
			    make_flat_row_iterator(rgb, width()));
	    break;
	  case GBRG:
	    bayerDecodeGBRG(make_flat_row_iterator(p, width()),
			    make_flat_row_iterator(p + height()*width(),
						   width()),
			    make_flat_row_iterator(rgb, width()));
	    break;
	  default:
	    for (int n = width() * height(); n-- > 0; )
	    {
		rgb->r = rgb->g = rgb->b = *p++;
		++rgb;
	    }
	    break;
	}
      }
	break;
	
      case MONO_16:
	if (_littleEndian)
	{
	    auto	p = static_cast<const u_short*>(_img);
	    
	    switch (_bayer)
	    {
	      case RGGB:
		bayerDecodeRGGB(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
						       width()),
				make_flat_row_iterator(rgb, width()));
		break;
	      case BGGR:
		bayerDecodeBGGR(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
						       width()),
				make_flat_row_iterator(rgb, width()));
		break;
	      case GRBG:
		bayerDecodeGRBG(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
						       width()),
				make_flat_row_iterator(rgb, width()));
		break;
	      case GBRG:
		bayerDecodeGBRG(make_flat_row_iterator(p, width()),
				make_flat_row_iterator(p + height()*width(),
						       width()),
				make_flat_row_iterator(rgb, width()));
		break;
	      default:
		for (int n = width() * height(); n-- > 0; )
		{
		    rgb->r = rgb->g = rgb->b = *p++;
		    ++rgb;
		}
	        break;
	    }
	}
	else
	{
	    auto	p = static_cast<const Mono16*>(_img);

	    switch (_bayer)
	    {
	      case RGGB:
		bayerDecodeRGGB(make_flat_row_iterator(p, width()),
				   make_flat_row_iterator(p + height()*width(),
							  width()),
				   make_flat_row_iterator(rgb, width()));
		break;
	      case BGGR:
		bayerDecodeBGGR(make_flat_row_iterator(p, width()),
				   make_flat_row_iterator(p + height()*width(),
							  width()),
				   make_flat_row_iterator(rgb, width()));
		break;
	      case GRBG:
		bayerDecodeGRBG(make_flat_row_iterator(p, width()),
				   make_flat_row_iterator(p + height()*width(),
							  width()),
				   make_flat_row_iterator(rgb, width()));
		break;
	      case GBRG:
		bayerDecodeGBRG(make_flat_row_iterator(p, width()),
				   make_flat_row_iterator(p + height()*width(),
							  width()),
				   make_flat_row_iterator(rgb, width()));
		break;
	      default:
		for (int n = width() * height(); n-- > 0; )
		{
		    rgb->r = rgb->g = rgb->b = *p++;
		    ++rgb;
		}
	        break;
	    }
	}
	break;
	
      default:
	throw std::domain_error("TU::IIDCCamera::captureBayerRaw: must be MONO_8 or MONO_16 format!!");
	break;
    }
    
    return *this;
}

//! IIDCカメラからの画像の先頭4byteにタイムスタンプを埋め込む/埋め込まないを指定する
/*!
  Point Grey社のカメラのみに有効
  \param enable	trueならば埋め込む, falseならば埋め込まない
  \return	このIIDCカメラオブジェクト
*/
IIDCCamera&
IIDCCamera::embedTimestamp(bool enable)
{
    if (unlockAdvancedFeature(PointGrey_Feature_ID, 10))
    {
	const auto	val = readQuadletFromACRegister(FRAME_INFO);
	if (val & Presence)
	    writeQuadletToACRegister(FRAME_INFO,
				     (enable ? val | 0x1u : val & ~0x1u));
    }
    return *this;
}

uint64_t
IIDCCamera::cycletimeToLocaltime(uint32_t cycletime) const
{
  // 現在のcycletimeとlocaltimeを獲得する.
    uint64_t	localtime0;
    const auto	cycletime0 = getCycletime(localtime0);

  // 現時刻と指定された時刻のサイクル時刻をサブサイクル値に直し，
  // 両者のずれを求める.
    const auto	cycle0 = cycletime_to_cycle(cycletime0);
    const auto	cycle  = cycletime_to_cycle(cycletime);
    const auto	diff   = (cycle0 + (128LL*8000LL) - cycle) % (128LL*8000LL);

  // ずれをmicro sec単位に直して(1 cycle = 125 usec)現在時刻から差し引く.
    return localtime0 - 125LL*diff;
}
    
//! unsinged intの値を同じビットパターンを持つ #Format に直す
/*!
  \param format	#Format に直したいunsigned int値
  \return	#Format 型のenum値
 */
IIDCCamera::Format
IIDCCamera::uintToFormat(u_int format)
{
    switch (format)
    {
      case YUV444_160x120:
	return YUV444_160x120;
      case YUV422_320x240:
	return YUV422_320x240;
      case YUV411_640x480:
	return YUV411_640x480;
      case YUV422_640x480:
	return YUV422_640x480;
      case RGB24_640x480:
	return RGB24_640x480;
      case MONO8_640x480:
	return MONO8_640x480;
      case MONO16_640x480:
	return MONO16_640x480;
      case YUV422_800x600:
	return YUV422_800x600;
      case RGB24_800x600:
	return RGB24_800x600;
      case MONO8_800x600:
	return MONO8_800x600;
      case YUV422_1024x768:
	return YUV422_1024x768;
      case RGB24_1024x768:
	return RGB24_1024x768;
      case MONO8_1024x768:
	return MONO8_1024x768;
      case MONO16_800x600:
	return MONO16_800x600;
      case MONO16_1024x768:
	return MONO16_1024x768;
      case YUV422_1280x960:
	return YUV422_1280x960;
      case RGB24_1280x960:
	return RGB24_1280x960;
      case MONO8_1280x960:
	return MONO8_1280x960;
      case YUV422_1600x1200:
	return YUV422_1600x1200;
      case RGB24_1600x1200:
	return RGB24_1600x1200;
      case MONO8_1600x1200:
	return MONO8_1600x1200;
      case MONO16_1280x960:
	return MONO16_1280x960;
      case MONO16_1600x1200:
	return MONO16_1600x1200;
      case Format_5_0:
	return Format_5_0;
      case MONO8_640x480x2:
	return MONO8_640x480x2;
      case Format_5_2:
	return Format_5_2;
      case Format_5_3:
	return Format_5_3;
      case Format_5_4:
	return Format_5_4;
      case Format_5_5:
	return Format_5_5;
      case Format_5_6:
	return Format_5_6;
      case Format_5_7:
	return Format_5_7;
      case Format_7_0:
	return Format_7_0;
      case Format_7_1:
	return Format_7_1;
      case Format_7_2:
	return Format_7_2;
      case Format_7_3:
	return Format_7_3;
      case Format_7_4:
	return Format_7_4;
      case Format_7_5:
	return Format_7_5;
      case Format_7_6:
	return Format_7_6;
      case Format_7_7:
	return Format_7_7;
    }

    throw std::invalid_argument("Unknown format!!");
    
    return YUV444_160x120;
}

//! unsinged intの値を同じビットパターンを持つ #FrameRate に直す
/*!
  \param rate	#FrameRate に直したいunsigned int値
  \return	#FrameRate 型のenum値
 */
IIDCCamera::FrameRate
IIDCCamera::uintToFrameRate(u_int rate)
{
    switch (rate)
    {
      case FrameRate_1_875:
	return FrameRate_1_875;
      case FrameRate_3_75:
	return FrameRate_3_75;
      case FrameRate_7_5:
	return FrameRate_7_5;
      case FrameRate_15:
	return FrameRate_15;
      case FrameRate_30:
	return FrameRate_30;
      case FrameRate_60:
	return FrameRate_60;
      case FrameRate_120:
	return FrameRate_120;
      case FrameRate_240:
	return FrameRate_240;
      case FrameRate_x:
	return FrameRate_x;
    }

    throw std::invalid_argument("Unknown frame rate!!");
    
    return FrameRate_1_875;
}

//! unsinged intの値を同じビットパターンを持つ #Feature に直す
/*!
  \param feature	#Feature に直したいunsigned int値
  \return		#Feature 型のenum値
 */
IIDCCamera::Feature
IIDCCamera::uintToFeature(u_int feature)
{
    switch (feature)
    {
      case BRIGHTNESS:
	return BRIGHTNESS;
      case AUTO_EXPOSURE:
	return AUTO_EXPOSURE;
      case SHARPNESS:
	return SHARPNESS;
      case WHITE_BALANCE:
	return WHITE_BALANCE;
      case HUE:
	return HUE;
      case SATURATION:
	return SATURATION;
      case GAMMA:
	return GAMMA;
      case SHUTTER:
	return SHUTTER;
      case GAIN:
	return GAIN;
      case IRIS:
	return IRIS;
      case FOCUS:
	return FOCUS;
      case TEMPERATURE:
	return TEMPERATURE;
      case TRIGGER_MODE:
	return TRIGGER_MODE;
      case TRIGGER_DELAY:
	return TRIGGER_DELAY;
      case FRAME_RATE:
	return FRAME_RATE;
      case ZOOM:
	return ZOOM;
      case PAN:
	return PAN;
      case TILT:
	return TILT;
      case OPTICAL_FILTER:
	return OPTICAL_FILTER;
      case CAPTURE_SIZE:
	return CAPTURE_SIZE;
      case CAPTURE_QUALITY:
	return CAPTURE_QUALITY;
    }

    throw std::invalid_argument("Unknown feature!!");
    
    return BRIGHTNESS;
}

//! unsinged intの値を同じビットパターンを持つ #TriggerMode に直す
/*!
  \param triggerMode	#TriggerMode に直したいunsigned int値
  \return		#TriggerMode 型のenum値
 */
IIDCCamera::TriggerMode
IIDCCamera::uintToTriggerMode(u_int triggerMode)
{
    switch (triggerMode)
    {
      case Trigger_Mode0:
	return Trigger_Mode0;
      case Trigger_Mode1:
	return Trigger_Mode1;
      case Trigger_Mode2:
	return Trigger_Mode2;
      case Trigger_Mode3:
	return Trigger_Mode3;
      case Trigger_Mode4:
	return Trigger_Mode4;
      case Trigger_Mode5:
	return Trigger_Mode5;
      case Trigger_Mode14:
	return Trigger_Mode14;
    }

    throw std::invalid_argument("Unknown trigger mode!!");
    
    return Trigger_Mode0;
}
 
//! unsinged intの値を同じビットパターンを持つ #PixelFormat に直す
/*!
  \param pixelFormat	#PixelFormat に直したいunsigned int値
  \return		#PixelFormat 型のenum値
*/
IIDCCamera::PixelFormat
IIDCCamera::uintToPixelFormat(u_int pixelFormat)
{
    switch (pixelFormat)
    {
      case MONO_8:
	return MONO_8;
      case YUV_411:
	return YUV_411;
      case YUV_422:
	return YUV_422;
      case YUV_444:
	return YUV_444;
      case RGB_24:
	return RGB_24;
      case MONO_16:
	return MONO_16;
      case RGB_48:
	return RGB_48;
      case SIGNED_MONO_16:
	return SIGNED_MONO_16;
      case SIGNED_RGB_48:
	return SIGNED_RGB_48;
      case RAW_8:
	return RAW_8;
      case RAW_16:
	return RAW_16;
    }

    throw std::invalid_argument("Unknown pixel format!!");
    
    return MONO_8;
}

//! 指定された出力ストリームにサイクルタイムを出力する．
/*!
 \param	out		出力ストリーム
 \param cycletime	サイクルタイム(sec: 7, cycle: 13, subcycle: 12 bits)
 \return		outに指定した出力ストリーム
*/
std::ostream&
IIDCCamera::printCycletime(std::ostream& out, uint32_t cycletime)
{
    const auto	sec	 = (cycletime & 0xfe000000) >> 25;
    const auto	cycle	 = (cycletime & 0x01fff000) >> 12;
    const auto	subcycle = (cycletime & 0x00000fff);

    return out << sec << '.' << cycle << '.' << subcycle;
}
    
//! 指定された出力ストリームにホストのローカルタイムを出力する．
/*!
 \param	out		出力ストリーム
 \param cycletime	ローカルタイム(microseconds)
 \return		outに指定した出力ストリーム
*/
std::ostream&
IIDCCamera::printLocaltime(std::ostream& out, uint64_t localtime)
{
    const auto	usec = localtime % 1000;
    const auto	tmp  = localtime / 1000;
    const auto	msec = tmp % 1000;
    const auto	sec  = tmp / 1000;

    return out << sec << '.' << msec << '.' << usec;
}
    
uint32_t
IIDCCamera::getAbsValueOffset(Feature feature) const
{
    const uint32_t	offset = (feature < ZOOM ? 0x100 : 0xc0);

    return CSR_REGISTER_BASE + 4 * readQuadletFromRegister(feature - offset)
	 - _cmdRegBase;
}
    
uint32_t
IIDCCamera::getStrobeOffset(Strobe strobe) const
{
    return CSR_REGISTER_BASE
	 + 4 * readQuadletFromRegister(Strobe_Output_Quadlet_Offset) + strobe
	 - _cmdRegBase;
}
    
IIDCCamera::nodeaddr_t
IIDCCamera::getFormat_7_BaseAddr(Format format7) const
{
    if (format7 < Format_7_0)
	throw std::invalid_argument("IIDCCamera::getFormat_7_BaseAddr: not Format_7_x!!");

    const auto	offset = 4 * inquireFrameRate_or_Format_7_Offset(format7);
    if (offset == 0)
	throw std::invalid_argument("IIDCCamera::getFormat_7_BaseAddr: unsupported Format_7_x!!");
    
    return CSR_REGISTER_BASE + offset;
}

u_int
IIDCCamera::setFormat_7_PacketSize(Format format7)
{
    constexpr quadlet_t	Presence    = 0x1u << 31;
    constexpr quadlet_t	Setting_1   = 0x1u << 30;
    constexpr quadlet_t	ErrorFlag_1 = 0x1u << 23;
    constexpr quadlet_t	ErrorFlag_2 = 0x1u << 22;
    const auto		base	    = getFormat_7_BaseAddr(format7);
    const bool		present	    = _node->readQuadlet(base + VALUE_SETTING)
				    & Presence;
    if (present)
    {
	_node->writeQuadlet(base + VALUE_SETTING, Setting_1);
	quadlet_t	quad;
	while ((quad = _node->readQuadlet(base + VALUE_SETTING)) & Setting_1)
	    ;
	if (quad & ErrorFlag_1)
	    throw std::runtime_error("IIDCCamera::setFormat_7_PacketSize: failed to read bytePerPacket value!!");
    }

    const u_int	recBytePerPacket = _node->readQuadlet(base + BYTE_PER_PACKET)
				 & 0xffff;
    if (recBytePerPacket != 0)	// 推奨値が有効ならば...
    {
      // 推奨値を設定
	_node->writeQuadlet(base + BYTE_PER_PACKET, recBytePerPacket << 16);

	if (!present ||
	    !(_node->readQuadlet(base + VALUE_SETTING) & ErrorFlag_2))
	    return recBytePerPacket;
    }

    const auto	quad = _node->readQuadlet(base + PACKET_PARA_INQ);
    const auto	unitBytePerPacket = (quad & 0xffff0000) >> 16;
    const auto	maxBytePerPacket  = (quad & 0xffff);
    for (u_int n = (unitBytePerPacket != 0 ?
		    maxBytePerPacket / unitBytePerPacket : 0); n > 0; --n)
    {
	const auto	bytePerPacket = n * unitBytePerPacket;
	_node->writeQuadlet(base + BYTE_PER_PACKET, bytePerPacket << 16);

	if (!present ||
	    !(_node->readQuadlet(base + VALUE_SETTING) & ErrorFlag_2))
	    return bytePerPacket;
    }

    throw std::runtime_error("IIDCCamera::setFormat_7_PacketSize: failed to set bytePerPacket!!");
    return 0;
}

uint32_t
IIDCCamera::inquireFrameRate_or_Format_7_Offset(Format format) const
{
    const u_int		V_FORMAT_INQ	= 0x100;
    const quadlet_t	Format_0	= 0x1u << 31;
    const quadlet_t	Format_1	= 0x1u << 30;
    const quadlet_t	Format_2	= 0x1u << 29;
    const quadlet_t	Format_5	= 0x1u << 26;
    const quadlet_t	Format_6	= 0x1u << 25;
    const quadlet_t	Format_7	= 0x1u << 24;
    quadlet_t		inq		= 0;
    switch (format)	// Check presence of format.
    {
      case YUV444_160x120:
      case YUV422_320x240:
      case YUV411_640x480:
      case YUV422_640x480:
      case RGB24_640x480:
      case MONO8_640x480:
      case MONO16_640x480:
	inq = readQuadletFromRegister(V_FORMAT_INQ) & Format_0;
	break;
      case YUV422_800x600:
      case RGB24_800x600:
      case MONO8_800x600:
      case YUV422_1024x768:
      case RGB24_1024x768:
      case MONO8_1024x768:
      case MONO16_800x600:
      case MONO16_1024x768:
	inq = readQuadletFromRegister(V_FORMAT_INQ) & Format_1;
	break;
      case YUV422_1280x960:
      case RGB24_1280x960:
      case MONO8_1280x960:
      case YUV422_1600x1200:
      case RGB24_1600x1200:
      case MONO8_1600x1200:
      case MONO16_1280x960:
      case MONO16_1600x1200:
	inq = readQuadletFromRegister(V_FORMAT_INQ) & Format_2;
	break;
      case Format_5_0:
      case MONO8_640x480x2:
      case Format_5_2:
      case Format_5_3:
      case Format_5_4:
      case Format_5_5:
      case Format_5_6:
      case Format_5_7:
	inq = readQuadletFromRegister(V_FORMAT_INQ) & Format_5;
	break;
      case Format_7_0:
      case Format_7_1:
      case Format_7_2:
      case Format_7_3:
      case Format_7_4:
      case Format_7_5:
      case Format_7_6:
      case Format_7_7:
	inq = readQuadletFromRegister(V_FORMAT_INQ) & Format_7;
	break;
    }
    if (inq == 0)
	return 0;

    const u_int		V_MODE_INQ_0	= 0x180;
    const u_int		V_MODE_INQ_1	= 0x184;
    const u_int		V_MODE_INQ_2	= 0x188;
    const u_int		V_MODE_INQ_5	= 0x194;
    const u_int		V_MODE_INQ_6	= 0x198;
    const u_int		V_MODE_INQ_7	= 0x19c;
    const quadlet_t	Mode_0		= 0x1u << 31;
    const quadlet_t	Mode_1		= 0x1u << 30;
    const quadlet_t	Mode_2		= 0x1u << 29;
    const quadlet_t	Mode_3		= 0x1u << 28;
    const quadlet_t	Mode_4		= 0x1u << 27;
    const quadlet_t	Mode_5		= 0x1u << 26;
    const quadlet_t	Mode_6		= 0x1u << 25;
    const quadlet_t	Mode_7		= 0x1u << 24;
    inq = 0;
    switch (format)	// Check presence of mode.
    {
      case YUV444_160x120:
	inq = readQuadletFromRegister(V_MODE_INQ_0) & Mode_0;
	break;
      case YUV422_320x240:
	inq = readQuadletFromRegister(V_MODE_INQ_0) & Mode_1;
	break;
      case YUV411_640x480:
	inq = readQuadletFromRegister(V_MODE_INQ_0) & Mode_2;
	break;
      case YUV422_640x480:
	inq = readQuadletFromRegister(V_MODE_INQ_0) & Mode_3;
	break;
      case RGB24_640x480:
	inq = readQuadletFromRegister(V_MODE_INQ_0) & Mode_4;
	break;
      case MONO8_640x480:
	inq = readQuadletFromRegister(V_MODE_INQ_0) & Mode_5;
	break;
      case MONO16_640x480:
	inq = readQuadletFromRegister(V_MODE_INQ_0) & Mode_6;
	break;
      case YUV422_800x600:
	inq = readQuadletFromRegister(V_MODE_INQ_1) & Mode_0;
	break;
      case RGB24_800x600:
	inq = readQuadletFromRegister(V_MODE_INQ_1) & Mode_1;
	break;
      case MONO8_800x600:
	inq = readQuadletFromRegister(V_MODE_INQ_1) & Mode_2;
	break;
      case YUV422_1024x768:
	inq = readQuadletFromRegister(V_MODE_INQ_1) & Mode_3;
	break;
      case RGB24_1024x768:
	inq = readQuadletFromRegister(V_MODE_INQ_1) & Mode_4;
	break;
      case MONO8_1024x768:
	inq = readQuadletFromRegister(V_MODE_INQ_1) & Mode_5;
	break;
      case MONO16_800x600:
	inq = readQuadletFromRegister(V_MODE_INQ_1) & Mode_6;
	break;
      case MONO16_1024x768:
	inq = readQuadletFromRegister(V_MODE_INQ_1) & Mode_7;
	break;
      case YUV422_1280x960:
	inq = readQuadletFromRegister(V_MODE_INQ_2) & Mode_0;
	break;
      case RGB24_1280x960:
	inq = readQuadletFromRegister(V_MODE_INQ_2) & Mode_1;
	break;
      case MONO8_1280x960:
	inq = readQuadletFromRegister(V_MODE_INQ_2) & Mode_2;
	break;
      case YUV422_1600x1200:
	inq = readQuadletFromRegister(V_MODE_INQ_2) & Mode_3;
	break;
      case RGB24_1600x1200:
	inq = readQuadletFromRegister(V_MODE_INQ_2) & Mode_4;
	break;
      case MONO8_1600x1200:
	inq = readQuadletFromRegister(V_MODE_INQ_2) & Mode_5;
	break;
      case MONO16_1280x960:
	inq = readQuadletFromRegister(V_MODE_INQ_2) & Mode_6;
	break;
      case MONO16_1600x1200:
	inq = readQuadletFromRegister(V_MODE_INQ_2) & Mode_7;
	break;
      case Format_5_0:
	inq = readQuadletFromRegister(V_MODE_INQ_5) & Mode_0;
	break;
      case MONO8_640x480x2:
	inq = readQuadletFromRegister(V_MODE_INQ_5) & Mode_1;
	break;
      case Format_5_2:
	inq = readQuadletFromRegister(V_MODE_INQ_5) & Mode_2;
	break;
      case Format_5_3:
	inq = readQuadletFromRegister(V_MODE_INQ_5) & Mode_3;
	break;
      case Format_5_4:
	inq = readQuadletFromRegister(V_MODE_INQ_5) & Mode_4;
	break;
      case Format_5_5:
	inq = readQuadletFromRegister(V_MODE_INQ_5) & Mode_5;
	break;
      case Format_5_6:
	inq = readQuadletFromRegister(V_MODE_INQ_5) & Mode_6;
	break;
      case Format_5_7:
	inq = readQuadletFromRegister(V_MODE_INQ_5) & Mode_7;
	break;
      case Format_7_0:
	inq = readQuadletFromRegister(V_MODE_INQ_7) & Mode_0;
	break;
      case Format_7_1:
	inq = readQuadletFromRegister(V_MODE_INQ_7) & Mode_1;
	break;
      case Format_7_2:
	inq = readQuadletFromRegister(V_MODE_INQ_7) & Mode_2;
	break;
      case Format_7_3:
	inq = readQuadletFromRegister(V_MODE_INQ_7) & Mode_3;
	break;
      case Format_7_4:
	inq = readQuadletFromRegister(V_MODE_INQ_7) & Mode_4;
	break;
      case Format_7_5:
	inq = readQuadletFromRegister(V_MODE_INQ_7) & Mode_5;
	break;
      case Format_7_6:
	inq = readQuadletFromRegister(V_MODE_INQ_7) & Mode_6;
	break;
      case Format_7_7:
	inq = readQuadletFromRegister(V_MODE_INQ_7) & Mode_7;
	break;
    }
    if (inq == 0)
	return 0;
    
    return readQuadletFromRegister(format);
}

//! カメラベンダー依存の機能へのアクセス制限を解除してそれを使えるようにする
/*
  \param featureId	アクセス制限を解除したい機能を表す48bitのID. 
  \param timeout	解除してからまたロックされるまでのタイムアウト値
			(単位: msec)
  \return		解除に成功すればtrueを, 失敗するかこの機能自体が
			存在しなければfalseを返す.
*/
bool
IIDCCamera::unlockAdvancedFeature(uint64_t featureId, u_int timeout) const
{
    return _acRegBase;
  /*
    if (_acRegBase == 0)
	return false;

    writeQuadletToACRegister(0, featureId >> 16);	// upper 32bits
    writeQuadletToACRegister(4, (featureId << 16) | 0xf000 | (timeout & 0xfff));
    u_int	busId_nodeId = readQuadletFromACRegister(0) >> 16;

    std::cerr << "nodeId: " << std::hex << _node->nodeId() << std::endl;
    std::cerr << "busId_nodeId: " << std::hex << busId_nodeId << std::endl;
    
    return _node->nodeId() == busId_nodeId;
  */
}

/************************************************************************
*  static member variables of IIDCCamera				*
************************************************************************/
constexpr IIDCCamera::FormatName	IIDCCamera::formatNames[];
constexpr IIDCCamera::FrameRateName	IIDCCamera::frameRateNames[];
constexpr IIDCCamera::PixelFormatName	IIDCCamera::pixelFormatNames[];
constexpr IIDCCamera::FeatureName	IIDCCamera::featureNames[];
constexpr IIDCCamera::TriggerModeName	IIDCCamera::triggerModeNames[];

/************************************************************************
*  global functions							*
************************************************************************/
//! 現在のカメラの設定をストリームに書き出す
/*!
  \param out		出力ストリーム
  \param camera		対象となるカメラ
  \return		outで指定した出力ストリーム
*/
std::ostream&
operator <<(std::ostream& out, const IIDCCamera& camera)
{
    using namespace	std;

    out << showbase << hex << camera.globalUniqueId()	// Write glob. uniq. ID.
	<< ' '
	<< find_if(begin(IIDCCamera::formatNames),	// Write format.
		   end(IIDCCamera::formatNames),
		   [&](IIDCCamera::FormatName fmt)
		   {
		       return camera.getFormat() == fmt.format;
		   })->name
	<< ' '
	<< find_if(begin(IIDCCamera::frameRateNames),	// Write frame rate.
		   end(IIDCCamera::frameRateNames),
		   [&](IIDCCamera::FrameRateName rt)
		   {
		       return camera.getFrameRate() == rt.frameRate;
		   })->name
	<< dec;
    
    for (const auto& feature : IIDCCamera::featureNames)
    {
	const auto	inq = camera.inquireFeatureFunction(feature.feature);

	if ((inq & IIDCCamera::Presence) &&
	    (inq & IIDCCamera::ReadOut)  &&
	    (inq & IIDCCamera::Manual))
	{
	    const auto	aut = (feature.feature == IIDCCamera::TRIGGER_MODE ?
			       camera.getTriggerPolarity()		   :
			       camera.isAuto(feature.feature));
	    const auto	abs = camera.isAbsControl(feature.feature);

	    out << ' ';
	    if (camera.isActive(feature.feature))
		out << '*';
	    if (aut)
		out << '+';
	    if (abs)
		out << '%';
	    out << feature.name << ' ';
	    
	    switch (feature.feature)
	    {
	      case IIDCCamera::TRIGGER_MODE:
		out << find_if(begin(IIDCCamera::triggerModeNames),
			       end(IIDCCamera::triggerModeNames),
			       [&](IIDCCamera::TriggerModeName mode)
			       {
				   return camera.getTriggerMode() == mode.mode;
			       })->name;
		break;
	      case IIDCCamera::WHITE_BALANCE:
		if (abs)
		{
		    float	ub, vr;
		    camera.getWhiteBalance(ub, vr);
		    out << ub << ' ' << vr;
		}
		else
		{
		    u_int	ub, vr;
		    camera.getWhiteBalance(ub, vr);
		    out << ub << ' ' << vr;
		}
	        break;
	      default:
		if (abs)
		    out << camera.getValue<float>(feature.feature);
		else
		    out << camera.getValue<u_int>(feature.feature);
		break;
	    }
	}
    }

    return out << endl;
}

//! ストリームから読み込んだ設定をカメラにセットする
/*!
  \param in		入力ストリーム
  \param camera		対象となるカメラ
  \return		inで指定した入力ストリーム
*/
std::istream&
operator >>(std::istream& in, IIDCCamera& camera)
{
    using namespace	std;

  // global unique IDを読み込む
    string	s;
    in >> s;
    const auto	uniqId = strtoull(s.c_str(), 0, 0);
    camera.initialize(uniqId);
    
  // formatを読み込む
    in >> s;
    const auto	format = find_if(begin(IIDCCamera::formatNames),
				 end(IIDCCamera::formatNames),
				 [&](IIDCCamera::FormatName fmt)
				 {
				     return s == fmt.name;
				 });
    if (format == end(IIDCCamera::formatNames))
	throw runtime_error("IIDCCamera: Unknown format[" + s + ']');

  // frame rateを読み込む
    in >> s;
    const auto	frameRate = find_if(begin(IIDCCamera::frameRateNames),
				    end(IIDCCamera::frameRateNames),
				    [&](IIDCCamera::FrameRateName rt)
				    {
					return s == rt.name;
				    });
    if (frameRate == end(IIDCCamera::frameRateNames))
	throw runtime_error("IIDCCamera: Unknown frame rate[" + s + ']');

    for (char c; (in >> TU::skipws).get(c) && (c != '\n'); )	// Parse rest
    {
      // featureのOn/Off, AUTO/Manual, Absolute/Relative を読み込む
	bool	on = false, aut = false, abs = false;
	do
	{
	    switch (c)
	    {
	      case '*':
		on = true;
		break;
	      case '+':
		aut = true;
		break;
	      case '%':
		abs = true;
		break;
	      default:
		in.putback(c);
		goto done;
	    }
	} while (in >> c);

      done:
      // feature名を読み込む
	in >> s;					// Read feature name.
	const auto	feature = find_if(begin(IIDCCamera::featureNames),
					  end(IIDCCamera::featureNames),
					  [&](IIDCCamera::FeatureName ftr)
					  {
					      return s == ftr.name;
					  });
	if (feature == end(IIDCCamera::featureNames))
	    throw runtime_error("IIDCCamera: Unknown feature[" + s + ']');

      // featureのOn/Off, AUTO/Manual, Absolute/Relative をセットする
	camera.setActive(feature->feature, on)
	      .setAbsControl(feature->feature, abs)
	      .setAuto(feature->feature, aut);

      // featureの値を読み込んでセットする
	switch (feature->feature)
	{
	  case IIDCCamera::TRIGGER_MODE:
	  {
	      in >> s;
	      const auto mode = find_if(begin(IIDCCamera::triggerModeNames),
					end(IIDCCamera::triggerModeNames),
					[&](IIDCCamera::TriggerModeName md)
					{
					    return s == md.name;
					});
	      if (mode == end(IIDCCamera::triggerModeNames))
		  throw runtime_error("IIDCCamera: Unknown trigger mode[" +
				      s + ']');
	      camera.setTriggerMode(mode->mode);
	  }
	    break;
	  case IIDCCamera::WHITE_BALANCE:
	    if (abs)
	    {
		float	ub, vr;
		in >> ub >> vr;
		camera.setWhiteBalance(ub, vr);
	    }
	    else
	    {
		u_int	ub, vr;
		in >> ub >> vr;
		camera.setWhiteBalance(ub, vr);
	    }
	    break;
	  default:
	    if (abs)
	    {
		float	val;
		in >> val;
		camera.setValue(feature->feature, val);
	    }
	    else
	    {
		u_int	val;
		in >> val;
		camera.setValue(feature->feature, val);
	    }
	    break;
	}
    }
    
    camera.setFormatAndFrameRate(format->format, frameRate->frameRate);

    return in;
}
 
template const IIDCCamera&
IIDCCamera::operator >>(Image<u_char>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<short>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<float>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<double>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<RGB>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<RGBA>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<BGR>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<ABGR>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<YUV444>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<YUV422>& image)	const	;
template const IIDCCamera&
IIDCCamera::operator >>(Image<YUV411>& image)	const	;
template const IIDCCamera&
IIDCCamera::captureRGBImage(Image<RGB>& image)	const	;
template const IIDCCamera&
IIDCCamera::captureRGBImage(Image<RGBA>& image)	const	;
template const IIDCCamera&
IIDCCamera::captureRGBImage(Image<BGR>& image)	const	;
template const IIDCCamera&
IIDCCamera::captureRGBImage(Image<ABGR>& image)	const	;
}
