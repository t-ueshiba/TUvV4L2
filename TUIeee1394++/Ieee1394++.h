/*
 *  $Id: Ieee1394++.h,v 1.2 2002-07-25 02:38:01 ueshiba Exp $
 */
#ifndef __TUIeee1394PP_h
#define __TUIeee1394PP_h

#include <libraw1394/raw1394.h>
#include <libraw1394/csr.h>
#include <video1394.h>
#ifdef HAVE_TUToolsPP
#  include "TU/Image++.h"
#else
#  include <iostream>
typedef unsigned long long	u_int64;
#endif

namespace TU
{
/************************************************************************
*  class Ieee1394Port							*
************************************************************************/
class Ieee1394Node;

/*!
  IEEE1394ポート(PCにインストールされたIEEE1394インターフェースカード
  が代表的)を表すクラス．
*/
class Ieee1394Port
{
  public:
    Ieee1394Port(int port_number, u_int delay=1)	;
    ~Ieee1394Port()					;

    u_int		nnodes()		const	;
    nodeid_t		nodeId()		const	;
    u_int		delay()			const	;
    Ieee1394Port&	setDelay(u_int delay)		;

  private:
    Ieee1394Port(const Ieee1394Port&)			;
    Ieee1394Port&	operator =(const Ieee1394Port&)	;

    raw1394handle_t	handle()		const	{return _handle;}
    int			fd()			const	{return _fd;}
    
    void	registerNode(const Ieee1394Node& node)			;
    void	unregisterNode(const Ieee1394Node& node)		;
    bool	isRegisteredNode(const Ieee1394Node& node)	const 	;
    
    const raw1394handle_t	_handle;
    const int			_fd;	  // file desc. for video1394 device.
    const Ieee1394Node*		_node[63];
    u_int			_delay;	  // delay for read/write registers.

    friend class	Ieee1394Node;
};

//! ポートに接続されているノード(このポート自身も含む)の数を返す
inline u_int
Ieee1394Port::nnodes() const
{
    return raw1394_get_nodecount(_handle);
}

//! このポートのノードIDを返す
inline nodeid_t
Ieee1394Port::nodeId() const
{
    return raw1394_get_local_id(_handle);
}

//! ポートに設定されている遅延時間を返す
/*!
  遅延時間についてはIeee1394Port()も参照．
  \return	遅延時間(単位: micro second)
*/
inline u_int
Ieee1394Port::delay()	const
{
    return _delay;
}

//! ポートに遅延時間を設定する
/*!
  遅延時間についてはIeee1394Port()も参照．
  \param delay	遅延時間(単位: micro second)
  \return	このIEEE1394ポートオブジェクト
*/
inline Ieee1394Port&
Ieee1394Port::setDelay(u_int delay)
{
    _delay = delay;
    return *this;
}

/************************************************************************
*  class Ieee1394Node							*
************************************************************************/
/*!
  IEEE1394のノードを表すクラス．一般には，より具体的な機能を持ったノード
  (ex. デジタルカメラ)を表すクラスの基底クラスとして用いられる．
*/
class Ieee1394Node
{
  protected:
  //! isochronous転送の速度
    enum Speed
    {
	SPD_100M = 0,				//!< 100Mbps
	SPD_200M = 1,				//!< 200Mbps
	SPD_400M = 2				//!< 400Mbps
    };

  public:
  //! このノードのID(IEEE1394 bus上のアドレス)を返す
  /*!
    \return	このノードのID．
   */
    nodeid_t	nodeId()			const	{return _nodeId;}
  //! このノードに割り当てられたisochronous受信用バッファのサイズを返す
  /*!
    \return	受信用バッファ1つあたりのサイズ(単位: byte)
   */
    u_int	getBufferSize()			const	{return _buf_size;}
    u_int64	globalUniqueId()		const	;
    
  protected:
    Ieee1394Node(Ieee1394Port& prt, u_int unit_spec_ID,
		   u_int channel, int sync_tag, int flag,
		   u_int64 uniqId)					;
    virtual		~Ieee1394Node()					;

    u_int		readValueFromUnitDependentDirectory(u_char key)
								const	;
    quadlet_t		readQuadlet(nodeaddr_t addr)		const	;
    void		writeQuadlet(nodeaddr_t addr, quadlet_t quad)	;
    void		mapListenBuffer(size_t packet_size,
					size_t buf_size,
					u_int nb_buffers)		;
    const u_char*	waitListenBuffer()				;
    void		requeueListenBuffer()				;
    void		flushListenBuffer()				;
    void		unmapListenBuffer()				;

  private:
    Ieee1394Node(const Ieee1394Node&)					;
    Ieee1394Node&	operator =(const Ieee1394Node&)			;

    u_int		readValueFromUnitDirectory(u_char key)	const	;
    u_int		readValueFromDirectory(u_char key,
					       u_int& offset)	const	;
    quadlet_t		readQuadletFromConfigROM(u_int offset)	const	;
    
    Ieee1394Port&	_port;
    nodeid_t		_nodeId;
    video1394_mmap	_mmap;		// mmap structure for video1394.
    u_int		_buf_size;	// buffer size excluding header.
    u_char*		_buf;		// addr. of mapped buffer.
    u_int		_current;	// index of current ready buffer.
    u_int		_nready;	// # of ready buffers.
};

inline quadlet_t
Ieee1394Node::readQuadletFromConfigROM(u_int offset) const
{
    return readQuadlet(CSR_REGISTER_BASE + CSR_CONFIG_ROM + offset);
}

/************************************************************************
*  class Ieee1394Camera							*
************************************************************************/
/*!
  1394-based Digital Camera Specification ver. 1.30に準拠したデジタルカメラ
  を表すクラス．
*/
class Ieee1394Camera : public Ieee1394Node
{
  public:
  //! カメラが出力する画像の形式
    enum Format
    {
	Format_0_0	= 0x200,	//!< 160x120 YUV(4:4:4)
	Format_0_1	= 0x204,	//!< 320x240 YUV(4:2:2)
	Format_0_2	= 0x208,	//!< 640x480 YUV(4:1:1)
	Format_0_3	= 0x20c,	//!< 640x480 YUV(4:2:2)
	Format_0_4	= 0x210,	//!< 640x480 RGB
	Format_0_5	= 0x214,	//!< 640x480 Y(mono)
	Format_0_6	= 0x218,	//!< 640x480 Y(mono16)
	Format_1_0	= 0x220,	//!< 800x600 YUV(4:2:2)
	Format_1_1	= 0x224,	//!< 800x600 RGB
	Format_1_2	= 0x228,	//!< 800x600 Y(mono)
	Format_1_3	= 0x22c,	//!< 1024x768 YUV(4:2:2)
	Format_1_4	= 0x230,	//!< 1024x768 RGB
	Format_1_5	= 0x234,	//!< 1024x768 Y(mono)
	Format_1_6	= 0x238,	//!< 800x600 Y(mono16)
	Format_1_7	= 0x23c,	//!< 1024x768 Y(mono16)
	Format_2_0	= 0x240,	//!< 1280x960 YUV(4:2:2)
	Format_2_1	= 0x244,	//!< 1280x960 RGB
	Format_2_2	= 0x248,	//!< 1280x960 Y(mono)
	Format_2_3	= 0x24c,	//!< 1600x1200 YUV(4:2:2)
	Format_2_4	= 0x250,	//!< 1600x1200 RGB
	Format_2_5	= 0x254,	//!< 1600x1200 Y(mono)
	Format_2_6	= 0x258,	//!< 1280x1024 Y(mono16)
	Format_2_7	= 0x25c,	//!< 1600x1200 Y(mono16)
	Format_7_0	= 0x2e0,	//!< カメラ機種に依存
	Format_7_1	= 0x2e4,	//!< カメラ機種に依存
	Format_7_2	= 0x2e8,	//!< カメラ機種に依存
	Format_7_3	= 0x2ec,	//!< カメラ機種に依存
	Format_7_4	= 0x2f0,	//!< カメラ機種に依存
	Format_7_5	= 0x2f4,	//!< カメラ機種に依存
	Format_7_6	= 0x2f8,	//!< カメラ機種に依存
	Format_7_7	= 0x2fc		//!< カメラ機種に依存
    };

  //! カメラのフレームレート
    enum FrameRate
    {
	FrameRate_0	= 0,		//!< 1.875fps
	FrameRate_1	= 1,		//!< 3.75fps
	FrameRate_2	= 2,		//!< 7.5fps
	FrameRate_3	= 3,		//!< 15fps
	FrameRate_4	= 4,		//!< 30fps
	FrameRate_5	= 5		//!< 60fps
    };
    
  //! 値を設定できるカメラの属性
    enum Feature
    {
	BRIGHTNESS	= 0x800,	//!< 明るさ調整
	AUTO_EXPOSURE	= 0x804,	//!< 自動露出調整
	SHARPNESS	= 0x808,	//!< 鮮明さ調整
	WHITE_BALANCE	= 0x80c,	//!< ホワイトバランス調整
	HUE		= 0x810,	//!< 色の色相調整
	SATURATION	= 0x814,	//!< 色の飽和度調整
	GAMMA		= 0x818,	//!< 輝度のガンマ補正調整
	SHUTTER		= 0x81c,	//!< シャッタースピード調整
	GAIN		= 0x820,	//!< ゲイン調整
	IRIS		= 0x824,	//!< アイリス調整
	FOCUS		= 0x828,	//!< フォーカス調整
	TEMPERATURE	= 0x82c,	//!< 色温度調整
	TRIGGER_MODE	= 0x830,	//!< 外部トリガ信号のモード
	ZOOM		= 0x880,	//!< ズーム調整
	PAN		= 0x884,	//!< パン(左右の首振り)調整
	TILT		= 0x888,	//!< チルト(上下の首振り)調整
	OPTICAL_FILTER	= 0x88c,
	CAPTURE_SIZE	= 0x8c0,
	CAPTURE_QUALITY	= 0x8c4
    };

  //! カメラの外部トリガーモード
    enum TriggerMode
    {
	Trigger_Mode0	= 0,
	Trigger_Mode1	= 1,
	Trigger_Mode2	= 2,
	Trigger_Mode3	= 3
    };

  //! カメラの外部トリガー信号の極性
    enum TriggerPolarity
    {
	LowActiveInput	= 0,		//!< lowでトリガon
	HighActiveInput	= (0x1 << 24)	//!< highでトリガon
    };

  //! 出力画像の画素の形式
    enum PixelFormat
    {
	YUV_444,			//!< YUV(4:4:4)
	YUV_422,			//!< YUV(4:2:2)
	YUV_411,			//!< YUV(4:1:1)
	RGB_24,				//!< RGB
	MONO,				//!< Y(mono)
	MONO_16				//!< Y(mono16)
    };

  //! 各属性(#Feature)についてカメラがサポートしている機能を表すビットマップ
    enum InquireFeature
    {
	Presence_Inq	= (0x1 << 31),	//!< この属性そのものをサポート
      //Abs_Control_Inq	= (0x1 << 30),	//!< この属性を値によって制御可能
	One_Push_Inq	= (0x1 << 28),	//!< one pushモードをサポート
	ReadOut_Inq	= (0x1 << 27),	//!< この属性の値を読み出しが可能
	OnOff_Inq	= (0x1 << 26),	//!< この属性のon/offが可能
	Auto_Inq	= (0x1 << 25),	//!< この属性の値の自動設定が可能
	Manual_Inq	= (0x1 << 24)	//!< この属性の値の手動設定が可能
    };
    
  //! 各フォーマット(#Format)についてサポートされているフレームレート(#FrameRate)を表すビットマップ
    enum InquireFrameRate
    {
	FrameRate_0_Inq	= (0x1 << 31),	//!< #FrameRate_0 (1.875fps)をサポート
	FrameRate_1_Inq	= (0x1 << 30),	//!< #FrameRate_1 (3.75fps)をサポート
	FrameRate_2_Inq	= (0x1 << 29),	//!< #FrameRate_2 (7.5fps)をサポート
	FrameRate_3_Inq	= (0x1 << 28),	//!< #FrameRate_3 (15fps)をサポート
	FrameRate_4_Inq	= (0x1 << 27),	//!< #FrameRate_4 (30fps)をサポート
	FrameRate_5_Inq	= (0x1 << 26)	//!< #FrameRate_5 (60fps)をサポート
    };

  //! カメラがサポートしている基本機能を表すビットマップ
    enum InquireBasicFunction
    {
	Advanced_Feature_Inq	= (0x1 << 31),	//!< カメラベンダ依存の機能
	Cam_Power_Cntl_Inq	= (0x1 << 15),	//!< 電源on/offの制御
	One_Shot_Inq		= (0x1 << 12),	//!< 画像1枚だけの撮影
	Multi_Shot_Inq		= (0x1 << 11)	//!< 指定された枚数の撮影
    };

  public:
    Ieee1394Camera(Ieee1394Port& port, u_int channel=0,
		     u_int64 uniqId=0)					;
    virtual ~Ieee1394Camera()						;

  // Power ON/OFF stuffs.
    quadlet_t		inquireBasicFunction()			const	;
    Ieee1394Camera&	powerOn()					;
    Ieee1394Camera&	powerOff()					;

  // Format and frame rate stuffs.
    quadlet_t		inquireFormat(Format format)		const	;
    Ieee1394Camera&	setFormatAndFrameRate(Format format,
					      FrameRate rate)		;
    Format		getFormat()				const	;
    FrameRate		getFrameRate()				const	;
    u_int		width()					const	;
    u_int		height()				const	;
    PixelFormat		pixelFormat()				const	;
    
  // Feature stuffs.
    quadlet_t		inquireFeature(Feature feature)		const	;
    Ieee1394Camera&	onePush(Feature feature)			;
    Ieee1394Camera&	turnOn(Feature feature)				;
    Ieee1394Camera&	turnOff(Feature feature)			;
    Ieee1394Camera&	setAutoMode(Feature feature)			;
    Ieee1394Camera&	setManualMode(Feature feature)			;
    Ieee1394Camera&	setValue(Feature feature, u_int value)		;
    bool		inOnePushOperation(Feature feature)	const	;
    bool		isTurnedOn(Feature feature)		const	;
    bool		isAuto(Feature feautre)			const	;
    void		getMinMax(Feature feature,
				  u_int& min, u_int& max)	const	;
    u_int		getValue(Feature feature)		const	;

  // White balance stuffs.
    Ieee1394Camera&	setWhiteBalance(u_int ub, u_int vr)		;
    void		getWhiteBalance(u_int& ub, u_int& vr)	const	;
    
  // Temperature stuffs.
    u_int		getAimedTemperature()			const	;
    
  // Trigger stuffs.
    Ieee1394Camera&	setTriggerMode(TriggerMode mode)		;
    TriggerMode		getTriggerMode()			const	;
    Ieee1394Camera&	setTriggerPolarity(TriggerPolarity polarity)	;
    TriggerPolarity	getTriggerPolarity()			const	;
    
  // Shotting stuffs.
    Ieee1394Camera&	continuousShot()				;
    Ieee1394Camera&	stopContinuousShot()				;
    bool		inContinuousShot()			const	;
    Ieee1394Camera&	oneShot()					;
    Ieee1394Camera&	multiShot(u_short nframes)			;

  // Configuration saving/restoring stuffs.
    Ieee1394Camera&	saveConfig(u_int mem_ch)			;
    Ieee1394Camera&	restoreConfig(u_int mem_ch)			;
    u_int		getMemoryChannelMax()			const	;

  // Capture stuffs.
    Ieee1394Camera&	snap()						;
#ifdef HAVE_TUToolsPP
    template <class T> const Ieee1394Camera&
			operator >>(Image<T>& image)		const	;
#endif
    const Ieee1394Camera&
			captureRaw(void* image)			const	;

  // Utility functions.
    static Format	uintToFormat(u_int format)			;
    static FrameRate	uintToFrameRate(u_int frameRate)		;
    static Feature	uintToFeature(u_int feature)			;
    static TriggerMode	uintToTriggerMode(u_int triggerMode)		;
    
  private:
    void	checkAvailability(Format format, FrameRate rate) const	;
    quadlet_t	checkAvailability(Feature feature, u_int inq)	 const	;
    void	checkAvailabilityOfBasicFunction(u_int inq)	 const	;
    quadlet_t	readQuadletFromRegister(u_int offset)		 const	;
    void	writeQuadletToRegister(u_int offset, quadlet_t quad)	;
    
    const nodeaddr_t	_cmdRegBase;
    u_int		_w, _h;		// width and height.
    PixelFormat		_p;		// pixel format.
    const u_char*	_buf;		// currently available image buffer.
};

//! 現在設定されている画像フォーマット(#Format)の幅を返す
inline u_int
Ieee1394Camera::width() const
{
    return _w;
}

//! 現在設定されている画像フォーマット(#Format)の高さを返す
inline u_int
Ieee1394Camera::height() const
{
    return _h;
}

//! 現在設定されている画像フォーマット(#Format)の画素形式(#PixelFormat)を返す
inline Ieee1394Camera::PixelFormat
Ieee1394Camera::pixelFormat() const
{
    return _p;
}

//! カメラがサポートしている基本機能を表すビットパターンを返す
/*!
  \return	サポートされている機能を#InquireBasicFunction型の列挙値
		のorとして返す．
 */
inline quadlet_t
Ieee1394Camera::inquireBasicFunction() const
{
    return readQuadletFromRegister(0x400);
}

//! カメラから出力される最初の画像を保持する．
/*!
  カメラからの画像出力は，continuousShot(), oneShot(), multiShot()のいずれか
  によって行われる．実際に画像データが受信されるまで，本関数は呼び出し側に
  制御を返さない．
  \return	このIEEE1394カメラオブジェクト．
 */
inline Ieee1394Camera&
Ieee1394Camera::snap()
{
    if (_buf != 0)
	requeueListenBuffer();
    _buf = waitListenBuffer();
    return *this;
}

inline void
Ieee1394Camera::checkAvailability(Format format, FrameRate rate) const
{
    using namespace	std;
    
    quadlet_t	quad = inquireFormat(format),
		inq = u_int(FrameRate_0_Inq) >> rate;
    if ((quad & inq) != inq)
	cerr << "Ieee1394Camera::checkAvailability: Incompatible combination of format[0x"
	     << hex << format
	     << "] and frame rate[0x"
	     << rate << "]!! (inq: " << inq << ")"
	     << endl;
  //	throw TUExceptionWithMessage("Ieee1394Camera::checkAvailability: Incompatible combination of frame rate and format!!");
}

inline quadlet_t
Ieee1394Camera::checkAvailability(Feature feature, u_int inq) const
{
    using namespace	std;
    
    quadlet_t	quad = inquireFeature(feature);
    if ((quad & inq) != inq)
	cerr << "Ieee1394Camera::checkAvailability: This feature[0x"
	     << hex << feature
	     << "] is not present or this field is not available (quad: 0x"
	     << quad << ", inq: " << inq << ")!!"
	     << endl;
  //	throw TUExceptionWithMessage("Ieee1394Camera::checkAvailability: This feature is not present or this field is not available!!");
    return quad;
}

inline void
Ieee1394Camera::checkAvailabilityOfBasicFunction(u_int inq) const
{
    using namespace	std;

    quadlet_t	quad = inquireBasicFunction();
    if ((quad & inq) != inq)
	cerr << "Ieee1394Camera::checkAvailabilityOfBasicFuntion: This fucntion is not present (quad: 0x"
	     << quad << ", inq: " << inq << ")!!"
	     << endl;
  //	throw TUExceptionWithMessage("Ieee1394Camera::checkAvailabilityOfBasicFunction: This function is not present!!");
}

inline quadlet_t
Ieee1394Camera::readQuadletFromRegister(u_int offset) const
{
    return readQuadlet(_cmdRegBase + offset);
}

inline void
Ieee1394Camera::writeQuadletToRegister(u_int offset, quadlet_t quad)
{
    writeQuadlet(_cmdRegBase + offset, quad);
}
 
}
#endif	/* !__TUIeee1394PP_h	*/
