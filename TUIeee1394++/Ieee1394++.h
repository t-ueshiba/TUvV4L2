/*
 *  $Id: Ieee1394++.h,v 1.8 2002-12-09 08:02:30 ueshiba Exp $
 */
#ifndef __TUIeee1394PP_h
#define __TUIeee1394PP_h

#include <libraw1394/raw1394.h>
#include <libraw1394/csr.h>
#include <video1394.h>
#include <netinet/in.h>
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
    u_int	bufferSize()			const	{return _buf_size;}
    u_int64	globalUniqueId()		const	;
    
  protected:
    Ieee1394Node(Ieee1394Port& prt, u_int unit_spec_ID,
		 u_int channel, int sync_tag, int flag, u_int64 uniqId)	;
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
  //! カメラがサポートしている基本機能を表すビットマップ
  /*! どのような基本機能がサポートされているかは，inquireBasicFunction()に
      よって知ることができる．*/
    enum BasicFunction
    {
	Advanced_Feature_Inq	= (0x1 << 31),	//!< カメラベンダ依存の機能
	Cam_Power_Cntl_Inq	= (0x1 << 15),	//!< 電源on/offの制御
	One_Shot_Inq		= (0x1 << 12),	//!< 画像1枚だけの撮影
	Multi_Shot_Inq		= (0x1 << 11)	//!< 指定された枚数の撮影
    };

  //! カメラが出力する画像の形式
    enum Format
    {
	YUV444_160x120	 = 0x200,	//!< Format_0_0: 160x120 YUV(4:4:4)
	YUV422_320x240	 = 0x204,	//!< Format_0_1: 320x240 YUV(4:2:2)
	YUV411_640x480	 = 0x208,	//!< Format_0_2: 640x480 YUV(4:1:1)
	YUV422_640x480	 = 0x20c,	//!< Format_0_3: 640x480 YUV(4:2:2)
	RGB24_640x480	 = 0x210,	//!< Format_0_4: 640x480 RGB
	MONO8_640x480	 = 0x214,	//!< Format_0_5: 640x480 Y(mono)
	MONO16_640x480	 = 0x218,	//!< Format_0_6: 640x480 Y(mono16)
	YUV422_800x600	 = 0x220,	//!< Format_1_0: 800x600 YUV(4:2:2)
	RGB24_800x600	 = 0x224,	//!< Format_1_1: 800x600 RGB
	MONO8_800x600	 = 0x228,	//!< Format_1_2: 800x600 Y(mono)
	YUV422_1024x768	 = 0x22c,	//!< Format_1_3: 1024x768 YUV(4:2:2)
	RGB24_1024x768	 = 0x230,	//!< Format_1_4: 1024x768 RGB
	MONO8_1024x768	 = 0x234,	//!< Format_1_5: 1024x768 Y(mono)
	MONO16_800x600	 = 0x238,	//!< Format_1_6: 800x600 Y(mono16)
	MONO16_1024x768	 = 0x23c,	//!< Format_1_7: 1024x768 Y(mono16)
	YUV422_1280x960	 = 0x240,	//!< Format_2_0: 1280x960 YUV(4:2:2)
	RGB24_1280x960	 = 0x244,	//!< Format_2_1: 1280x960 RGB
	MONO8_1280x960	 = 0x248,	//!< Format_2_2: 1280x960 Y(mono)
	YUV422_1600x1200 = 0x24c,	//!< Format_2_3: 1600x1200 YUV(4:2:2)
	RGB24_1600x1200	 = 0x250,	//!< Format_2_4: 1600x1200 RGB
	MONO8_1600x1200	 = 0x254,	//!< Format_2_5: 1600x1200 Y(mono)
	MONO16_1280x960	 = 0x258,	//!< Format_2_6: 1280x960 Y(mono16)
	MONO16_1600x1200 = 0x25c,	//!< Format_2_7: 1600x1200 Y(mono16)
	Format_7_0	 = 0x2e0,	//!< Format_7_0: カメラ機種に依存
	Format_7_1	 = 0x2e4,	//!< Format_7_1: カメラ機種に依存
	Format_7_2	 = 0x2e8,	//!< Format_7_2: カメラ機種に依存
	Format_7_3	 = 0x2ec,	//!< Format_7_3: カメラ機種に依存
	Format_7_4	 = 0x2f0,	//!< Format_7_4: カメラ機種に依存
	Format_7_5	 = 0x2f4,	//!< Format_7_5: カメラ機種に依存
	Format_7_6	 = 0x2f8,	//!< Format_7_6: カメラ機種に依存
	Format_7_7	 = 0x2fc	//!< Format_7_7: カメラ機種に依存
    };

  //! カメラのフレームレートを表すビットマップ
  /*! どのようなフレームレートがサポートされているかは，inquireFrameRate()
      によって知ることができる．*/
    enum FrameRate
    {
	FrameRate_1_875	= (0x1 << 31),	//!< 1.875fps
	FrameRate_3_75	= (0x1 << 30),	//!< 3.75fps
	FrameRate_7_5	= (0x1 << 29),	//!< 7.5fps
	FrameRate_15	= (0x1 << 28),	//!< 15fps
	FrameRate_30	= (0x1 << 27),	//!< 30fps
	FrameRate_60	= (0x1 << 26),	//!< 60fps
	FrameRate_x	= (0x1 << 25)	//!< 特殊なフレームレート
    };
    
  //! 出力画像の画素の形式
    enum PixelFormat
    {
	MONO_8		= (0x1 << 31),	//!< Y(mono)	 8bit/pixel
	YUV_411		= (0x1 << 30),	//!< YUV(4:1:1)	12bit/pixel
	YUV_422		= (0x1 << 29),	//!< YUV(4:2:2)	16bit/pixel
	YUV_444		= (0x1 << 28),	//!< YUV(4:4:4)	24bit/pixel
	RGB_24		= (0x1 << 27),	//!< RGB	24bit/pixel
	MONO_16		= (0x1 << 26),	//!< Y(mono16)	16bit/pixel
	RGB_48		= (0x1 << 25)	//!< RGB	48bit/pixel
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

  //! 各属性(#Feature)についてカメラがサポートしている機能を表すビットマップ
  /*! どのような機能がサポートされているかは，inquireFeatureFunction()によっ
      て知ることができる．*/
    enum FeatureFunction
    {
	Presence	= (0x1 << 31),	//!< この属性そのものをサポート
      //Abs_Control	= (0x1 << 30),	//!< この属性を値によって制御可能
	One_Push	= (0x1 << 28),	//!< one pushモードをサポート
	ReadOut		= (0x1 << 27),	//!< この属性の値を読み出しが可能
	OnOff		= (0x1 << 26),	//!< この属性のon/offが可能
	Auto		= (0x1 << 25),	//!< この属性の値の自動設定が可能
	Manual		= (0x1 << 24)	//!< この属性の値の手動設定が可能
    };
    
  //! カメラの外部トリガーモード
    enum TriggerMode
    {
	Trigger_Mode0	= 0,	//!< トリガonから#SHUTTERで指定した時間だけ蓄積
	Trigger_Mode1	= 1,	//!< トリガonからトリガoffになるまで蓄積
	Trigger_Mode2	= 2,
	Trigger_Mode3	= 3
    };

  //! カメラの外部トリガー信号の極性
    enum TriggerPolarity
    {
	LowActiveInput	= 0,		//!< lowでトリガon
	HighActiveInput	= (0x1 << 24)	//!< highでトリガon
    };

  //! 本カメラがサポートするFormat_7に関する情報(getFormat_7_Info()で得られる)
    struct Format_7_Info
    {
	u_int		maxWidth;		//!< 画像の最大幅
	u_int		maxHeight;		//!< 画像の最大高さ
	u_int		unitWidth;		//!< 画像幅の最小単位
	u_int		unitHeight;		//!< 画像高さの最小単位
	u_int		unitU0;			//!< 原点水平位置指定の最小単位
	u_int		unitV0;			//!< 原点垂直位置指定の最小単位
	u_int		u0;			//!< 原点水平位置
	u_int		v0;			//!< 原点垂直位置
	u_int		width;			//!< 画像の幅
	u_int		height;			//!< 画像の高さ
	PixelFormat	pixelFormat;		//!< 画像の画素形式
	u_int		availablePixelFormats;	//!< 利用できる画素形式
    };
    
  private:
    struct Mono16
    {
	operator u_char()		const	{return u_char(ntohs(s));}
	operator short()		const	{return ntohs(s);}
	operator float()		const	{return float(ntohs(s));}
	operator double()		const	{return double(ntohs(s));}
	      
	short	s;
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
    quadlet_t		inquireFrameRate(Format format)		const	;
    Ieee1394Camera&	setFormatAndFrameRate(Format format,
					      FrameRate rate)		;
    Format		getFormat()				const	;
    FrameRate		getFrameRate()				const	;
    u_int		width()					const	;
    u_int		height()				const	;
    PixelFormat		pixelFormat()				const	;

  // Format_7 stuffs.
    Format_7_Info	getFormat_7_Info(Format format7)	const	;
    Ieee1394Camera&	setFormat_7_ROI(Format format7, 
					u_int u0, u_int v0,
					u_int width, u_int height)	;
    Ieee1394Camera&	setFormat_7_PixelFormat(Format format7, 
						PixelFormat pixelFormat);
    
  // Feature stuffs.
    quadlet_t		inquireFeatureFunction(Feature feature)	const	;
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
    static PixelFormat	uintToPixelFormat(u_int pixelFormat)		;
    
  private:
    nodeaddr_t	getFormat_7_BaseAddr(Format format7)		 const	;
    u_int	setFormat_7_PacketSize(Format format7)			;
    quadlet_t	inquireFrameRate_or_Format_7_Offset(Format format) const;
    void	checkAvailability(Format format, FrameRate rate) const	;
    quadlet_t	checkAvailability(Feature feature, u_int inq)	 const	;
    void	checkAvailability(BasicFunction func)		 const	;
    quadlet_t	readQuadletFromRegister(u_int offset)		 const	;
    void	writeQuadletToRegister(u_int offset, quadlet_t quad)	;
    
    const nodeaddr_t	_cmdRegBase;
    u_int		_w, _h;	// width and height of current image format.
    PixelFormat		_p;	// pixel format of current image format.
    const u_char*	_buf;	// currently available image buffer.
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

//! カメラがサポートしている基本機能を返す
/*!
  \return	サポートされている機能を#BasicFunction型の列挙値のorとして
		返す．
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
    
    quadlet_t	quad = inquireFrameRate(format);
    if (!(quad & rate))
	cerr << "Ieee1394Camera::checkAvailability: Incompatible combination of format[0x"
	     << hex << format << "] and frame rate[0x" << rate << "]!!"
	     << endl;
  //	throw TUExceptionWithMessage("Ieee1394Camera::checkAvailability: Incompatible combination of frame rate and format!!");
}

inline quadlet_t
Ieee1394Camera::checkAvailability(Feature feature, u_int inq) const
{
    using namespace	std;
    
    quadlet_t	quad = inquireFeatureFunction(feature);
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
Ieee1394Camera::checkAvailability(BasicFunction func) const
{
    using namespace	std;

    quadlet_t	quad = inquireBasicFunction();
    if (!(quad & func))
	cerr << "Ieee1394Camera::checkAvailabilityOfBasicFuntion: This fucntion is not present (quad: 0x"
	     << hex << quad << ", func: " << func << ")!!"
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
