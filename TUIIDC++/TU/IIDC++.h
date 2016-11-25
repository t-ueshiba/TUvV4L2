/*
 *  $Id: IIDC++.h,v 1.1.1.1 2012-09-15 08:03:09 ueshiba Exp $
 */
/*!
  \mainpage	libTUIIDC++ - IIDCに準拠したFireWireまたはUSBベースのデジタルカメラを制御するC++ライブラリ
  \anchor	libTUIIDC

  \section copyright 著作権
  Copyright (C) 2003-2006 Toshio UESHIBA
  National Institute of Advanced Industrial Science and Technology (AIST)
 
  Written by Toshio UESHIBA <t.ueshiba@aist.go.jp>
 
  This library is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 2.1 of the
  License, or (at your option) any later version.  This library is
  distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
  License for more details.
 
  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA

  \section abstract 概要
  libTUIIDC+は，IIDCに準拠したFireWireまたはUSBベースのデジタルカメラを
  制御するC++ライブラリである. 同一または異なるバスに接続された複数のカメラを，
  それぞれ独立にコントロールすることができる. 

  実装されている主要なクラスおよびそのpublicなメンバ関数は，おおまかに
  以下のように分類される. 

  #TU::IIDCCamera - IIDCデジタルカメラを表すクラス

  - <b>基本機能</b>
    - #TU::IIDCCamera::globalUniqueId()
    - #TU::IIDCCamera::inquireBasicFunction()
    - #TU::IIDCCamera::setPower()
    - #TU::IIDCCamera::bayerTileMapping()
    - #TU::IIDCCamera::isLittleEndian()
    - #TU::IIDCCamera::setSpeed()
    - #TU::IIDCCamera::getSpeed()
  
  - <b>画像フォーマットとフレームレート</b>
    - #TU::IIDCCamera::inquireFrameRate()
    - #TU::IIDCCamera::setFormatAndFrameRate()
    - #TU::IIDCCamera::getFormat()
    - #TU::IIDCCamera::getFrameRate()
    - #TU::IIDCCamera::width()
    - #TU::IIDCCamera::height()
    - #TU::IIDCCamera::pixelFormat()

  - <b>特殊フォーマット(Format_7)</b>
    - #TU::IIDCCamera::getFormat_7_Info()
    - #TU::IIDCCamera::setFormat_7_ROI()
    - #TU::IIDCCamera::setFormat_7_PixelFormat()

  - <b>画像の撮影モードの設定</b>
    - #TU::IIDCCamera::continuousShot()
    - #TU::IIDCCamera::inContinuousShot()
    - #TU::IIDCCamera::oneShot()
    - #TU::IIDCCamera::multiShot()

  - <b>画像の撮影</b>
    - #TU::IIDCCamera::snap()

  - <b>画像の取り込み</b>
    - #TU::IIDCCamera::operator >>()
    - #TU::IIDCCamera::captureRGBImage()
    - #TU::IIDCCamera::captureDirectly()
    - #TU::IIDCCamera::captureRaw()
    - #TU::IIDCCamera::captureBayerRaw()
    - #TU::IIDCCamera::embedTimestamp()
    - #TU::IIDCCamera::getTimestamp()

  - <b>カメラの様々な機能の制御</b>
    - #TU::IIDCCamera::inquireFeatureFunction()
    - #TU::IIDCCamera::onePush()
    - #TU::IIDCCamera::inOnePushOperation()
    - #TU::IIDCCamera::setActive()
    - #TU::IIDCCamera::isActive()
    - #TU::IIDCCamera::setAbsControl()
    - #TU::IIDCCamera::isAbsControl()
    - #TU::IIDCCamera::setAuto()
    - #TU::IIDCCamera::isAuto()
    - #TU::IIDCCamera::setValue()
    - #TU::IIDCCamera::getValue()
    - #TU::IIDCCamera::getMinMax()
    - #TU::IIDCCamera::setAbsValue()
    - #TU::IIDCCamera::getAbsValue()
    - #TU::IIDCCamera::getAbsMinMax()
    - #TU::IIDCCamera::setWhiteBalance()
    - #TU::IIDCCamera::getWhiteBalance()
    - #TU::IIDCCamera::getAimedTemperature()

  - <b>トリガモード</b>
    - #TU::IIDCCamera::setTriggerMode()
    - #TU::IIDCCamera::getTriggerMode()
    - #TU::IIDCCamera::setTriggerPolarity()
    - #TU::IIDCCamera::getTriggerPolarity()

  - <b>ストロボ</b>
    - #TU::IIDCCamera::inquireStrobeFunction()
    - #TU::IIDCCamera::setActive(Strobe, bool)
    - #TU::IIDCCamera::setPolarity()
    - #TU::IIDCCamera::setDelay()
    - #TU::IIDCCamera::setDuration()
    - #TU::IIDCCamera::isStrobeActive()
    - #TU::IIDCCamera::getPolarity()
    - #TU::IIDCCamera::getMinMax(Strobe, u_int&, u_int&)
    - #TU::IIDCCamera::getDelay()
    - #TU::IIDCCamera::getDuration()
    
  - <b>GPIOピン</b>
    - #TU::IIDCCamera::setOutputVoltage()

  - <b>カメラ設定の保存</b>
    - #TU::IIDCCamera::saveConfig()
    - #TU::IIDCCamera::restoreConfig()
    - #TU::IIDCCamera::getMemoryChannelMax()

  #TU::IIDCCameraArray - IEEE1394デジタルカメラの配列を表すクラス
    - #TU::IIDCCameraArray::IIDCCameraArray()

  \file		IIDC++.h
  \brief	IEEE1394デバイスに関連するクラスの定義と実装
*/
#ifndef __TU_IIDCPP_H
#define __TU_IIDCPP_H

#if defined(HAVE_CONFIG_H)
#  include <config.h>
#endif
#include <cstddef>	// for size_t
#include <cstdint>	// for uintXX_t
#include <sys/types.h>	// for u_int
#include <memory>	// for std::unique_ptr<T>
#include <list>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <netinet/in.h>
#include "TU/Image++.h"

/*!
  \namespace	TU
  \brief	本ライブラリで定義されたクラスおよび関数を納める名前空間
*/
namespace TU
{
/************************************************************************
*  class IIDCNode							*
************************************************************************/
//! IIDCのノードを表すクラス
/*!
  一般には, より具体的な機能を持ったノード(ex. デジタルカメラ)を表す
  クラスの基底クラスとして用いられる. 
*/
class IIDCNode
{
  public:
    typedef uint32_t	quadlet_t;
    typedef uint16_t	nodeid_t;
    typedef uint64_t	nodeaddr_t;
    
  public:
			IIDCNode()					;
    virtual		~IIDCNode()					;
			IIDCNode(const IIDCNode&)		= delete;
    IIDCNode&		operator =(const IIDCNode&)		= delete;
    
    uint64_t		globalUniqueId()			const	;
    uint32_t		unitSpecId()				const	;
    nodeaddr_t		commandRegisterBase()			const	;

  //! このノードのID(bus上のアドレス)を返す
  /*!
    \return		このノードのID
  */
    virtual nodeid_t	nodeId()				const	= 0;
    
  //! ノード内の指定されたアドレスから4byteの値を読み出す
  /*!
    \param addr		個々のノード内の絶対アドレス
  */
    virtual quadlet_t	readQuadlet(nodeaddr_t addr)		const	= 0;

  //! ノード内の指定されたアドレスに4byteの値を書き込む
  /*!
    \param addr		個々のノード内の絶対アドレス
    \param quad		書き込む4byteの値
  */
    virtual void	writeQuadlet(nodeaddr_t addr, quadlet_t quad)	= 0;

  //! isochronous受信用のバッファを割り当てる
  /*!
    \param packet_size	受信するパケット1つあたりのサイズ(単位: byte)
    \param buf_size	バッファ1つあたりのサイズ(単位: byte)
    \param nb_buffers	割り当てるバッファ数
    \return		割り当てられたisochronous受信用のチャンネル
  */
    virtual u_char	mapListenBuffer(u_int packet_size,
					u_int buf_size,
					u_int nb_buffers)		= 0;

  //! ノードに割り当てたすべての受信用バッファを廃棄する
    virtual void	unmapListenBuffer()				= 0;
    
  //! isochronousデータが受信されるのを待つ
  /*!
    実際にデータが受信されるまで, 本関数は呼び出し側に制御を返さない. 
    \return		データの入ったバッファの先頭アドレス. 
  */
    virtual const void*	waitListenBuffer()				= 0;

  //! データ受信済みのバッファを再びキューイングして次の受信データに備える
    virtual void	requeueListenBuffer()				= 0;

  //! すべての受信用バッファの内容を空にする
    virtual void	flushListenBuffer()				= 0;

  //! バスのサイクルタイムとOSのローカルタイムを取得する
  /*!
    \param localtime	OSのローカルタイムが返される.
    \return		バスのサイクルタイム.
  */
    virtual uint32_t	getCycletime(uint64_t& localtime)	const	= 0;

  protected:
    bool		isUnique()				const	;
    void		addToList()				const	;
    
  private:
    uint32_t		readValueFromUnitDependentDirectory(uint8_t key)
								  const	;
    uint32_t		readValueFromUnitDirectory(uint8_t key)	  const	;
    uint32_t		readValueFromDirectory(uint8_t key,
					       uint32_t& offset)  const	;
    quadlet_t		readQuadletFromConfigROM(uint32_t offset) const	;

  private:
    static std::list<const IIDCNode*>	_nodes;
};

/************************************************************************
*  class IIDCCamera							*
************************************************************************/
//! IIDCデジタルカメラを表すクラス
/*!
  1394-based Digital Camera Specification ver. 1.30に準拠. 
*/
class IIDCCamera
{
  private:
    typedef IIDCNode::nodeid_t		nodeid_t;
    typedef IIDCNode::nodeaddr_t	nodeaddr_t;
    
  public:
    typedef IIDCNode::quadlet_t		quadlet_t;

  //! カメラのタイプ
    enum Type
    {
	Monocular		= 0x00a02d,	//!< 単眼カメラ
	Binocular		= 0x00b09d,	//!< 二眼カメラ
    };
	
  //! カメラがサポートしている基本機能を表すビットマップ
  /*! どのような基本機能がサポートされているかは, inquireBasicFunction() に
      よって知ることができる. */
    enum BasicFunction
    {
	Advanced_Feature_Inq	= (0x1u << 31),	//!< カメラベンダ依存の機能
	I1394b_mode_Capability	= (0x1u << 23),	//!< 1394bモード
	Cam_Power_Cntl_Inq	= (0x1u << 15),	//!< 電源on/offの制御
	One_Shot_Inq		= (0x1u << 12),	//!< 画像1枚だけの撮影
	Multi_Shot_Inq		= (0x1u << 11)	//!< 指定された枚数の撮影
    };

  //! isochronous転送の速度
    enum Speed
    {
	SPD_100M	= 0,			//!< 100Mbps
	SPD_200M	= 1,			//!< 200Mbps
	SPD_400M	= 2,			//!< 400Mbps
	SPD_800M	= 3,			//!< 800Mbps
	SPD_1_6G	= 4,			//!< 1.6Gbps
	SPD_3_2G	= 5			//!< 3.2Gbps
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
	Format_5_0	 = 0x2a0,	//!< Format_5_0: 
	MONO8_640x480x2	 = 0x2a4,	//!< Format_5_1: 640x480x2 Y(mono)
	Format_5_2	 = 0x2a8,	//!< Format_5_2: 
	Format_5_3	 = 0x2ac,	//!< Format_5_3: 
	Format_5_4	 = 0x2b0,	//!< Format_5_4: 
	Format_5_5	 = 0x2b4,	//!< Format_5_5: 
	Format_5_6	 = 0x2b8,	//!< Format_5_6: 
	Format_5_7	 = 0x2bc,	//!< Format_5_7: 
	Format_7_0	 = 0x2e0,	//!< Format_7_0: カメラ機種に依存
	Format_7_1	 = 0x2e4,	//!< Format_7_1: カメラ機種に依存
	Format_7_2	 = 0x2e8,	//!< Format_7_2: カメラ機種に依存
	Format_7_3	 = 0x2ec,	//!< Format_7_3: カメラ機種に依存
	Format_7_4	 = 0x2f0,	//!< Format_7_4: カメラ機種に依存
	Format_7_5	 = 0x2f4,	//!< Format_7_5: カメラ機種に依存
	Format_7_6	 = 0x2f8,	//!< Format_7_6: カメラ機種に依存
	Format_7_7	 = 0x2fc,	//!< Format_7_7: カメラ機種に依存
    };

  //! 画像の形式とその名称
    struct FormatName
    {
	const Format		format;		//!< 画像の形式
	const char* const	name;		//!< 名称
    };

  //! カメラのフレームレートを表すビットマップ
  /*! どのようなフレームレートがサポートされているかは, inquireFrameRate()
      によって知ることができる. */
    enum FrameRate
    {
	FrameRate_1_875	= (0x1u << 31),	//!< 1.875fps
	FrameRate_3_75	= (0x1u << 30),	//!< 3.75fps
	FrameRate_7_5	= (0x1u << 29),	//!< 7.5fps
	FrameRate_15	= (0x1u << 28),	//!< 15fps
	FrameRate_30	= (0x1u << 27),	//!< 30fps
	FrameRate_60	= (0x1u << 26),	//!< 60fps
	FrameRate_120	= (0x1u << 25),	//!< 120fps
	FrameRate_240	= (0x1u << 24),	//!< 240fps
	FrameRate_x	= (0x1u << 23)	//!< 特殊なフレームレート
    };
    
  //! フレームレートとその名称
    struct FrameRateName
    {
	const FrameRate		frameRate;	//!< フレームレート
	const char* const	name;		//!< 名称
    };

  //! 出力画像の画素の形式
    enum PixelFormat
    {
	MONO_8		= (0x1u << 31),	//!< Y(mono)	  8bit/pixel
	YUV_411		= (0x1u << 30),	//!< YUV(4:1:1)	 12bit/pixel
	YUV_422		= (0x1u << 29),	//!< YUV(4:2:2)	 16bit/pixel
	YUV_444		= (0x1u << 28),	//!< YUV(4:4:4)	 24bit/pixel
	RGB_24		= (0x1u << 27),	//!< RGB	 24bit/pixel
	MONO_16		= (0x1u << 26),	//!< Y(mono16)	 16bit/pixel
	RGB_48		= (0x1u << 25),	//!< RGB	 48bit/pixel
	SIGNED_MONO_16	= (0x1u << 24),	//!< SIGNED MONO 16bit/pixel
	SIGNED_RGB_48	= (0x1u << 23),	//!< SIGNED RGB	 48bit/pixel
	RAW_8		= (0x1u << 22),	//!< RAW	  8bit/pixel
	RAW_16		= (0x1u << 21)	//!< RAW	 16bit/pixel
    };

  //! 画素の形式とその名称
    struct PixelFormatName
    {
	const PixelFormat	pixelFormat;	//!< 画素の形式
	const char* const	name;		//!< 名称
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
	TRIGGER_DELAY	= 0x834,	//!< トリガ入力から露光開始までの遅れ
	FRAME_RATE	= 0x83c,	//!< フレームレートの手動設定
	ZOOM		= 0x880,	//!< ズーム調整
	PAN		= 0x884,	//!< パン(左右の首振り)調整
	TILT		= 0x888,	//!< チルト(上下の首振り)調整
	OPTICAL_FILTER	= 0x88c,
	CAPTURE_SIZE	= 0x8c0,
	CAPTURE_QUALITY	= 0x8c4
    };

  //! 各属性( Feature)についてカメラがサポートしている機能を表すビットマップ
  /*! どのような機能がサポートされているかは, inquireFeatureFunction() に
      よって知ることができる. */
    enum FeatureFunction
    {
	Presence	= (0x1u << 31),	//!< この属性そのものをサポート
	Abs_Control	= (0x1u << 30),	//!< この属性を値によって制御可能
	One_Push	= (0x1u << 28),	//!< one pushモードをサポート
	ReadOut		= (0x1u << 27),	//!< この属性の値の読み出しが可能
	OnOff		= (0x1u << 26),	//!< この属性のon/offが可能
	Auto		= (0x1u << 25),	//!< この属性の値の自動設定が可能
	Manual		= (0x1u << 24)	//!< この属性の値の手動設定が可能
    };
    
  //! 属性とその名称
    struct FeatureName
    {
	const Feature		feature;	//!< 属性
	const char* const	name;		//!< 名称
    };

  //! カメラの外部トリガーモード
  /*! どのトリガーモードがサポートされているかは, inquireTriggerMode() に
      よって知ることができる. */
    enum TriggerMode
    {
	Trigger_Mode0	= (0x1u << 15),  //!< トリガonから #SHUTTER で指定した時間だけ蓄積
	Trigger_Mode1	= (0x1u << 14),  //!< トリガonからトリガoffになるまで蓄積
	Trigger_Mode2	= (0x1u << 13),
	Trigger_Mode3	= (0x1u << 12),
	Trigger_Mode4	= (0x1u << 11),
	Trigger_Mode5	= (0x1u << 10),
	Trigger_Mode14	= (0x1u << 1),
	Trigger_Mode15	= 0x1u
    };

  //! トリガーモードとその名称
    struct TriggerModeName
    {
	const TriggerMode	mode;		//!< トリガーモード
	const char* const	name;		//!< 名称
    };
    
  //! カメラのストロボ出力
    enum Strobe
    {
	Strobe_0	= 0x100,		//!< ストロボ信号0
	Strobe_1	= 0x104,		//!< ストロボ信号1
	Strobe_2	= 0x108,		//!< ストロボ信号2
	Strobe_3	= 0x10c,		//!< ストロボ信号3
    };
	
  //! 各ストロボ信号( Strobe)についてカメラがサポートしている機能を表すビットマップ
  /*! どのような機能がサポートされているかは, inquireStrobeFunction() に
      よって知ることができる. */
    enum StrobeFunction
    {
	Strobe_Presence	= (0x1u << 31),	//!< このストロボ信号そのものをサポート
	Strobe_ReadOut	= (0x1u << 27),	//!< このストロボ信号の値の読み出しが可能
	Strobe_OnOff	= (0x1u << 26),	//!< このストロボ信号のon/offが可能
	Strobe_Polarity	= (0x1u << 25),	//!< このストロボ信号の極性設定が可能
    };
    
  //! 本カメラがサポートするFormat_7に関する情報(getFormat_7_Info() で得られる)
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
    
  //! Bayerパターン
    enum Bayer
    {
	RGGB = 0x52474742,	//!< 左上から右下に向かってR, G, G, B
	BGGR = 0x42474752,	//!< 左上から右下に向かってB, G, G, R
	GRBG = 0x47524247,	//!< 左上から右下に向かってG, R, B, G
	GBRG = 0x47425247,	//!< 左上から右下に向かってG, B, R, G
	YYYY = 0x59595959	//!< すべてY(monochrome)
    };

  private:
    struct Mono16
    {
	operator short()		const	{return ntohs(s);}
	
	short	s;
    };

  public:
			IIDCCamera()					;
			IIDCCamera(uint64_t uniqId,
				   Type type=Monocular)			;
			~IIDCCamera()					;
			IIDCCamera(const IIDCCamera&)		= delete;
    IIDCCamera&		operator =(const IIDCCamera&)		= delete;
			IIDCCamera(IIDCCamera&& camera)			;
    IIDCCamera&		operator =(IIDCCamera&& camera)			;

    uint64_t		globalUniqueId()			const	;
    IIDCCamera&		initialize(uint64_t uniqId=0,
				   Type type=Monocular)			;
    IIDCCamera&		terminate()					;
    IIDCCamera&		reset()						;

  // Basic function stuffs.
    uint32_t		inquireBasicFunction()			const	;
    IIDCCamera&		setPower(bool enable)				;
    Bayer		bayerTileMapping()			const	;
    bool		isLittleEndian()			const	;
    IIDCCamera&		setSpeed(Speed speed)				;
    Speed		getSpeed()				const	;
    
  // Format and frame rate stuffs.
    uint32_t		inquireFrameRate(Format format)		const	;
    IIDCCamera&		setFormatAndFrameRate(Format format,
					      FrameRate rate)		;
    Format		getFormat()				const	;
    FrameRate		getFrameRate()				const	;
    size_t		width()					const	;
    size_t		height()				const	;
    PixelFormat		pixelFormat()				const	;

  // Format_7 stuffs.
    Format_7_Info	getFormat_7_Info(Format format7)	const	;
    IIDCCamera&		setFormat_7_ROI(Format format7, 
					u_int u0, u_int v0,
					u_int width, u_int height)	;
    IIDCCamera&		setFormat_7_PixelFormat(Format format7, 
						PixelFormat pixelFormat);
    
  // Feature stuffs.
    uint32_t		inquireFeatureFunction(Feature feature)	const	;
    IIDCCamera&		onePush(Feature feature)			;
    IIDCCamera&		setActive(Feature feature, bool enable)		;
    IIDCCamera&		setAbsControl(Feature feature, bool enable)	;
    IIDCCamera&		setAuto(Feature feature, bool enable)		;
    IIDCCamera&		setValue(Feature feature, u_int value)		;
    IIDCCamera&		setValue(Feature feature, float value)		;
    bool		inOnePushOperation(Feature feature)	const	;
    bool		isActive(Feature feature)		const	;
    bool		isAbsControl(Feature feautre)		const	;
    bool		isAuto(Feature feautre)			const	;
    void		getMinMax(Feature feature,
				  u_int& min, u_int& max)	const	;
    void		getMinMax(Feature feature,
				  float& min, float& max)	const	;
    template <class T=u_int>
    T			getValue(Feature feature)		const	;
    
  // White balance stuffs.
    IIDCCamera&		setWhiteBalance(u_int ub, u_int vr)		;
    IIDCCamera&		setWhiteBalance(float ub, float vr)		;
    void		getWhiteBalance(u_int& ub, u_int& vr)	const	;
    void		getWhiteBalance(float& ub, float& vr)	const	;
    
  // Temperature stuffs.
    u_int		getAimedTemperature()			const	;

  // Trigger stuffs.
    IIDCCamera&		setTriggerMode(TriggerMode mode)		;
    TriggerMode		getTriggerMode()			const	;
    IIDCCamera&		setTriggerPolarity(bool highActive)		;
    bool		getTriggerPolarity()			const	;
    IIDCCamera&		setSoftwareTrigger()				;
    IIDCCamera&		resetSoftwareTrigger()				;

  // Strobe stuffs.
    uint32_t		inquireStrobeFunction(Strobe strobe)	const	;
    IIDCCamera&		setActive(Strobe strobe, bool enable)		;
    IIDCCamera&		setPolarity(Strobe strobe, bool highActive)	;
    IIDCCamera&		setDelay(Strobe strobe, u_int delay)		;
    IIDCCamera&		setDuration(Strobe strobe, u_int duration)	;
    bool		isStrobeActive(Strobe strobe)		const	;
    bool		getPolarity(Strobe strobe)		const	;
    void		getMinMax(Strobe strobe,
				  u_int& min, u_int& max)	const	;
    u_int		getDelay(Strobe strobe)			const	;
    u_int		getDuration(Strobe strobe)		const	;

  // Output voltage stuffs. (Blackfly only)
    IIDCCamera&		setOutputVoltage(bool enable)			;
    
  // Shotting stuffs.
    IIDCCamera&		continuousShot(bool enable)			;
    bool		inContinuousShot()			const	;
    IIDCCamera&		oneShot()					;
    IIDCCamera&		multiShot(u_short nframes)			;

  // Configuration saving/restoring stuffs.
    IIDCCamera&		saveConfig(u_int mem_ch)			;
    IIDCCamera&		restoreConfig(u_int mem_ch)			;
    u_int		getMemoryChannelMax()			const	;
    
  // Capture stuffs.
    IIDCCamera&		snap()						;
    template <class T>
    const IIDCCamera&	operator >>(Image<T>& image)		const	;
    template <class T>
    const IIDCCamera&	captureRGBImage(Image<T>& image)	const	;
    template <class T>
    const IIDCCamera&	captureDirectly(Image<T>& image)	const	;
    const IIDCCamera&	captureRaw(void* image)			const	;
    const IIDCCamera&	captureBayerRaw(void* image)		const	;
    IIDCCamera&		embedTimestamp(bool enable)			;
    uint64_t		getTimestamp()				const	;
    uint32_t		getCycletime(uint64_t& localtime)	const	;
    uint64_t		cycletimeToLocaltime(uint32_t cycletime) const	;
    
  // Utility functions.
    static Format		uintToFormat(u_int format)		;
    static FrameRate		uintToFrameRate(u_int frameRate)	;
    static Feature		uintToFeature(u_int feature)		;
    static TriggerMode		uintToTriggerMode(u_int triggerMode)	;
    static PixelFormat		uintToPixelFormat(u_int pixelFormat)	;
    static std::ostream&	printCycletime(std::ostream& out,
					       uint32_t cycletime)	;
    static std::ostream&	printLocaltime(std::ostream& out,
					       uint64_t localtime)	;
    
  private:
    uint32_t	getAbsValueOffset(Feature feature)		 const	;
    uint32_t	getStrobeOffset(Strobe strobe)			 const	;
    nodeaddr_t	getFormat_7_BaseAddr(Format format7)		 const	;
    u_int	setFormat_7_PacketSize(Format format7)			;
    uint32_t	inquireFrameRate_or_Format_7_Offset(Format format) const;
    bool	unlockAdvancedFeature(uint64_t featureId,
				      u_int timeout)		 const	;
    void	checkAvailability(Format format, FrameRate rate) const	;
    uint32_t	checkAvailability(Feature feature, uint32_t inq) const	;
    void	checkAvailability(Strobe strove, uint32_t inq)	 const	;
    void	checkAvailability(BasicFunction func)		 const	;
    quadlet_t	readQuadletFromRegister(uint32_t offset)	 const	;
    void	writeQuadletToRegister(uint32_t offset, quadlet_t quad)	;
    quadlet_t	readQuadletFromACRegister(uint32_t offset)	 const	;
    void	writeQuadletToACRegister(uint32_t offset,
					 quadlet_t quad)		;

  private:
    std::unique_ptr<IIDCNode>	_node;
    nodeaddr_t			_cmdRegBase;
    nodeaddr_t			_acRegBase;
    size_t			_w, _h;
    PixelFormat			_p;
    const void*			_img;
    size_t			_img_size;
    Bayer			_bayer;
    bool			_littleEndian;	//!<  MONO16 がlittle endian

  public:
    static constexpr FormatName
    formatNames[] =
    {
	{YUV444_160x120,	"160x120-YUV(4:4:4)"},
	{YUV422_320x240,	"320x240-YUV(4:2:2)"},
	{YUV411_640x480,	"640x480-YUV(4:1:1)"},
	{YUV422_640x480,	"640x480-YUV(4:2:2)"},
	{RGB24_640x480,		"640x480-RGB"},
	{MONO8_640x480,		"640x480-Y(mono)"},
	{MONO16_640x480,	"640x480-Y(mono16)"},
	{YUV422_800x600,	"800x600-YUV(4:2:2)"},
	{RGB24_800x600,		"800x600-RGB"},
	{MONO8_800x600,		"800x600-Y(mono)"},
	{YUV422_1024x768,	"1024x768-YUV(4:2:2)"},
	{RGB24_1024x768,	"1024x768-RGB"},
	{MONO8_1024x768,	"1024x768-Y(mono)"},
	{MONO16_800x600,	"800x600-Y(mono16)"},
	{MONO16_1024x768,	"1024x768-Y(mono16)"},
	{YUV422_1280x960,	"1280x960-YUV(4:2:2)"},
	{RGB24_1280x960,	"1280x960-RGB"},
	{MONO8_1280x960,	"1280x960-Y(mono)"},
	{YUV422_1600x1200,	"1600x1200-YUV(4:2:2)"},
	{RGB24_1600x1200,	"1600x1200-RGB"},
	{MONO8_1600x1200,	"1600x1200-Y(mono)"},
	{MONO16_1280x960,	"1280x960-Y(mono16)"},
	{MONO16_1600x1200,	"1600x1200-Y(mono16)"},
	{Format_5_0,		"Format_5_0"},
	{MONO8_640x480x2,	"640x480x2-Y(mono)"},
	{Format_5_2,		"Format_5_2"},
	{Format_5_3,		"Format_5_3"},
	{Format_5_4,		"Format_5_4"},
	{Format_5_5,		"Format_5_5"},
	{Format_5_6,		"Format_5_6"},
	{Format_5_7,		"Format_5_7"},
	{Format_7_0,		"Format_7_0"},
	{Format_7_1,		"Format_7_1"},
	{Format_7_2,		"Format_7_2"},
	{Format_7_3,		"Format_7_3"},
	{Format_7_4,		"Format_7_4"},
	{Format_7_5,		"Format_7_5"},
	{Format_7_6,		"Format_7_6"},
	{Format_7_7,		"Format_7_7"}
    };
    static constexpr size_t	NFORMATS = sizeof(formatNames)
					 / sizeof(formatNames[0]);
    
    static constexpr FrameRateName
    frameRateNames[] =
    {
	{FrameRate_1_875,	"1.875fps"},
	{FrameRate_3_75,	"3.75fps"},
	{FrameRate_7_5,		"7.5fps"},
	{FrameRate_15,		"15fps"},
	{FrameRate_30,		"30fps"},
	{FrameRate_60,		"60fps"},
	{FrameRate_120,		"120fps"},
	{FrameRate_240,		"240fps"},
	{FrameRate_x,		"custom_frame_rate"}
    };
    static constexpr size_t	NRATES = sizeof(frameRateNames)
				       / sizeof(frameRateNames[0]);
    
    static constexpr PixelFormatName
    pixelFormatNames[] =
    {
	{MONO_8,		"Y(mono)"},
	{YUV_411,		"YUV(4:1:1)"},
	{YUV_422,		"YUV(4:2:2)"},
	{YUV_444,		"YUV(4:4:4)"},
	{RGB_24,		"RGB"},
	{MONO_16,		"Y(mono16)"},
	{RGB_48,		"RGB(color48)"},
	{SIGNED_MONO_16,	"Y(signed mono16)"},
	{SIGNED_RGB_48,		"RGB(signed color48)"},
	{RAW_8,			"RAW(raw8)"},
	{RAW_16,		"RAW(raw16)"}
    };
    static constexpr size_t	NPIXELFORMATS = sizeof(pixelFormatNames)
					      / sizeof(pixelFormatNames[0]);
    
    static constexpr FeatureName
    featureNames[] = 
    {
	{BRIGHTNESS,		"BRIGHTNESS"},
	{AUTO_EXPOSURE,		"AUTO_EXPOSURE"},
	{SHARPNESS,		"SHARPNESS"},
	{WHITE_BALANCE,		"WHITE_BALANCE"},
	{HUE,			"HUE"},
	{SATURATION,		"SATURATION"},
	{GAMMA,			"GAMMA"},
	{SHUTTER,		"SHUTTER"},
	{GAIN,			"GAIN"},
	{IRIS,			"IRIS"},
	{FOCUS,			"FOCUS"},
	{TEMPERATURE,		"TEMPERATURE"},
	{TRIGGER_MODE,		"TRIGGER_MODE"},
	{TRIGGER_DELAY,		"TRIGGER_DELAY"},
	{FRAME_RATE,		"FRAME_RATE"},
	{ZOOM,			"ZOOM"},
	{PAN,			"PAN"},
	{TILT,			"TILT"}
    };
    static constexpr size_t	NFEATURES = sizeof(featureNames)
					  / sizeof(featureNames[0]);

    static constexpr TriggerModeName
    triggerModeNames[] =
    {
	{Trigger_Mode0,		"Trigger_Mode0" },
	{Trigger_Mode1,		"Trigger_Mode1" },
	{Trigger_Mode2,		"Trigger_Mode2" },
	{Trigger_Mode3,		"Trigger_Mode3" },
	{Trigger_Mode4,		"Trigger_Mode4" },
	{Trigger_Mode5,		"Trigger_Mode5" },
	{Trigger_Mode14,	"Trigger_Mode14"},
	{Trigger_Mode15,	"Trigger_Mode15"},
    };
    static constexpr size_t	NTRIGGERMODES = sizeof(triggerModeNames)
					      / sizeof(triggerModeNames[0]);
};

//! このカメラがサポートするBayerパターン(#Bayer)を返す
inline IIDCCamera::Bayer
IIDCCamera::bayerTileMapping() const
{
    return _bayer;
}

//! このカメラの#MONO_16 フォーマットがlittle endianであるかを調べる
inline bool
IIDCCamera::isLittleEndian() const
{
    return _littleEndian;
}

//! 現在設定されている画像フォーマット(#Format)の幅を返す
inline size_t
IIDCCamera::width() const
{
    return _w;
}

//! 現在設定されている画像フォーマット(#Format)の高さを返す
inline size_t
IIDCCamera::height() const
{
    return _h;
}

//! 現在設定されている画像フォーマット(#Format)の画素形式(#PixelFormat)を返す
inline IIDCCamera::PixelFormat
IIDCCamera::pixelFormat() const
{
    return _p;
}

//! カメラがサポートしている基本機能を返す
/*!
  \return	サポートされている機能を #BasicFunction 型の列挙値のorとして
		返す
 */
inline uint32_t
IIDCCamera::inquireBasicFunction() const
{
    return readQuadletFromRegister(0x400);
}

//! カメラから出力される最初の画像を保持する
/*!
  カメラからの画像出力は, continuousShot(), oneShot(), multiShot() の
  いずれかによって行われる. 実際に画像データが受信されるまで, 本関数は
  呼び出し側に制御を返さない. 
  \return	このIIDCカメラオブジェクト
 */
inline IIDCCamera&
IIDCCamera::snap()
{
    if (_img)
	_node->requeueListenBuffer();
    _img = _node->waitListenBuffer();
    return *this;
}

//! IIDCカメラから出力された画像を直接的に取り込む
/*!
  #operator >>() との違いは, 画像形式の変換を行わないことと, Image<T> 構造体
  の中のデータ領域へのポインタをIIDC入力バッファへのポインタに書き換える
  ことによって, 実際にはデータのコピーを行わないことである. 
  テンプレートパラメータTは, 格納先の画像の画素形式を表す. なお, 本関数を
  呼び出す前に snap() によってカメラからの画像を保持しておかなければならない. 
  \param image	画像データを格納する画像オブジェクト. 画像の幅と高さは, 
		現在カメラに設定されている画像サイズに合わせて自動的に
		設定される. 
  \return	このIIDCカメラオブジェクト
*/
template <class T> const IIDCCamera&
IIDCCamera::captureDirectly(Image<T>& image) const
{
    if (!_img)
	throw std::runtime_error("TU::IIDCCamera::captureDirectly: no images snapped!!");
    image.resize(static_cast<T*>(const_cast<void*>(_img)),
		 height(), width());

    return *this;
}

//! 画像に埋め込まれた撮影時刻を得る．
/*!
  予め embedTimestamp() によって画像への撮影時刻埋め込みを指示しなければならない．
  \return	サイクル時刻単位で表した画像の撮影時刻
*/
inline uint64_t
IIDCCamera::getTimestamp() const
{
    return (_img ?
	    cycletimeToLocaltime(ntohl(*static_cast<const uint32_t*>(_img))) :
	    0);
}

//! 同時にサイクル時刻とシステム時刻を得る．
/*!
  \param localtime	マイクロ秒単位でシステム時刻が返される
  \return		サイクル時刻
*/
inline uint32_t
IIDCCamera::getCycletime(uint64_t& localtime) const
{
    return _node->getCycletime(localtime);
}

inline void
IIDCCamera::checkAvailability(Format format, FrameRate rate) const
{
    auto	quad = inquireFrameRate(format);
    if (!(quad & rate))
    {
	using namespace	std;
    
	ostringstream	s;
	s << "IIDCCamera::checkAvailability: Incompatible combination of format["
	  << showbase << hex << format << "] and frame rate[" << rate << "]!!";
      	throw runtime_error(s.str());
    }
}

inline uint32_t
IIDCCamera::checkAvailability(Feature feature, uint32_t inq) const
{
    auto	quad = inquireFeatureFunction(feature);
    if ((quad & inq) != inq)
    {
	using namespace	std;
    
	ostringstream	s;
	s << "IIDCCamera::checkAvailability: This feature["
	  << showbase << hex << feature
	  << "] is not present or this field is unavailable (quad: "
	  << quad << ", inq: " << inq << ")!!";
      	throw runtime_error(s.str());
    }
    return quad;
}

inline void
IIDCCamera::checkAvailability(Strobe strobe, uint32_t inq) const
{
    auto	quad = inquireStrobeFunction(strobe);
    if ((quad & inq) != inq)
    {
	using namespace	std;
    
	ostringstream	s;
	s << "IIDCCamera::checkAvailability: This strobe["
	  << showbase << hex << strobe
	  << "] is not present or this field is unavailable (quad: "
	  << quad << ", inq: " << inq << ")!!";
      	throw runtime_error(s.str());
    }
}

inline void
IIDCCamera::checkAvailability(BasicFunction func) const
{
    auto	quad = inquireBasicFunction();
    if (!(quad & func))
    {
	using namespace	std;

	ostringstream	s;
	s << "IIDCCamera::checkAvailability: This fucntion is not present (quad: "
	  << showbase << hex << quad << ", func: " << func << ")!!";
      	throw runtime_error(s.str());
    }
}

inline IIDCCamera::quadlet_t
IIDCCamera::readQuadletFromRegister(uint32_t offset) const
{
    return _node->readQuadlet(_cmdRegBase + offset);
}

inline void
IIDCCamera::writeQuadletToRegister(uint32_t offset, quadlet_t quad)
{
    _node->writeQuadlet(_cmdRegBase + offset, quad);
}

inline IIDCCamera::quadlet_t
IIDCCamera::readQuadletFromACRegister(uint32_t offset) const
{
    return _node->readQuadlet(_acRegBase + offset);
}

inline void
IIDCCamera::writeQuadletToACRegister(uint32_t offset, quadlet_t quad)
{
    _node->writeQuadlet(_acRegBase + offset, quad);
}

/************************************************************************
*  global functions							*
************************************************************************/
std::ostream&	operator <<(std::ostream& out, const IIDCCamera& camera);
std::istream&	operator >>(std::istream& in, IIDCCamera& camera)	;

}
#endif	// !__TU_IIDCPP_H
