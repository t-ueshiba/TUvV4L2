/*
 *  $Id: raw1394.h,v 1.15 2012-08-29 19:33:23 ueshiba Exp $
 */
/*!
  \mainpage	libraw1394 - Linux上のlibraw1394に互換な機能をMacOS X上で提供するエミュレーションライブラリ
  \anchor	libraw1394

  \section copyright 著作権
  Copyright (C) 2003-2006 Toshio UESHIBA
  National Institute of Advanced Industrial Science and Technology (AIST)
 
  Written by Toshio UESHIBA <t.ueshiba@aist.go.jp>
 
  This library is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 2.1 of the
  License, or (at your option) any later version.
 
  This library is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.
 
  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA

  \section functions 機能
  次のインタフェースを提供している. 
  - raw1394_new_handle()
  - raw1394_destroy_handle()
  - raw1394_set_userdata()
  - raw1394_get_userdata()
  - raw1394_command_register_base()
  - raw1394_read()
  - raw1394_write()
  - raw1394_loop_iterate()
  - raw1394_iso_recv_init()
  - raw1394_iso_shutdown()
  - raw1394_iso_recv_start()
  - raw1394_iso_stop()
  - raw1394_iso_recv_flush()
  - raw1394_read_cycle_timer()
  
  \file		raw1394.h
  \brief	グローバルな名前空間に置かれたC向けの型定義とインタフェース
*/
#ifndef _LIBRAW1394_RAW1394_H
#define _LIBRAW1394_RAW1394_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif
/************************************************************************
*  type definitions							*
************************************************************************/
/*! ::raw1394 構造体へのハンドル */
typedef struct raw1394*		raw1394handle_t;

typedef u_int16_t		nodeid_t;	//!< ノードのID
typedef u_int64_t		nodeaddr_t;	//!< ノードのFireWireアドレス

/*! asynchronous転送データの単位(4 bytes) */
typedef u_int32_t		quadlet_t;

/*! ishochronous送/受信ハンドラの終了状態を表し, さらにこれを呼んだ
    raw1394_loop_iterate() にどのような処理をさせるかを指定するコード
*/
enum raw1394_iso_disposition
{
  /*! ハンドラは正常に終了, 次のパケットを処理. */
    RAW1394_ISO_OK		= 0,
  /*! ハンドラは正常に終了, #raw1394_loop_iterate() から直ちにリターン */
    RAW1394_ISO_DEFER		= 1,
  /*! ハンドラにエラーが発生, #raw1394_loop_iterate() から直ちにリターン */
    RAW1394_ISO_ERROR		= 2,
  /*! #raw1394_loop_iterate() から直ちにリターンしてisochronous受信を停止 */
    RAW1394_ISO_STOP		= 3,
  /*! (isochronous送信時のみ使用) */
    RAW1394_ISO_STOP_NOSYNC	= 4,
  /*! (isochronous送信時のみ使用) */
    RAW1394_ISO_AGAIN		= 5
};

/*! isochronous受信のモード */    
enum raw1394_iso_dma_recv_mode
{
    RAW1394_DMA_DEFAULT		  = -1,	/*!< デフォルトモード */
    RAW1394_DMA_BUFFERFILL	  =  1,	/*!< BUFFER_FILLモード */
    RAW1394_DMA_PACKET_PER_BUFFER =  2	/*!< PACKET_PER_BUFFERモード */
};

/*! カーネルが受信した各パケットに対して実行されるハンドラ
  \param handle		::raw1394 構造体へのハンドル
  \param data		パケット中のデータへのポインタ(ヘッダ含まず)
  \param len		パケット中のデータのバイト数(ヘッダ含まず)
  \param channel	isochronous受信のチャンネル
  \param tag		パケットのヘッダ中のtagビット
  \param sy		パケットのヘッダ中のsyビット
  \param cycle		パケットのcycle番号
  \param dropped	とりこぼしたパケット数
  \return
*/
typedef raw1394_iso_disposition	(*raw1394_iso_recv_handler_t)(
    raw1394handle_t	handle,
    unsigned char*	data,
    unsigned int	len,
    unsigned char	channel,
    unsigned char	tag,
    unsigned char	sy,
    unsigned int	cycle,
    unsigned int	dropped);

/************************************************************************
*  wrapper C functions							*
************************************************************************/
//! IEEE1394インターフェースである ::raw1394 構造体を生成する
/*!
  \param unit_spec_ID	この構造体が表すIEEE1394ノードの種類を示すID
  \param uniqId		個々の機器固有の64bit ID
  \return		生成された ::raw1394 構造体へのハンドル
*/
raw1394handle_t
	raw1394_new_handle(u_int32_t unit_spec_ID, u_int64_t uniqId);

//! IEEE1394インターフェースである ::raw1394 構造体を破壊する
/*!
  \param handle		::raw1394 構造体へのハンドル
*/
void	raw1394_destroy_handle(raw1394handle_t handle);

//! ::raw1394 構造体にユーザが指定したデータへのポインタを貼付ける
/*!
  \param handle		::raw1394 構造体へのハンドル
  \param data		貼付けたいデータへのポインタ
*/
void	raw1394_set_userdata(raw1394handle_t handle, void* data);

//! ::raw1394 構造体に貼付けたデータへのポインタを得る
/*!
  \param handle		::raw1394 構造体へのハンドル
  \return		貼付けたデータへのポインタ
*/
void*	raw1394_get_userdata(raw1394handle_t handle);

//! ::raw1394 構造体が表すノードのコマンドレジスタのベースアドレスを返す
/*!
  \param handle		::raw1394 構造体へのハンドル
  \return		コマンドレジスタのベースアドレス
*/
nodeaddr_t
	raw1394_command_register_base(raw1394handle_t handle);

//! 指定したFireWireアドレスから任意バイト数のデータをasynchronous転送で読み込む
/*!
  \param handle		::raw1394 構造体へのハンドル
  \param node		(ダミー引数)
  \param addr		読み込み元のFireWireアドレス
  \param length		読み込みデータのバイト数
  \param quad		読み込み先のバッファアドレス
  \return		読み込みが成功すれば0, そうでなければ-1
*/
int	raw1394_read(raw1394handle_t handle, nodeid_t node,
		     nodeaddr_t addr, size_t length, quadlet_t* quad);

//! 指定したFireWireアドレスに任意バイト数のデータをasynchronous転送で書き込む
/*!
  \param handle		::raw1394 構造体へのハンドル
  \param node		(ダミー引数)
  \param addr		書き込み先のFireWireアドレス
  \param length		書き込みデータのバイト数
  \param quad		書き込み元のバッファアドレス
  \return		書き込みが成功すれば0, そうでなければ-1
*/
int	raw1394_write(raw1394handle_t handle, nodeid_t node,
		      nodeaddr_t addr, size_t length, quadlet_t* quad);

//! isochronous転送のループを1回実行する
/*!
  \param handle		::raw1394 構造体へのハンドル
  \return		ループの実行が正常に終了すれば0, そうでなければ-1
*/
int	raw1394_loop_iterate(raw1394handle_t handle);

//! isochronous受信の初期設定を行う
/*!
  \param handle			::raw1394 構造体へのハンドル
  \param handler		カーネルが受信した各パケットに対して実行される
				ハンドラ
  \param buf_packets		受信するパケット数
  \param max_packet_size	受信パケットの最大バイト数
  \param channel		ishochronous受信のチャンネル
  \param mode			(ダミー引数)
  \param irq_interval		カーネルが割り込みをかける間隔を指定する
				パケット数
  \return			初期設定が成功すれば0, そうでなければ-1
*/
int	raw1394_iso_recv_init(raw1394handle_t		     handle,
			      raw1394_iso_recv_handler_t     handler,
			      unsigned int		     buf_packets,
			      unsigned int		     max_packet_size,
			      unsigned char&		     channel,
			      raw1394_iso_dma_recv_mode	     mode,
			      int			     irq_interval);

//! isochronous転送を停止して必要なリソースを解放する
/*!
  \param handle		::raw1394 構造体へのハンドル
*/
void	raw1394_iso_shutdown(raw1394handle_t handle);

//! isochronous受信を開始する
/*!
  \param handle		::raw1394 構造体へのハンドル
  \param start_on_cycle	(ダミー引数)
  \param tag_mask	(ダミー引数)
  \param sync		(ダミー引数)
  \return		開始が成功すれば0, そうでなければ-1
*/
int	raw1394_iso_recv_start(raw1394handle_t handle,
			       int start_on_cycle, int tag_mask,
			       int sync);
//! isochronous受信を停止する
/*!
  \param handle		::raw1394 構造体へのハンドル
*/
void	raw1394_iso_stop(raw1394handle_t handle);

//! カーネル空間に保持されているisochronous受信データをユーザ空間に転送する
/*!
  \param handle		::raw1394 構造体へのハンドル
  \return		転送が成功すれば0, そうでなければ-1
*/
int	raw1394_iso_recv_flush(raw1394handle_t handle);

//! 現在のサイクル時刻とそれに対応する時刻を取得する
/*!
  \param handle		::raw1394 構造体へのハンドル
  \param cycle_timer	サイクル時刻が返される
  \param local_time	サイクル時刻に対応する時刻(micro sec)が返される
  \return		取得に成功すれば0, そうでなければ-1
*/
int	raw1394_read_cycle_timer(raw1394handle_t handle,
				 u_int32_t* cycle_timer, u_int64_t* local_time);

#ifdef __cplusplus
}
#endif

#endif /* _LIBRAW1394_RAW1394_H */
