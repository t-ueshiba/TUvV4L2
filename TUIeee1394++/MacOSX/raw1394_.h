/*
 * libTUIeee1394++: C++ Library for Controlling IIDC 1394-based Digital Cameras
 * Copyright (C) 2003-2006 Toshio UESHIBA
 *   National Institute of Advanced Industrial Science and Technology (AIST)
 *
 * Written by Toshio UESHIBA <t.ueshiba@aist.go.jp>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  $Id: raw1394_.h,v 1.6 2007-01-19 02:27:48 ueshiba Exp $
 */
#include "raw1394.h"
#include <IOKit/firewire/IOFireWireLibIsoch.h>
#include <mach/mach.h>
/************************************************************************
*  struct raw1394							*
************************************************************************/
//! IEEE1394デバイスへのlibraw1394に互換なアクセスをMacOS X上で提供する構造体
struct raw1394
{
  private:
    class Interval
    {
      public:
	Interval()					;
	~Interval()					;

	UInt32		nPackets()		const	{return _nPackets;}
	const NuDCLRef& operator [](int i)	const	{return _packet[i];}
	NuDCLRef&	operator [](int i)		{return _packet[i];}
	const NuDCLRef&	first()			const	{return _packet[0];}
	const NuDCLRef&	last()			const	{return
							 _packet[_nPackets-1];}
	const Interval*	prev()			const	{return _prev;}
	raw1394*	parent()		const	{return _parent;}
	
	void		resize(UInt32 n, const Interval& prv, raw1394* prnt);

      private:
	Interval(const Interval&)			;
	Interval&	operator =(const Interval&)	;
	
      private:
	UInt32		_nPackets;
	NuDCLRef*	_packet;
	const Interval*	_prev;
	raw1394*	_parent;

      public:
	UInt32		nPacketsDropped;
    };
    
  public:
    raw1394(UInt32 unit_spec_ID, UInt64 uniqId)				;
    ~raw1394()								;

    void	setUserData(void* data)					;
    void*	getUserData()					const	;
    FWAddress	cmdRegBase()					const	;
    IOReturn	read(const FWAddress& addr,
		     void* buf, UInt32 size)			const	;
    IOReturn	readQuadlet(const FWAddress& addr, UInt32* quad)const	;
    IOReturn	write(const FWAddress& addr,
		      const void* buf, UInt32 size)		const	;
    IOReturn	writeQuadlet(const FWAddress& addr, UInt32 quad)const	;
    IOReturn	isoRecvInit(raw1394_iso_recv_handler_t handler,
			    UInt32 nPackets, UInt32 maxPacketSize,
			    UInt8& channel, SInt32 irqInterval)	;
    void	isoShutdown()						;
    IOReturn	isoRecvStart()						;
    IOReturn	isoStop()						;
    IOReturn	isoRecvFlush()						;
    SInt32	loopIterate()						;
    
  private:
    raw1394(const raw1394&)						;
    raw1394&	operator =(const raw1394&)				;
    
    static void	receiveHandler(void* refcon, NuDCLRef dcl)		;
    static IOReturn
		getSupportedHandler(IOFireWireLibIsochPortRef isochPort,
				    IOFWSpeed*		      speed,
				    UInt64*		      channel)	;
    static IOReturn
		allocatePortHandler(IOFireWireLibIsochPortRef isochPort,
				    IOFWSpeed		      speed,
				    UInt32		      channel)	;
    static IOReturn
		stopHandler(IOFireWireLibIsochPortRef isochPort)	;

  private:
    IOCFPlugInInterface**		_cfPlugInInterface;
    IOFireWireLibDeviceRef		_fwDeviceInterface;
    CFStringRef				_runLoopMode;

    IOFireWireLibNuDCLPoolRef		_dclPool;
    vm_address_t			_vm;
    vm_size_t				_vmSize;
    raw1394_iso_recv_handler_t		_recvHandlerExt;
    UInt32				_nIntervals;
    Interval*				_interval;
    IOFireWireLibLocalIsochPortRef	_localIsochPort;
    
    UInt32				_channel;
    IOFireWireLibRemoteIsochPortRef	_remoteIsochPort;
    IOFireWireLibIsochChannelRef	_isochChannel;

    void*				_userData;

    static UInt32			_nnodes;
};

//! ::raw1394構造体にユーザが指定したデータへのポインタを貼付ける
/*!
  \param data	貼付けたいデータへのポインタ
*/
inline void
raw1394::setUserData(void* data)
{
    _userData = data;
}
    
//! この::raw1394構造体に貼付けたデータへのポインタを得る
/*!
  \return	貼付けたデータへのポインタ
*/
inline void*
raw1394::getUserData() const
{
    return _userData;
}

//! 指定したFireWireアドレスから任意バイト数のデータをasynchronous転送で読み込む
/*!
  \param addr	読み込み元のFireWireアドレス
  \param buf	読み込み先のバッファアドレス
  \param size	読み込みデータのバイト数
  \return	読み込みが成功すればkIOReturnSuccess，そうでなければ
		エラーの原因を示すコード
*/
inline IOReturn
raw1394::read(const FWAddress& addr, void* buf, UInt32 size) const
{
    return (*_fwDeviceInterface)
	->Read(_fwDeviceInterface,
	       (*_fwDeviceInterface)->GetDevice(_fwDeviceInterface),
	       &addr, buf, &size, kFWDontFailOnReset, 0);
}

//! 指定したFireWireアドレスから4バイトのデータをasynchronous転送で読み込む
/*!
  \param addr	読み込み元のFireWireアドレス
  \param quad	読み込み先のアドレス
  \return	読み込みが成功すればkIOReturnSuccess，そうでなければ
		エラーの原因を示すコード
*/
inline IOReturn
raw1394::readQuadlet(const FWAddress& addr, UInt32* quad) const
{
    return (*_fwDeviceInterface)
	->ReadQuadlet(_fwDeviceInterface,
		      (*_fwDeviceInterface)->GetDevice(_fwDeviceInterface),
		      &addr, quad, kFWDontFailOnReset, 0);
}

//! 指定したFireWireアドレスに任意バイト数のデータをasynchronous転送で書き込む
/*!
  \param addr	書き込み先のFireWireアドレス
  \param buf	書き込み元のバッファアドレス
  \param size	書き込みデータのバイト数
  \return	書き込みが成功すればkIOReturnSuccess，そうでなければ
		エラーの原因を示すコード
*/
inline IOReturn
raw1394::write(const FWAddress& addr, const void* buf, UInt32 size) const
{
    return (*_fwDeviceInterface)
	->Write(_fwDeviceInterface,
		(*_fwDeviceInterface)->GetDevice(_fwDeviceInterface),
		&addr, buf, &size, kFWDontFailOnReset, 0);
}
	       
//! 指定したFireWireアドレスに4バイトのデータをasynchronous転送で書き込む
/*!
  \param addr	書き込み先のFireWireアドレス
  \param quad	書き込むデータ
  \return	書き込みが成功すればkIOReturnSuccess，そうでなければ
		エラーの原因を示すコード
*/
inline IOReturn
raw1394::writeQuadlet(const FWAddress& addr, UInt32 quad) const
{
    return (*_fwDeviceInterface)
	->WriteQuadlet(_fwDeviceInterface,
		       (*_fwDeviceInterface)->GetDevice(_fwDeviceInterface),
		       &addr, quad, kFWDontFailOnReset, 0);
}

//! isochronous受信を開始する
/*!
  \return	開始が成功すればkIOReturnSuccess，そうでなければ
		エラーの原因を示すコード
*/
inline IOReturn
raw1394::isoRecvStart()
{
    return (*_isochChannel)->Start(_isochChannel);
}
    
//! isochronous受信を停止する
/*!
  \return	停止が成功すればkIOReturnSuccess，そうでなければ
		エラーの原因を示すコード
*/
inline IOReturn
raw1394::isoStop()
{
    return (*_isochChannel)->Stop(_isochChannel);
}

//! isochronous転送のループを1回実行する
/*!
  \return	ループの実行が終了した原因を示すコード
*/
inline SInt32
raw1394::loopIterate()
{
    return CFRunLoopRunInMode(_runLoopMode, (CFTimeInterval)0, true);
}

