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
 *  $Id: raw1394_.h,v 1.1 2006-05-24 08:06:26 ueshiba Exp $
 */
#include "raw1394.h"
#include <IOKit/firewire/IOFireWireLibIsoch.h>
#include <mach/mach.h>

/************************************************************************
*  struct raw1394							*
************************************************************************/
struct raw1394
{
  private:
    class Interval
    {
      public:
	Interval()				;
	~Interval()				;

	UInt32		nPackets()	const	{return _nPackets;}
	DCLCommand**	commandList()	const	{return (DCLCommand**)_packet;}
	DCLTransferPacket*
			operator [](int i)const	{return _packet[i];}
	DCLTransferPacket*&
			operator [](int i)	{return _packet[i];}
	raw1394*	parent()	const	{return _parent;}

	void		resize(UInt32 n, raw1394* parent)	;

      private:
	Interval(const Interval&)				;
	Interval&	operator =(const Interval&)		;
	
      private:
	UInt32			_nPackets;
	DCLTransferPacket**	_packet;
	raw1394*		_parent;
      public:
	Interval*		prev;
	DCLLabel*		label;
	DCLJump*		jump;
	UInt32			nPacketsDropped;
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
			    UInt8 channel, SInt32 irqInterval)		;
    void	isoShutdown()						;
    IOReturn	isoRecvStart()						;
    IOReturn	isoStop()						;
    IOReturn	isoRecvFlush()						;
    SInt32	loopIterate()						;
    
  private:
    raw1394(const raw1394&)						;
    raw1394&	operator =(const raw1394&)				;
    
    static void	receiveHandler(DCLCommand* dcl)				;
    static IOReturn
		getSupportedHandler(IOFireWireLibIsochPortRef isochPort,
				    IOFWSpeed*		      speed,
				    UInt64*		      channel)	;
    static IOReturn
		stopHandler(IOFireWireLibIsochPortRef isochPort)	;
    
  private:
    IOCFPlugInInterface**		_cfPlugInInterface;
    IOFireWireLibDeviceRef		_fwDeviceInterface;
    CFStringRef				_runLoopMode;

    IOFireWireLibDCLCommandPoolRef	_commandPool;
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

    static UInt32			_id;
};

inline void
raw1394::setUserData(void* data)
{
    _userData = data;
}
    
inline void*
raw1394::getUserData() const
{
    return _userData;
}
    
inline IOReturn
raw1394::read(const FWAddress& addr, void* buf, UInt32 size) const
{
    return (*_fwDeviceInterface)
	->Read(_fwDeviceInterface,
	       (*_fwDeviceInterface)->GetDevice(_fwDeviceInterface),
	       &addr, buf, &size, kFWDontFailOnReset, 0);
}

inline IOReturn
raw1394::readQuadlet(const FWAddress& addr, UInt32* quad) const
{
    return (*_fwDeviceInterface)
	->ReadQuadlet(_fwDeviceInterface,
		      (*_fwDeviceInterface)->GetDevice(_fwDeviceInterface),
		      &addr, quad, kFWDontFailOnReset, 0);
}

inline IOReturn
raw1394::write(const FWAddress& addr, const void* buf, UInt32 size) const
{
    return (*_fwDeviceInterface)
	->Write(_fwDeviceInterface,
		(*_fwDeviceInterface)->GetDevice(_fwDeviceInterface),
		&addr, buf, &size, kFWDontFailOnReset, 0);
}
	       
inline IOReturn
raw1394::writeQuadlet(const FWAddress& addr, UInt32 quad) const
{
    return (*_fwDeviceInterface)
	->WriteQuadlet(_fwDeviceInterface,
		       (*_fwDeviceInterface)->GetDevice(_fwDeviceInterface),
		       &addr, quad, kFWDontFailOnReset, 0);
}

inline IOReturn
raw1394::isoRecvStart()
{
    return (*_isochChannel)->Start(_isochChannel);
}
    
inline IOReturn
raw1394::isoStop()
{
    return (*_isochChannel)->Stop(_isochChannel);
}
    
inline SInt32
raw1394::loopIterate()
{
    return CFRunLoopRunInMode(_runLoopMode, (CFTimeInterval)1, true);
}
