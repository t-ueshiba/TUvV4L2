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
 *  $Id: raw1394.cc,v 1.10 2009-10-16 02:47:25 ueshiba Exp $
 */
#include "raw1394_.h"
#include <stdexcept>
#include <algorithm>
#include <sys/mman.h>
#ifdef DEBUG
#  include <iostream>
#endif
/************************************************************************
*  class raw1394::Buffer						*
************************************************************************/
raw1394::Buffer::Buffer()
    :_nPackets(0), _packets(0), _prev(0), _next(0), _parent(0)
{
}

raw1394::Buffer::~Buffer()
{
    delete [] _packets;
}
    
void
raw1394::Buffer::resize(UInt32 n, const Buffer& prv,
			const Buffer& nxt, raw1394* prnt)
{
    delete [] _packets;
    _nPackets = n;
    _packets  = new NuDCLRef[_nPackets];
    _prev     = &prv;
    _next     = &nxt;
    _parent   = prnt;
}

/************************************************************************
*  class raw1394							*
************************************************************************/
//! ::raw1394構造体を生成する
/*!
  \param unit_spec_ID	この構造体が表すIEEE1394ノードの種類を示すID
  \param uniqId		個々の機器固有の64bit ID
*/
raw1394::raw1394(UInt32 unit_spec_ID, UInt64 uniqId)
    :_cfPlugInInterface(0), _fwDeviceInterface(0),
     _dclPool(0), _vm(0), _vmSize(0), _recvHandlerExt(0),
     _nBuffers(0), _buffers(0), _localIsochPort(0),
     _channel(0), _remoteIsochPort(0), _isochChannel(0),
#ifndef SINGLE_THREAD
     _mutex(), _ready(0),
#endif
      _dropped(0), _userData(0)
{
    using namespace	std;
    
  // Find a specified device node.
    CFMutableDictionaryRef
		dictionary = IOServiceMatching("IOFireWireUnit");
    CFNumberRef	cfValue = CFNumberCreate(kCFAllocatorDefault,
					 kCFNumberSInt32Type, &unit_spec_ID);
    CFDictionaryAddValue(dictionary, CFSTR("Unit_Spec_Id"), cfValue);
    CFRelease(cfValue);
    if (uniqId != 0)
    {
	cfValue = CFNumberCreate(kCFAllocatorDefault,
				 kCFNumberLongLongType, &uniqId);
	CFDictionaryAddValue(dictionary, CFSTR("GUID"), cfValue);
	CFRelease(cfValue);
    }
    io_iterator_t	iterator;
    if (IOServiceGetMatchingServices(kIOMasterPortDefault,
				     dictionary, &iterator) != kIOReturnSuccess)
	throw runtime_error("raw1394::raw1394: failed to get a matched service(=device)!!");

  // Find a FireWire device interface.
    for (io_object_t service; service = IOIteratorNext(iterator); )
    {
	SInt32	theScore;
	if ((IOCreatePlugInInterfaceForService(service, kIOFireWireLibTypeID,
		kIOCFPlugInInterfaceID, &_cfPlugInInterface, &theScore)
	     == kIOReturnSuccess) &&
	    ((*_cfPlugInInterface)->QueryInterface(_cfPlugInInterface,
		CFUUIDGetUUIDBytes(kIOFireWireDeviceInterfaceID_v8),
		(void**)&_fwDeviceInterface)
	     == S_OK) &&
	    ((*_fwDeviceInterface)->Open(_fwDeviceInterface)
	     == kIOReturnSuccess))
	    break;

	if (_cfPlugInInterface)
	{
	    if (_fwDeviceInterface)
	    {
		(*_fwDeviceInterface)->Release(_fwDeviceInterface);
		_fwDeviceInterface = 0;
	    }
	    IODestroyPlugInInterface(_cfPlugInInterface);
	    _cfPlugInInterface = 0;
	}
    }
    IOObjectRelease(iterator);
    if (!_fwDeviceInterface)
	throw runtime_error("raw1394::raw1394: no specified service(=device) found!!");
#ifndef SINGLE_THREAD
  // Create mutex.
    MPCreateCriticalRegion (&_mutex);
#endif    
  // Add a callback dispatcher to RunLoop with a specific mode.
    if ((*_fwDeviceInterface)->AddCallbackDispatcherToRunLoopForMode(
	    _fwDeviceInterface, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode)
	    != kIOReturnSuccess)
	    throw runtime_error("raw1394::raw1394: failed to add a callback dispatcher!!");
    if ((*_fwDeviceInterface)->AddIsochCallbackDispatcherToRunLoopForMode(
	    _fwDeviceInterface, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode)
	    != kIOReturnSuccess)
	    throw runtime_error("raw1394::raw1394: failed to add an isochronous callback dispatcher!!");
    if (!(*_fwDeviceInterface)->TurnOnNotification(_fwDeviceInterface))
	    throw runtime_error("raw1394::raw1394: failed to turn on notification!!");
}

//! ::raw1394構造体を破壊する
raw1394::~raw1394()
{
    if (_cfPlugInInterface)
    {
	if (_fwDeviceInterface)
	{
	    isoShutdown();
	    (*_fwDeviceInterface)
		->RemoveIsochCallbackDispatcherFromRunLoop(_fwDeviceInterface);
	    (*_fwDeviceInterface)
		->RemoveCallbackDispatcherFromRunLoop(_fwDeviceInterface);
#ifndef SINGLE_THREAD
	    MPDeleteCriticalRegion (_mutex);
#endif
	    (*_fwDeviceInterface)->Close(_fwDeviceInterface);
	    (*_fwDeviceInterface)->Release(_fwDeviceInterface);
	}
	IODestroyPlugInInterface(_cfPlugInInterface);
    }
}

//! この::raw1394構造体が表すノードのコマンドレジスタのベースアドレスを返す
/*!
  \return		コマンドレジスタのベースアドレス
*/
FWAddress
raw1394::cmdRegBase() const
{
    using namespace	std;
    
    IOFireWireLibConfigDirectoryRef	unitDirectory = 0;
    if (!(unitDirectory = (*_fwDeviceInterface)->GetConfigDirectory(
	      _fwDeviceInterface,
	      CFUUIDGetUUIDBytes(kIOFireWireConfigDirectoryInterfaceID))))
	throw runtime_error("TU:raw1394::cmdRegBase: failed to get Unit Directory!!");
    IOFireWireLibConfigDirectoryRef	unitDependentDirectory = 0;
    if ((*unitDirectory)->GetKeyValue_ConfigDirectory(
	    unitDirectory, kConfigUnitDependentInfoKey, &unitDependentDirectory,
	    CFUUIDGetUUIDBytes(kIOFireWireConfigDirectoryInterfaceID), nil)
	!= kIOReturnSuccess)
	throw runtime_error("raw1394::cmdRegBase: failed to get Unit Dependent Directory!!");
    
    FWAddress	addr;
    CFStringRef	text;
    if ((*unitDependentDirectory)->GetKeyOffset_FWAddress(
	    unitDependentDirectory, 0x00, &addr, &text) != kIOReturnSuccess)
	throw runtime_error("raw1394::cmdRegBase: failed to get base address of command registers!!");
    (*unitDependentDirectory)->Release(unitDependentDirectory);
    (*unitDirectory)->Release(unitDirectory);

    return addr;
}

//! isochronous受信の初期設定を行う
/*!
  \param handler	カーネルが受信した各パケットに対して実行されるハンドラ
  \param nPackets	受信するパケット数
  \param maxPacketSize	受信パケットの最大バイト数
  \param channel	ishochronous受信のチャンネル
  \param irqInterval	カーネルが割り込みをかける間隔を指定するパケット数
  \return		初期設定が成功すればkIOReturnSuccess，そうでなければ
			エラーの原因を示すコード
*/
IOReturn
raw1394::isoRecvInit(raw1394_iso_recv_handler_t handler,
		     UInt32 nPackets, UInt32 maxPacketSize,
		     UInt8& channel, SInt32 irqInterval)
{
#ifdef DEBUG
    using namespace	std;

    cerr << "*** BEGIN [isoRecvInit] ***" << endl;
#endif
    isoShutdown();		// Release preveously allocated resouces.
    
    if (irqInterval <= 0)
	irqInterval = 16;	// Default vlaue of IrqInterval.
    if (irqInterval > nPackets)
	irqInterval = nPackets;
#ifdef DEBUG
    cerr << "  nPackets = " << dec << nPackets
	 << ", irqInterval = " << irqInterval
	 << ", maxPacketSize = " << maxPacketSize
	 << endl;
#endif
  // [Step 1] Set up a local isochronous port.
  // [Step 1.1] Create DCL command pool.
    if (!(_dclPool = (*_fwDeviceInterface)->CreateNuDCLPool(
	      _fwDeviceInterface, nPackets,
	      CFUUIDGetUUIDBytes(kIOFireWireNuDCLPoolInterfaceID))))
	return kIOReturnError;

  // [Step 1.2] Allocate virtual memory. Don't use "new" or "malloc"!!
    _vmSize = nPackets * (4 + maxPacketSize);	// 4 bytes for isoch. header
    if ((_vm = (IOVirtualAddress)mmap(NULL, _vmSize, PROT_READ | PROT_WRITE,
				      MAP_ANON | MAP_SHARED, - 1, 0))
	== (IOVirtualAddress)MAP_FAILED)
	return kIOReturnError;

  // [Step 1.3] Write a DCL program.
    _recvHandlerExt = handler;
    _nBuffers	    = (nPackets - 1) / irqInterval + 1;
    _buffers	    = new Buffer[_nBuffers];
    IOVirtualRange	ranges[2] = {{_vm,		  4},
				     {_vm + nPackets * 4, maxPacketSize}};
    for (UInt32 n = 0, i = 0; i < _nBuffers; ++i)
    {
	Buffer&		buffer = _buffers[i];

	buffer.resize((i < _nBuffers-1 ? irqInterval : nPackets-n),
		      (i > 0 ? _buffers[i-1] : _buffers[_nBuffers-1]),
		      (i < _nBuffers-1 ? _buffers[i+1] : _buffers[0]),
		      this);

      // Allocate ReceivePacketStart commands.
	NuDCLRef	dcl;
	for (UInt32 j = 0; j < buffer.nPackets(); ++j)
	{
	    dcl = (*_dclPool)->AllocateReceivePacket(_dclPool,
						     NULL, 4, 2, ranges);
	    if (j == 0)
		(*_dclPool)->SetDCLWaitControl(dcl, true);
	    (*_dclPool)->SetDCLFlags(dcl, kNuDCLDynamic);
	    buffer[j] = dcl;
	    ranges[0].address += ranges[0].length;
	    ranges[1].address += ranges[1].length;

	    ++n;
	}

      // Bind the buffer and set the isochronous receive handler
      // to the last ReceivePacket command.
	(*_dclPool)->SetDCLRefcon(dcl, &buffer);
	(*_dclPool)->SetDCLCallback(dcl, dclCallback);
#ifdef DEBUG
	cerr << i << ": dcl = " << hex << dcl << endl;
#endif
    }
    (*_dclPool)->SetDCLBranch(_buffers[_nBuffers-1].last(),
			      _buffers[0].first());

  // [Step 1.4] Create a local isochronous port and bind it the DCL program.
    ranges[0].address = _vm;
    ranges[0].length  = _vmSize;
    if (!(_localIsochPort =
	  (*_fwDeviceInterface)->CreateLocalIsochPortWithOptions(
	      _fwDeviceInterface, false, (*_dclPool)->GetProgram(_dclPool),
	      kFWDCLSyBitsEvent, 0x1, 0x1, nil, 0, ranges, 1,
	      kFWIsochPortUseSeparateKernelThread,
	      CFUUIDGetUUIDBytes(kIOFireWireLocalIsochPortInterfaceID))))
	return kIOReturnError;

  // [Step 2] Set up a remote isochronous port.
    if (!(_remoteIsochPort = (*_fwDeviceInterface)->CreateRemoteIsochPort(
	      _fwDeviceInterface, true,
	      CFUUIDGetUUIDBytes(kIOFireWireRemoteIsochPortInterfaceID))))
	return kIOReturnError;
    (*_remoteIsochPort)->SetRefCon((IOFireWireLibIsochPortRef)_remoteIsochPort,
				   this);
    (*_remoteIsochPort)->SetGetSupportedHandler(_remoteIsochPort,
						getSupportedHandler);
    (*_remoteIsochPort)->SetAllocatePortHandler(_remoteIsochPort,
						allocatePortHandler);
    (*_remoteIsochPort)->SetStopHandler(_remoteIsochPort, stopHandler);

  // [Step 3] Set up an isochronous channel.
    if (!(_isochChannel = (*_fwDeviceInterface)->CreateIsochChannel(
	      _fwDeviceInterface, true, maxPacketSize, kFWSpeed400MBit,
	      CFUUIDGetUUIDBytes(kIOFireWireIsochChannelInterfaceID))))
	return kIOReturnError;
    IOReturn	err;
    if ((err = (*_isochChannel)->SetTalker(_isochChannel,
	   (IOFireWireLibIsochPortRef)_remoteIsochPort)) != kIOReturnSuccess)
	return err;
    if ((err = (*_isochChannel)->AddListener(_isochChannel,
	   (IOFireWireLibIsochPortRef)_localIsochPort)) != kIOReturnSuccess)
	return err;
    if ((err = (*_isochChannel)->AllocateChannel(_isochChannel))
	!= kIOReturnSuccess)
	return err;
    channel = _channel;		// Return the assinged channel number.
#ifndef SINGLE_THREAD
  // [Step 4] Reset pointer to the ready buffer.
    MPEnterCriticalRegion(_mutex, kDurationForever);
    _ready   = 0;
    _dropped = 0;
    MPExitCriticalRegion(_mutex);
#else
    _dropped = 0;
#endif
#ifdef DEBUG
    cerr << "*** END [isoRecvInit] ***" << endl;
#endif
    return kIOReturnSuccess;
}

//! isochronous転送を停止して必要なリソースを解放する
void
raw1394::isoShutdown()
{
    if (_isochChannel)
    {
	(*_isochChannel)->ReleaseChannel(_isochChannel);
	(*_isochChannel)->Release(_isochChannel);
	_isochChannel = 0;
    }
    if (_remoteIsochPort)
    {
	(*_remoteIsochPort)->Release(_remoteIsochPort);
	_remoteIsochPort = 0;
    }
    if (_localIsochPort)
    {
	(*_localIsochPort)->Release(_localIsochPort);
	_localIsochPort = 0;
    }
    if (_buffers)
    {
	delete [] _buffers;
	_buffers = 0;
    }
    if (_vm)
    {
	munmap((void*)_vm, _vmSize);
	_vm = 0;
    }
    if (_dclPool)
    {
	(*_dclPool)->Release(_dclPool);
	_dclPool = 0;
    }
}

//! カーネル空間に保持されているisochronous受信データをユーザ空間に転送する
/*!
  \return		転送が成功すればkIOReturnSuccess, そうでなければ
			原因を示すエラーコード
*/
IOReturn
raw1394::isoRecvFlush()
{
    return kIOReturnSuccess;
}

//! isochronous転送のループを1回実行する
/*!
  \return		ループの実行が成功すればkIOReturnSuccess,
			そうでなければ原因を示すエラーコード
*/
IOReturn
raw1394::loopIterate()
{
#ifndef SINGLE_THREAD
#  ifdef DEBUG
    using namespace	std;
    
    cerr << "*** BEGIN [loopIterate] ***" << endl;
#  endif
  // いずれかのバッファにデータ読み込まれるまで待機
    while (!_ready)
	CFRunLoopRunInMode(kCFRunLoopDefaultMode, (CFTimeInterval)0, true);
#  ifdef DEBUG
    cerr << "  ready buffer = " << hex << _ready << endl;
#  endif
  // 最新のデータが読み込まれたバッファ:_readyと取りこぼしたパケット数:
  // _droppedはdclCallback()によって変えられる可能性があるので排他制御
    MPEnterCriticalRegion(_mutex, kDurationForever);
    const Buffer* const	ready	= _ready;
    const UInt32	dropped	= _dropped;
    _ready = 0;
    _dropped = 0;
    MPExitCriticalRegion(_mutex);

  // バッファをリングから取り出す
    (*_dclPool)->SetDCLBranch(ready->prev()->last(), ready->next()->first());
    (*_dclPool)->SetDCLBranch(ready->last(), ready->first());
    void*	dcls[] = {ready->prev()->last(), ready->last()};
    (*_localIsochPort)->Notify(_localIsochPort,
			       kFWNuDCLModifyJumpNotification, dcls, 2);

  // 先頭パケットのヘッダとデータアドレスを取り出す
    IOVirtualRange	ranges[2];
    (*_dclPool)->GetDCLRanges(ready->first(), 2, ranges);
    UInt32		header = *((UInt32*)ranges[0].address);

  // ユーザハンドラを介してデータを転送する
    raw1394_iso_disposition	ret =
	_recvHandlerExt(this,
			(UInt8*)ranges[1].address,
			ready->nPackets() * ranges[1].length,
			(header & kFWIsochChanNum) >> kFWIsochChanNumPhase,
			(header & kFWIsochTag)     >> kFWIsochTagPhase,
			(header & kFWIsochSy)      >> kFWIsochSyPhase,
			(header & kFWIsochTCode)   >> kFWIsochTCodePhase,
			dropped);

  // バッファをリングに挿入する
    (*_dclPool)->SetDCLBranch(ready->last(), ready->next()->first());
    (*_dclPool)->SetDCLBranch(ready->prev()->last(), ready->first());
    dcls[0] = ready->last();
    dcls[1] = ready->prev()->last();
    (*_localIsochPort)->Notify(_localIsochPort,
			       kFWNuDCLModifyJumpNotification, dcls, 2);
#  ifdef DEBUG
    cerr << "*** END [loopIterate] ***" << endl;
#  endif
    return ((ret == RAW1394_ISO_OK) || (ret == RAW1394_ISO_DEFER) ?
	    kIOReturnSuccess : kIOReturnError);
#else
    return (CFRunLoopRunInMode(kCFRunLoopDefaultMode, (CFTimeInterval)0, true)
	    == kCFRunLoopRunHandledSource ? kIOReturnSuccess : kIOReturnError);
#endif
}

/************************************************************************
*  Isochronous handlers implemented as static member functions		*
************************************************************************/
void
raw1394::dclCallback(void* refcon, NuDCLRef dcl)
{
#ifdef DEBUG
    using namespace	std;
    
    cerr << "*** BEGIN [dclCallback] ***" << endl;
#endif
    Buffer*	buffer = (Buffer*)refcon;
    raw1394*	me     = buffer->parent();
    
  // Notify() は一度に30個のDCLしか処理できない
    for (UInt32 j = 0; j < buffer->nPackets(); j += 30)
	(*me->_localIsochPort)->Notify(me->_localIsochPort,
				       kFWNuDCLUpdateNotification,
				       (void**)&(*buffer)[j],
				       std::min(buffer->nPackets() - j,
						UInt32(30)));

  // 最初のパケットのみヘッダにsyビットが立っていることを確認
    UInt32	dropped = 0;
    for (UInt32 j = 0; j < buffer->nPackets(); ++j)
    {
	IOVirtualRange	ranges[2];
	(*me->_dclPool)->GetDCLRanges((*buffer)[j], 2, ranges);
	UInt32	header  = *((UInt32*)ranges[0].address),
		sy	= (header & kFWIsochSy) >> kFWIsochSyPhase;
	if ((j == 0) ^ (sy != 0))
	{
	    dropped = buffer->nPackets();
	    break;
	}
    }
#ifndef SINGLE_THREAD
  // _readyと_droppedはloopIterate()によって変えられる可能性があるので排他制御
    MPEnterCriticalRegion(me->_mutex, kDurationForever);
    if (me->_ready != 0)	// 前回のデータがユーザハンドラに未渡しだったら
    {				// これを取りこぼしたデータとしてカウントする
	me->_dropped += me->_ready->nPackets();
	me->_ready    = 0;
    }
    if (dropped == 0)			// 今回のデータが壊れていなければ
	me->_ready = buffer;		// これを最新のバッファとする
    else				// 壊れていれば
	me->_dropped += dropped;	// 取りこぼしたデータとしてカウントする
    MPExitCriticalRegion(me->_mutex);
#else
    if (dropped == 0)
    {
      // バッファをリングから取り出す
	(*me->_dclPool)->SetDCLBranch(buffer->prev()->last(),
				      buffer->next()->first());
	(*me->_dclPool)->SetDCLBranch(buffer->last(), buffer->first());
	void*	dcls[] = {buffer->prev()->last(), buffer->last()};
	(*me->_localIsochPort)->Notify(me->_localIsochPort,
				       kFWNuDCLModifyJumpNotification, dcls, 2);

      // 先頭パケットのヘッダを取り出し，ユーザハンドラを介してデータを転送する
	IOVirtualRange	ranges[2];
	(*me->_dclPool)->GetDCLRanges(buffer->first(), 2, ranges);
	UInt32		header = *((UInt32*)ranges[0].address);
	me->_recvHandlerExt(me, (UInt8*)ranges[1].address,
			    buffer->nPackets() * ranges[1].length,
			    (header & kFWIsochChanNum) >> kFWIsochChanNumPhase,
			    (header & kFWIsochTag)     >> kFWIsochTagPhase,
			    (header & kFWIsochSy)      >> kFWIsochSyPhase,
			    (header & kFWIsochTCode)   >> kFWIsochTCodePhase,
			    me->_dropped);
	me->_dropped = 0;

      // バッファをリングに挿入する
	(*me->_dclPool)->SetDCLBranch(buffer->last(), buffer->next()->first());
	(*me->_dclPool)->SetDCLBranch(buffer->prev()->last(), buffer->first());
	dcls[0] = buffer->last();
	dcls[1] = buffer->prev()->last();
	(*me->_localIsochPort)->Notify(me->_localIsochPort,
				       kFWNuDCLModifyJumpNotification, dcls, 2);
    }
    else
	me->_dropped += dropped;
#endif    
#ifdef DEBUG
    cerr << "*** END [dclCallback] ***" << endl;
#endif
}

IOReturn
raw1394::getSupportedHandler(IOFireWireLibIsochPortRef	isochPort,
			     IOFWSpeed*			speed,
			     UInt64*			channel)
{
    *speed = kFWSpeedMaximum;
    *channel = 0xffff000000000000ULL;
    return kIOReturnSuccess;
}
    
IOReturn
raw1394::allocatePortHandler(IOFireWireLibIsochPortRef	isochPort,
			     IOFWSpeed			speed,
			     UInt32			channel)
{
    raw1394*	me = (raw1394*)(*isochPort)->GetRefCon(isochPort);
    me->_channel = channel;
#ifdef DEBUG
    std::cerr << "raw1394::allocatePortHandler: channel[" << dec << channel
	      << "] assigned." << std::endl;
#endif
    return kIOReturnSuccess;
}

IOReturn
raw1394::stopHandler(IOFireWireLibIsochPortRef isochPort)
{
    raw1394*	me = (raw1394*)(*isochPort)->GetRefCon(isochPort);
    while (CFRunLoopRunInMode(kCFRunLoopDefaultMode, (CFTimeInterval)0, true)
	   != kCFRunLoopRunTimedOut)
	;
    return kIOReturnSuccess;
}

/************************************************************************
*  wrapper C functions							*
************************************************************************/
extern "C" raw1394handle_t
raw1394_new_handle(u_int32_t unit_spec_ID, u_int64_t uniqId)
{
    return new raw1394(unit_spec_ID, uniqId);
}

extern "C" void
raw1394_destroy_handle(raw1394handle_t handle)
{
    delete handle;
}

extern "C" void
raw1394_set_userdata(raw1394handle_t handle, void* data)
{
    handle->setUserData(data);
}

extern "C" void*
raw1394_get_userdata(raw1394handle_t handle)
{
    return handle->getUserData();
}

extern "C" nodeaddr_t
raw1394_command_register_base(raw1394handle_t handle)
{
    FWAddress	fwaddr = handle->cmdRegBase();
    return (UInt64(fwaddr.addressHi) << 32) | UInt64(fwaddr.addressLo);
}

extern "C" int
raw1394_read(raw1394handle_t handle, nodeid_t node,
	     nodeaddr_t addr, size_t length, quadlet_t* quad)
{
    return (handle->read(FWAddress(addr >> 32, addr), quad, length)
	    == kIOReturnSuccess ? 0 : -1);
}

extern "C" int
raw1394_write(raw1394handle_t handle, nodeid_t node,
	      nodeaddr_t addr, size_t length, quadlet_t* quad)
{
    return (handle->write(FWAddress(addr >> 32, addr), quad, length)
	    == kIOReturnSuccess ? 0 : -1);
}

extern "C" int
raw1394_loop_iterate(raw1394handle_t handle)
{
    return (handle->loopIterate() == kIOReturnSuccess ? 0 : -1);
}

extern "C" int
raw1394_iso_recv_init(raw1394handle_t			handle,
		      raw1394_iso_recv_handler_t	handler,
		      unsigned int			buf_packets,
		      unsigned int			max_packet_size,
		      unsigned char&			channel,
		      raw1394_iso_dma_recv_mode		mode,
		      int				irq_interval)
{
    return (handle->isoRecvInit(handler, buf_packets, max_packet_size, channel,
				irq_interval) == kIOReturnSuccess ? 0 : -1);
}

extern "C" void
raw1394_iso_shutdown(raw1394handle_t handle)
{
    handle->isoShutdown();
}

extern "C" int
raw1394_iso_recv_start(raw1394handle_t handle, int start_on_cycle, 
		       int tag_mask, int sync)
{
    return (handle->isoRecvStart() == kIOReturnSuccess ? 0 : -1);
}

extern "C" void
raw1394_iso_stop(raw1394handle_t handle)
{
    handle->isoStop();
}

extern "C" int
raw1394_iso_recv_flush(raw1394handle_t handle)
{
    return (handle->isoRecvFlush() == kIOReturnSuccess ? 0 : -1);
}
