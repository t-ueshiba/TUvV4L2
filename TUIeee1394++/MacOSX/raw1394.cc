/*
 * libraw1394: Transplanation of Linux version to MacOS X
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
 *  $Id: raw1394.cc,v 1.14 2012-06-30 20:00:33 ueshiba Exp $
 */
#include "raw1394_.h"
#include <stdexcept>
#include <algorithm>
#include <sys/mman.h>
#include <sys/time.h>
#ifdef DEBUG
#  include <iostream>
#endif

#define USE_SY

/************************************************************************
*  class raw1394::Buffer						*
************************************************************************/
raw1394::Buffer::Buffer()
    :_nPackets(0), _dclList(nil), _packets(0), _timeStamps(0),
     _prev(0), _next(0), _parent(0), valid(false)
{
}

raw1394::Buffer::~Buffer()
{
    delete [] _timeStamps;
    delete [] _packets;
    if (_dclList != nil)
	CFRelease(_dclList);
}
    
void
raw1394::Buffer::resize(UInt32 n, const Buffer& prv, Buffer& nxt, raw1394* prnt)
{
    delete [] _timeStamps;
    delete [] _packets;
    if (_dclList != nil)
	CFRelease(_dclList);
    
    _nPackets	= n;
    _dclList	= CFSetCreateMutable(NULL, 0, NULL);
    _packets	= new NuDCLRef[_nPackets];
    _timeStamps = new UInt32[_nPackets];
    _prev	= &prv;
    _next	= &nxt;
    _parent	= prnt;
    valid	= false;
}

/************************************************************************
*  class raw1394							*
************************************************************************/
//! ::raw1394 構造体を生成する
/*!
  \param unit_spec_ID	この構造体が表すIEEE1394ノードの種類を示すID
  \param uniqId		個々の機器固有の64bit ID
*/
raw1394::raw1394(UInt32 unit_spec_ID, UInt64 uniqId)
    :_cfPlugInInterface(0), _fwDeviceInterface(0),
     _dclPool(0), _vm(0), _vmSize(0), _recvHandlerExt(0),
     _nBuffers(0), _buffers(0), _localIsochPort(0),
     _channel(0), _remoteIsochPort(0), _isochChannel(0),
     _mutex(), _lastProcessed(0), _lastReceived(0), _dropped(0),
     _userData(0)
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

  // Create mutex.
    MPCreateCriticalRegion(&_mutex);

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

//! ::raw1394 構造体を破壊する
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

	    MPDeleteCriticalRegion (_mutex);

	    (*_fwDeviceInterface)->Close(_fwDeviceInterface);
	    (*_fwDeviceInterface)->Release(_fwDeviceInterface);
	}
	IODestroyPlugInInterface(_cfPlugInInterface);
    }
}

//! この ::raw1394 構造体が表すノードのコマンドレジスタのベースアドレスを返す
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
  \return		初期設定が成功すればkIOReturnSuccess, そうでなければ
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
    cerr << "  nPackets = " << nPackets
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
						     buffer.dclList(),
						     4, 2, ranges);
	    if (j == 0)
		(*_dclPool)->SetDCLWaitControl(dcl, true);
	    (*_dclPool)->SetDCLFlags(dcl, kNuDCLDynamic);
	    (*_dclPool)->SetDCLTimeStampPtr(dcl, &buffer.timeStamp(j));
	    buffer[j] = dcl;
	    ranges[0].address += ranges[0].length;
	    ranges[1].address += ranges[1].length;

	    ++n;
	}

      // Bind the buffer and set the isochronous receive handler
      // to the last ReceivePacket command.
	(*_dclPool)->SetDCLUpdateList(dcl, buffer.dclList());
	(*_dclPool)->SetDCLRefcon(dcl, &buffer);
	(*_dclPool)->SetDCLCallback(dcl, dclCallback);
#ifdef DEBUG
	cerr << i << ": dcl = " << dcl << endl;
#endif
    }
  // Set the last buffer self-loop.
    (*_dclPool)->SetDCLBranch(_buffers[_nBuffers-1].last(),
			      _buffers[_nBuffers-1].first());

  // [Step 1.4] Create a local isochronous port and bind it the DCL program.
    ranges[0].address = _vm;
    ranges[0].length  = _vmSize;
    if (!(_localIsochPort =
	  (*_fwDeviceInterface)->CreateLocalIsochPortWithOptions(
	      _fwDeviceInterface, false, (*_dclPool)->GetProgram(_dclPool),
#ifdef USE_SY
	      kFWDCLSyBitsEvent, 0x1, 0x1,
#else
	      0, 0, 0,
#endif
	      nil, 0, ranges, 1,
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

  // [Step 4] Reset pointer to the ready buffer.
    MPEnterCriticalRegion(_mutex, kDurationForever);
    _lastProcessed = &_buffers[_nBuffers-1];
    _lastReceived  = &_buffers[_nBuffers-1];
    _dropped	   = 0;
    MPExitCriticalRegion(_mutex);

#ifdef DEBUG
    cerr << "*** END   [isoRecvInit] ***" << endl;
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

//! 現在のサイクル時刻とそれに対応する時刻を取得する
/*!
  \param cycle_timer	サイクル時刻が返される
  \param local_time	サイクル時刻に対応する時刻(micro sec)が返される
  \return		取得に成功すればkIOReturnSuccess, そうでなければ
			原因を示すエラーコード
*/
IOReturn
raw1394::readCycleTimer(UInt32* cycle_timer, UInt64* local_time) const
{
  // gettimeofday() は不正確なので UpTime() を使う．
    Nanoseconds	upTime = AbsoluteToNanoseconds(UpTime());
    *local_time = ((UInt64(upTime.hi) << 32) | UInt64(upTime.lo)) / 1000LL;

    UInt32	busTime;
    return (*_fwDeviceInterface)->GetBusCycleTime(_fwDeviceInterface,
						  &busTime, cycle_timer);
}

//! isochronous転送のループを1回実行する
/*!
  \return		データが壊れていなければユーザハンドラが返す値,
			壊れていれば-1
*/
SInt32
raw1394::loopIterate()
{
  // 受信済みだが処理されていないデータは, バッファ
  // (_lastProcessed, _lastReceived] に収められる. dclCallback() は, 
  // データを受信するたびにそのバッファを _lastReceived に記録する.
  // loopIterate() は, dclCallback() によって _lastReceived が更新されて
  // _lastProcessed < _lastReceived となるのを待ち, _lastProcessed を
  // 1つ先に進めてその内容が有効であればユーザ側に転送する.
#ifdef DEBUG
    using namespace	std;
    
    cerr << "*** BEGIN [loopIterate] ***" << endl;
#endif
  // dclCallback() によって新たなデータ読み込まれ, _lastReceived が更新される
  // まで待機
    while (_lastProcessed == _lastReceived)
	CFRunLoopRunInMode(kCFRunLoopDefaultMode, (CFTimeInterval)0, true);
    _lastProcessed = _lastProcessed->next();

    Buffer*	buffer = _lastProcessed;  // 未処理の最初のバッファ
#ifdef DEBUG
    cerr << "  buffer[" << buffer << "]: " << endl;
#endif

  // バッファデータの有効性を記憶し, 次の受信に備えて無効化しておく
    MPEnterCriticalRegion(_mutex, kDurationForever);
    const bool	valid = buffer->valid;
    buffer->valid = false;
    MPExitCriticalRegion(_mutex);
	
    SInt32	ret = -1;
    if (valid)	// データが有効(壊れていない)なら
    {
      // これまでに取りこぼしたパケット数を記憶し, 次の受信に備えて0にしておく
	MPEnterCriticalRegion(_mutex, kDurationForever);
	const UInt32	dropped = _dropped;
	_dropped = 0;
	MPExitCriticalRegion(_mutex);
#ifdef USE_SY
	UInt32		cycle = (buffer->timeStamp(0) & 0x1fff000) >> 12;
	IOVirtualRange	ranges[2];
	(*_dclPool)->GetDCLRanges(buffer->first(), 2, ranges);
	const UInt32	header = *((UInt32*)ranges[0].address);
	ret = _recvHandlerExt(this, (UInt8*)ranges[1].address,
		buffer->nPackets() * ranges[1].length,
		(header & kFWIsochChanNum) >> kFWIsochChanNumPhase,
		(header & kFWIsochTag)	   >> kFWIsochTagPhase,
		(header & kFWIsochSy)	   >> kFWIsochSyPhase,
		cycle, dropped);
#else
      // 先頭パケットのヘッダとデータアドレスを取り出し, ユーザハンドラを介して
      // パケットを一つずつ転送する
	for (UInt32 j = 0; j < buffer->nPackets(); ++j)
	{
	    UInt32		cycle = (buffer->timeStamp(j) & 0x1fff000)>> 12;
	    IOVirtualRange	ranges[2];
	    (*_dclPool)->GetDCLRanges((*buffer)[j], 2, ranges);
	    const UInt32	header = *((UInt32*)ranges[0].address);
	    ret = _recvHandlerExt(this, (UInt8*)ranges[1].address,
		    (header & kFWIsochDataLength) >> kFWIsochDataLengthPhase,
		    (header & kFWIsochChanNum)	  >> kFWIsochChanNumPhase,
		    (header & kFWIsochTag)	  >> kFWIsochTagPhase,
		    (header & kFWIsochSy)	  >> kFWIsochSyPhase,
		    cycle, dropped);
	}
#endif
#ifdef DEBUG
	cerr << "  " << buffer->nPackets() << " pkts sent, "
	     << dropped << " pkts dropped..." << endl;
#endif
    }
#ifdef DEBUG
    else
    {
	cerr << "CORRUPTED." << endl;
    }
#endif
    
  // 最後の処理済みバッファ _lastProcessed のDCLは, 以降の未処理バッファ
  // への上書きを防ぐために常に自己ループでなければならない. よって, この
  // バッファを自己ループにし, 1つ前のバッファをこのバッファに接続する.
    (*_dclPool)->SetDCLBranch(buffer->last(), buffer->first());
    (*_dclPool)->SetDCLBranch(buffer->prev()->last(), buffer->first());
    void*	dcls[] = {buffer->last(), buffer->prev()->last()};
    (*_localIsochPort)->Notify(_localIsochPort,
			       kFWNuDCLModifyJumpNotification, dcls, 2);

#ifdef DEBUG
    cerr << "*** END   [loopIterate] ***" << endl;
#endif
    return ret;
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

  // 最初のパケットのみヘッダにsyビットが立っていることを確認
    bool	valid = true;
#ifdef USE_SY
    for (UInt32 j = 0; j < buffer->nPackets(); ++j)
    {
	IOVirtualRange	ranges[2];
	(*me->_dclPool)->GetDCLRanges((*buffer)[j], 2, ranges);
	const UInt32	header = *((UInt32*)ranges[0].address),
			sy     = (header & kFWIsochSy) >> kFWIsochSyPhase;
	if ((j == 0) ^ (sy != 0))
	{
	    valid = false;
	    break;
	}
    }
#endif
  // このバッファが既に受信済みであるか今回のデータが壊れているなら, 今回の
  // データを取りこぼしたものとして扱う. (_dropped と Buffer::valid
  // は loopIterate() によって変えられる可能性があるので排他制御が必要)
    MPEnterCriticalRegion(me->_mutex, kDurationForever);
    if (buffer->valid || !valid)
	me->_dropped += buffer->nPackets(); // 今回とりこぼしたパケット数を加算
    buffer->valid = valid;		    // 今回のデータの有効性をセット
    MPExitCriticalRegion(me->_mutex);

  // このバッファを, データを受信した最新のバッファとする
    me->_lastReceived = buffer;

#ifdef DEBUG
    cerr << "  buffer[" << buffer << "]: ";
    if (buffer->valid)
	cerr << "valid." << endl;
    else
	cerr << "CURRUPTED." << endl;
    cerr << "*** END   [dclCallback] ***" << endl;
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
    std::cerr << "raw1394::allocatePortHandler: channel["
	      << channel << "] assigned." << std::endl;
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
    return handle->loopIterate();
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

extern "C" int
raw1394_read_cycle_timer(raw1394handle_t handle,
			 u_int32_t* cycle_timer, u_int64_t* local_time)
{
    return (handle->readCycleTimer(cycle_timer, local_time)
	    == kIOReturnSuccess ? 0 : -1);
}
