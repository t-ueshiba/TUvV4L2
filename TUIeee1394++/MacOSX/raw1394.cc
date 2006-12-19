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
 *  $Id: raw1394.cc,v 1.4 2006-12-19 07:05:20 ueshiba Exp $
 */
#include "raw1394_.h"
#include <stdexcept>
#ifdef DEBUG
#  include <iostream>
#endif
/************************************************************************
*  class raw1394::Interval						*
************************************************************************/
raw1394::Interval::Interval()
    :_nPackets(0), _packet(0), _parent(0),
     prev(0), label(0), jump(0), nPacketsDropped(0)
{
}

raw1394::Interval::~Interval()
{
    delete [] _packet;
}
    
void
raw1394::Interval::resize(UInt32 n, raw1394* parent)
{
    delete [] _packet;
    _nPackets	    = n;
    _packet	    = new DCLTransferPacket*[_nPackets];
    _parent	    = parent;
    prev	    = 0;
    label	    = 0;
    jump	    = 0;
    nPacketsDropped = 0;
}
    
/************************************************************************
*  class raw1394							*
************************************************************************/
static SInt32	DefaultIrqInterval = 16;

raw1394::raw1394(UInt32 unit_spec_ID, UInt64 uniqId)
    :_cfPlugInInterface(0), _fwDeviceInterface(0), _runLoopMode(),
     _commandPool(0), _vm(0), _vmSize(0), _recvHandlerExt(0),
     _nIntervals(0), _interval(0), _localIsochPort(0),
     _channel(0), _remoteIsochPort(0), _isochChannel(0),
     _userData(0)
#ifdef USE_THREAD
     ,_mutex(), _cond(), _thread(), _state(Idle)
#endif
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

#ifdef USE_THREAD
    pthread_mutex_init(&_mutex, NULL);
    pthread_cond_init(&_cond, NULL);

    pthread_create(&_thread, NULL, threadProc, this);
#else
  // Add a callback dispatcher to RunLoop with a specific mode.
    char	mode[] = "raw1394.x";
    mode[strlen(mode)-1] = '0' + _nnodes++;
    _runLoopMode = CFStringCreateWithCString(kCFAllocatorDefault, mode,
					     CFStringGetSystemEncoding());
    if ((*_fwDeviceInterface)->AddIsochCallbackDispatcherToRunLoopForMode(
	    _fwDeviceInterface, CFRunLoopGetCurrent(), _runLoopMode)
	    != kIOReturnSuccess)
	    throw runtime_error("raw1394::raw1394: failed to add an isochronous callback dispatcher!!");
#endif
}

raw1394::~raw1394()
{
    if (_cfPlugInInterface)
    {
	if (_fwDeviceInterface)
	{
	    isoShutdown();
#ifdef USE_THREAD
	    pthread_mutex_lock(&_mutex);
	    _state = Exit;
	    pthread_cond_signal(&_cond);    // Send Exit signal to the child.
	    pthread_mutex_unlock(&_mutex);

	    pthread_join(_thread, NULL);    // Wait the child terminating.
	    pthread_cond_destroy(&_cond);
	    pthread_mutex_destroy(&_mutex);
#else
	    (*_fwDeviceInterface)
		->RemoveIsochCallbackDispatcherFromRunLoop(_fwDeviceInterface);
	    CFRelease(_runLoopMode);
	    --_nnodes;
#endif
	    (*_fwDeviceInterface)->Close(_fwDeviceInterface);
	    (*_fwDeviceInterface)->Release(_fwDeviceInterface);
	}
	IODestroyPlugInInterface(_cfPlugInInterface);
    }
}

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

IOReturn
raw1394::isoRecvInit(raw1394_iso_recv_handler_t handler,
		     UInt32 nPackets, UInt32 maxPacketSize,
		     UInt8& channel, SInt32 irqInterval)
{
    isoShutdown();		// Release preveously allocated resouces.
    
  // [Step 1] Set up a local isochronous port.
  // [Step 1.1] Create DCL command pool.
    maxPacketSize += 4;			// Add 4bytes for isochronous header.
    if (irqInterval <= 0)
	irqInterval = DefaultIrqInterval;
    if (irqInterval > nPackets)
	irqInterval = nPackets;
    const UInt32 nIntervals		= (nPackets - 1) / irqInterval + 1,
		 nPacketsOfLastInterval	= (nPackets - 1) % irqInterval + 1;
    const UInt32 poolSize = nIntervals*(sizeof(DCLLabel) +
					sizeof(DCLUpdateDCLList) +
					sizeof(DCLCallProc) +
					sizeof(DCLJump)) +
			      nPackets* sizeof(DCLTransferPacket);
    if (!(_commandPool = (*_fwDeviceInterface)->CreateDCLCommandPool(
	      _fwDeviceInterface, poolSize,
	      CFUUIDGetUUIDBytes(kIOFireWireDCLCommandPoolInterfaceID))))
	return kIOReturnError;

  // [Step 1.2] Allocate virtual memory. Don't use "new" or "malloc"!!
    UInt32	pageSize = 0;
    while (maxPacketSize > pageSize)
	pageSize += getpagesize();
    const UInt32 nPacketsPerPage = pageSize / maxPacketSize,
		 frac		 = pageSize % maxPacketSize,
		 nPages		 = nPackets / nPacketsPerPage + 1;
    _vmSize = nPages * pageSize;
    IOReturn	err;
    if ((err = vm_allocate(mach_task_self(), &_vm, _vmSize, TRUE))
	!= KERN_SUCCESS)
	return err;

  // [Step 1.3] Write a DCL program.
    _recvHandlerExt = handler;
    _nIntervals	    = nIntervals;
    _interval	    = new Interval[_nIntervals];
    DCLCommand*	dcl = nil;
    UInt8*	packet = (UInt8*)_vm;
    for (int n = 0, i = 0; i < _nIntervals; ++i)
    {
	Interval&	interval = _interval[i];

	interval.resize((i < _nIntervals - 1 ?
			 irqInterval : nPacketsOfLastInterval), this);

      // Allocate a Label.
	dcl = (*_commandPool)->AllocateLabelDCL(_commandPool, dcl);
	interval.label = (DCLLabel*)dcl;

      // Allocate ReceivePacketStart commands.
	for (int j = 0; j < interval.nPackets(); ++j)
	{
	    dcl = (*_commandPool)->AllocateReceivePacketStartDCL(
		_commandPool, dcl, packet, maxPacketSize);
	    interval[j] = (DCLTransferPacket*)dcl;
	    packet += maxPacketSize;
	    if (++n == nPacketsPerPage)
	    {
		n = 0;
		packet += frac;
	    }
	}

      // Allocate an UpdateDCLList command.
	dcl = (*_commandPool)->AllocateUpdateDCLListDCL(_commandPool, dcl,
							interval.commandList(),
							interval.nPackets());

      // Allocate a CallProc command with the isochronous receive handler.
	dcl = (*_commandPool)->AllocateCallProcDCL(_commandPool, dcl,
						   receiveHandler,
						   (UInt32)&interval);
#ifdef DEBUG
	std::cerr << i << ": dcl = " << std::hex << dcl << std::endl;
#endif
      // Allocate a Jump command.
	dcl = (*_commandPool)->AllocateJumpDCL(_commandPool, dcl,
					       interval.label);
	interval.jump = (DCLJump*)dcl;
    }

  // [Step 1.4] Create a local isochronous port.
#ifdef USE_THREAD
    if (!(_localIsochPort =
	  (*_fwDeviceInterface)->CreateLocalIsochPort(
	      _fwDeviceInterface, false, (DCLCommand*)_interval[0].label,
	      kFWDCLSyBitsEvent, 0x1, 0x1, nil, 0, nil, 0,
	      CFUUIDGetUUIDBytes(kIOFireWireLocalIsochPortInterfaceID))))
#else
    if (!(_localIsochPort =
	  (*_fwDeviceInterface)->CreateLocalIsochPortWithOptions(
	      _fwDeviceInterface, false, (DCLCommand*)_interval[0].label,
	      kFWDCLSyBitsEvent, 0x1, 0x1, nil, 0, nil, 0,
	      kFWIsochPortUseSeparateKernelThread,
	      CFUUIDGetUUIDBytes(kIOFireWireLocalIsochPortInterfaceID))))
#endif
	return kIOReturnError;

  // [Step 1.5] Modify jump labels.
    for (int i = 0; i < _nIntervals - 1; ++i)
    {
	(*_localIsochPort)->ModifyJumpDCL(_localIsochPort, _interval[i].jump,
					  _interval[i+1].label);
	_interval[i+1].prev = &_interval[i];
    }
    _interval[0].prev = &_interval[_nIntervals - 1];

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

  // [Step 4] Set up an isochronous channel.
    if (!(_isochChannel = (*_fwDeviceInterface)->CreateIsochChannel(
	      _fwDeviceInterface, true, maxPacketSize, kFWSpeed400MBit,
	      CFUUIDGetUUIDBytes(kIOFireWireIsochChannelInterfaceID))))
	return kIOReturnError;
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
    
    return kIOReturnSuccess;
}

inline void
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
    if (_interval)
    {
	delete [] _interval;
	_interval = 0;
    }
    if (_vm)
    {
	vm_deallocate(mach_task_self(), _vm, _vmSize);
	_vm = 0;
    }
    if (_commandPool)
    {
	(*_commandPool)->Release(_commandPool);
	_commandPool = 0;
    }
}

IOReturn
raw1394::isoRecvFlush()
{
    return kIOReturnSuccess;
}

/************************************************************************
*  Isochronous handlers implemented as static member functions		*
************************************************************************/
void
raw1394::receiveHandler(DCLCommand* dcl)
{
#ifdef DEBUG
    static int	n = 0;
    std::cerr << "BEGIN [" << std::dec << n++ << "] receiveHandler: dcl = "
	      << std::hex << dcl << std::endl;
#endif
    Interval*	interval =  (Interval*)((DCLCallProc*)dcl)->procData;
    if (interval->jump->pJumpDCLLabel == interval->label)
    {
	interval->nPacketsDropped += interval->nPackets();
	return;
    }
    
    raw1394*	me = interval->parent();
    for (int j = 0; j < interval->nPackets(); ++j)
    {
	UInt8*	p	= (UInt8*)(*interval)[j]->buffer;
	UInt32	header  = *((UInt32*)p),
		len	= (header & kFWIsochDataLength)
						     >> kFWIsochDataLengthPhase,
	  	tag	= (header & kFWIsochTag)     >> kFWIsochTagPhase,
		channel = (header & kFWIsochChanNum) >> kFWIsochChanNumPhase,
		sy	= (header & kFWIsochSy)      >> kFWIsochSyPhase,
	  	cycle   = (header & kFWIsochTCode)   >> kFWIsochTCodePhase;
	me->_recvHandlerExt(me, p + 4, len, channel, tag, sy, cycle,
			    interval->nPacketsDropped);
	interval->nPacketsDropped = 0;
    }

  // This interval with which all packets has been processed becomes
  // a new dead-end.
    (*(me->_localIsochPort))->ModifyJumpDCL(me->_localIsochPort,
					    interval->jump, interval->label);
    (*(me->_localIsochPort))->ModifyJumpDCL(me->_localIsochPort,
					    interval->prev->jump,
					    interval->label);
#ifdef DEBUG
    std::cerr << "END   [" << std::dec << n-1 << "] receiveHandler:"
	      << std::endl;
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
    std::cerr << "raw1394::allocatePortHandler: channel[" << channel
	      << "] assigned." << std::endl;
#endif
    return kIOReturnSuccess;
}
    
IOReturn
raw1394::stopHandler(IOFireWireLibIsochPortRef isochPort)
{
#ifndef USE_THREAD
    raw1394*	me = (raw1394*)(*isochPort)->GetRefCon(isochPort);
    while (me->loopIterate() == kCFRunLoopRunHandledSource);
#endif
    return kIOReturnSuccess;
}

/************************************************************************
*  static member variables for class raw1394				*
************************************************************************/
UInt32	raw1394::_nnodes = 0;

#ifdef USE_THREAD
/************************************************************************
*  thread relevant stuffs for class raw1394				*
************************************************************************/
void
raw1394::wait() const
{
    pthread_mutex_lock(&_mutex);
    while (_state != Idle)
	pthread_cond_wait(&_cond, &_mutex);
    pthread_mutex_unlock(&_mutex);
}

void*
raw1394::threadProc(void* r1394)
{
    raw1394*	me = (raw1394*)r1394;

    pthread_mutex_lock(&me->_mutex);
    (*me->_fwDeviceInterface)
	->AddIsochCallbackDispatcherToRunLoop(me->_fwDeviceInterface,
					      CFRunLoopGetCurrent());
    for (;;)
    {
	pthread_cond_wait(&me->_cond, &me->_mutex);
	if (me->_state == Ready)
	{
	    CFRunLoopRunInMode(kCFRunLoopDefaultMode, (CFTimeInterval)0, true);
	    me->_state = Idle;
	    pthread_cond_signal(&me->_cond);  // Send Idle signal to the parent.
	}
	else if (me->_state == Exit)
	    break;
    }
    (*me->_fwDeviceInterface)->RemoveIsochCallbackDispatcherFromRunLoop(
	me->_fwDeviceInterface);
    pthread_mutex_unlock(&me->_mutex);

    return 0;
}
#endif

/************************************************************************
*  wrapper C functions							*
************************************************************************/
extern "C" raw1394handle_t
raw1394_new_handle(unsigned int unit_spec_ID, unsigned long long uniqId)
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

#ifdef USE_THREAD
extern "C" void
raw1394_raise(raw1394handle_t handle)
{
    handle->raise();
}

extern "C" void
raw1394_wait(raw1394handle_t handle)
{
    handle->wait();
}
#else
extern "C" int
raw1394_loop_iterate(raw1394handle_t handle)
{
    return (handle->loopIterate() == kCFRunLoopRunHandledSource ? 0 : -1);
}
#endif

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
