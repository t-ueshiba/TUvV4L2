/*
 *  $Id: USBNode.cc 1655 2014-10-03 01:37:50Z ueshiba $
 */
#include <iostream>
#include <chrono>
#include "USBNode_.h"

namespace TU
{
/************************************************************************
*  static data								*
************************************************************************/
static constexpr struct
{
    uint16_t	idVendor;
    uint16_t	idProduct;
    
} usbProducts[] =
{
    {0x1e10, 0x2000},	// Point Grey Firefly MV Color
    {0x1e10, 0x2001},	// Point Grey Firefly MV Mono
    {0x1e10, 0x2002},	// Point Grey Firefly MV HiRes Color
    {0x1e10, 0x2003},	// Point Grey Firefly MV HiRes Mono
    {0x1e10, 0x2004},	// Point Grey Chameleon Color
    {0x1e10, 0x2005},	// Point Grey Chameleon Mono
    {0x1e10, 0x3000},	// Point Grey Flea 3
    {0x1e10, 0x3005},	// Point Grey Flea 3 (FL3-U3_13Y3M)
    {0x1e10, 0x3006},	// Point Grey Flea 3 (FL3-U3_13S2C)
    {0x1e10, 0x3008},	// Point Grey Flea 3 (FL3-U3_88S2C)
    {0x1e10, 0x300a},	// Point Grey Flea 3 (FL3-U3_13E4C)
    {0x1e10, 0x300b},	// Point Grey Flea 3 (FL3-U3_13E4M)
    {0x1e10, 0x3103},	// Point Grey Grasshopper 3 (GS3-U3-23S6M)
    {0x1e10, 0x3300},	// Point Grey Flea 3 (FL3-U3_13Y3M, Firmware-2.x.x)
};

static constexpr u_int		RequestTimeout	= 1000;
static constexpr uint8_t	EndpointAddress	= 0x81;
    
/************************************************************************
*  static functions							*
************************************************************************/
static inline uint8_t
address_to_request(IIDCNode::nodeaddr_t addr)
{
    switch (addr >> 32)
    {
      case 0xffff:
	return 0x7f;
      case 0xd000:
	return 0x80;
      case 0xd0001:
	return 0x81;
      default:
	break;
    }

    throw std::runtime_error("Invalid high address for request!!");

    return 0;
}
    
/************************************************************************
*  class USBNode							*
************************************************************************/
USBNode::Context	USBNode::_ctx;
    
//! USBノードオブジェクトを生成する
/*!
  \param uniqId		個々の機器固有の64bit ID. 同一のIEEE1394 busに
			同一のunit_spec_IDを持つ複数の機器が接続されて
			いる場合, これによって同定を行う. 
			0が与えられると, 指定されたunit_spec_IDを持ち
			まだ#USBNodeオブジェクトを割り当てられて
			いない機器のうち, 一番最初にみつかったものがこの
			オブジェクトと結びつけられる. 
*/
USBNode::USBNode(uint32_t unit_spec_ID, uint64_t uniqId)
    :_handle(nullptr), _interface(-1), _altsetting(-1), _iso_handle(nullptr),
     _nbuffers(0), _buffers(nullptr), _ready(),
     _run(false), _mutex(), _cond(), _thread()
{
    try
    {
	setHandle(unit_spec_ID, uniqId);
	setEndpoint();
    }
    catch (const std::runtime_error& err)
    {
	if (_handle)
	{
	    libusb_close(_handle);
	    _handle = nullptr;
	}
	throw err;
    }
}
	     
//! USBノードオブジェクトを破壊する
USBNode::~USBNode()
{
    unmapListenBuffer();
    libusb_close(_handle);
}

IIDCNode::nodeid_t
USBNode::nodeId() const
{
    return libusb_get_device_address(libusb_get_device(_handle));
}
    
IIDCNode::quadlet_t
USBNode::readQuadlet(nodeaddr_t addr) const
{
    quadlet_t	quad;
    check(libusb_control_transfer(_handle,
				  LIBUSB_ENDPOINT_IN |
				  LIBUSB_REQUEST_TYPE_VENDOR |
				  LIBUSB_RECIPIENT_DEVICE,
				  address_to_request(addr),
				  addr & 0xffff, (addr >> 16) & 0xffff,
				  reinterpret_cast<u_char*>(&quad),
				  sizeof(quad), RequestTimeout),
	  "USBNode::readQuadlet: failed to read from port!!");
    return quad;
}

void
USBNode::writeQuadlet(nodeaddr_t addr, quadlet_t quad)
{
    check(libusb_control_transfer(_handle,
				  LIBUSB_ENDPOINT_OUT |
				  LIBUSB_REQUEST_TYPE_VENDOR |
				  LIBUSB_RECIPIENT_DEVICE,
				  address_to_request(addr),
				  addr & 0xffff, (addr >> 16) & 0xffff,
				  reinterpret_cast<u_char*>(&quad),
				  sizeof(quad), RequestTimeout),
	  "USBNode::writeQuadlet: failed to write to port!!");
}

u_char
USBNode::mapListenBuffer(u_int packet_size, u_int buf_size, u_int nb_buffers)
{
    unmapListenBuffer();

    _nbuffers = nb_buffers;
    _buffers  = new Buffer[_nbuffers];

    startIso();

  // バッファサイズをendpointのパケットサイズの整数倍にしてoverrunを阻止する
    const auto	maxPacketSize = libusb_get_max_packet_size(
				    libusb_get_device(_iso_handle),
				    EndpointAddress);
    buf_size = maxPacketSize * ((buf_size - 1) / maxPacketSize + 1);

    for (size_t i = 0; i < _nbuffers; ++i)
    {
	_buffers[i].map(this, buf_size);
	_buffers[i].enqueue();			// 待機queueに入れる
    }

    _run    = true;				// 稼働フラグを立てる
    _thread = std::thread(mainLoop, this);	// 子スレッドを起動
    
    return 0;
}

void
USBNode::unmapListenBuffer()
{
    if (_run)					// 子スレッドが走っていたら...
    {
	for (size_t i = 0; i < _nbuffers; ++i)
	    _buffers[i].cancel();

	_run = false;				// 稼働フラグを落とす
	_thread.join();				// 子スレッドの終了を待つ
	stopIso();
    }
    
    while (!_ready.empty())
	_ready.pop();

    delete [] _buffers;
    _buffers  = nullptr;
    _nbuffers = 0;
}

const void*
USBNode::waitListenBuffer()
{
    std::unique_lock<std::mutex>	lock(_mutex);
    _cond.wait(lock, [&]{ return !_ready.empty(); });	// 子スレッドの受信を待つ
    return _ready.front()->data();	// 受信済みqueueの先頭データを返す
}

void
USBNode::requeueListenBuffer()
{
    std::lock_guard<std::mutex>	lock(_mutex);
    if (!_ready.empty())
    {
	_ready.front()->enqueue();	// 受信済みqueueの先頭を待機queueに入れて
	_ready.pop();			// 受信済みqueueから取り除く
    }
}

void
USBNode::flushListenBuffer()
{
}

uint32_t
USBNode::getCycletime(uint64_t& localtime) const
{
    using namespace	std::chrono;
    
    constexpr nodeaddr_t	CMD_REG_BASE = 0xfffff0f00000ULL;
    constexpr uint32_t		CYCLE_TIME   = 0x1ea8;
    
    localtime = duration_cast<microseconds>(
		    high_resolution_clock::now().time_since_epoch()).count();
    return readQuadlet(CMD_REG_BASE + CYCLE_TIME);
}

void
USBNode::setHandle(uint32_t unit_spec_ID, uint64_t uniqId)
{
    for (DeviceIterator dev(_ctx); *dev; ++dev)	// for each device...
    {
	libusb_device_descriptor	desc;
	check(libusb_get_device_descriptor(*dev, &desc),
	      "Failed to get device descriptor!!");
#if defined(DEBUG)
	using namespace	std;

	cerr << showbase << hex
	     << "\nBus:\t\t"	 << int(libusb_get_bus_number(*dev))
	     << "\nAddress:\t"   << int(libusb_get_device_address(*dev))
	     << "\nidVendor:\t"  << desc.idVendor
	     << "\nidProduct:\t" << desc.idProduct << dec << endl;
#endif
	for (const auto& product : usbProducts)
	    if (desc.idVendor  == product.idVendor &&
		desc.idProduct == product.idProduct)
	    {
		check(libusb_open(*dev, &_handle), "Failed to open device!!");
	    
		if ((unitSpecId() == unit_spec_ID)		&&
		    (uniqId == 0 || globalUniqueId() == uniqId)	&&
		    isUnique())
		{
		    if (libusb_kernel_driver_active(_handle, 0))
			check(libusb_detach_kernel_driver(_handle, 0),
			      "Fialed to detach kernel driver!!");
		    check(libusb_set_configuration(_handle, 1),
			  "Failed to set configuration!!");

		    addToList();
		    return;
		}

		libusb_close(_handle);
		_handle = nullptr;
	    }
    }

    throw std::runtime_error("No device with specidied unit_spec_ID and globalUniqId found!!");
}

void
USBNode::setEndpoint()
{
    libusb_config_descriptor*	config;
    check(libusb_get_active_config_descriptor(libusb_get_device(_handle),
					      &config),
	  "Failed to get config descriptor!!");

    for (uint8_t i = 0; i < config->bNumInterfaces; ++i)
    {
	const auto&	interface = config->interface[i];
	
	for (int j = 0; j < interface.num_altsetting; ++j)
	{
	    const auto&	altsetting = interface.altsetting[j];

	    for (uint8_t k = 0; k < altsetting.bNumEndpoints; ++k)
	    {
		const auto&	endpoint = altsetting.endpoint[k];
		
		if (endpoint.bEndpointAddress == EndpointAddress)
		{
		    _interface  = i;
		    _altsetting = j;
		    
		    libusb_free_config_descriptor(config);
		    return;
		}
	    }
	}
    }

    libusb_free_config_descriptor(config);
}
    
void
USBNode::startIso()
{
    const auto	bus  = libusb_get_bus_number(libusb_get_device(_handle));
    const auto	addr = libusb_get_device_address(libusb_get_device(_handle));
    
    for (DeviceIterator dev(_ctx); *dev; ++dev)	// for each device...
	if ((libusb_get_bus_number(*dev)     == bus ) &&
	    (libusb_get_device_address(*dev) == addr))
	{
	    check(libusb_open(*dev, &_iso_handle), "Failed to open device!!");
	    check(libusb_claim_interface(_iso_handle, _interface),
		  "USBNode::startIso: Failed to claim interface!!");
	    check(libusb_set_interface_alt_setting(_iso_handle,
						   _interface, _altsetting),
		  "USBNode::startIso: Failed to set interface alt setting!!");

	    return;
	}
}

void
USBNode::stopIso()
{
    libusb_release_interface(_iso_handle, _interface);
    libusb_close(_iso_handle);
    _iso_handle = nullptr;
}
    
//! capture threadのmain loop
void
USBNode::mainLoop(USBNode* node)
{
    while (node->_run)
    {
	timeval	timeout{0, 100000};
	libusb_handle_events_timeout(node->_ctx, &timeout);
	node->_cond.notify_all();	// イベントが処理されたことを親に伝える
    }
}
    
/************************************************************************
*  class USBNode::Buffer						*
************************************************************************/
USBNode::Buffer::Buffer()
    :_node(nullptr), _p(nullptr), _transfer(nullptr)
{
}

USBNode::Buffer::~Buffer()
{
    unmap();
}
    
void
USBNode::Buffer::map(USBNode* node, u_int size)
{
    unmap();
    
    _node     = node;
    _p	      = new u_char[size];
    _transfer = libusb_alloc_transfer(0);
    libusb_fill_bulk_transfer(_transfer, _node->_iso_handle, EndpointAddress,
			      _p, size, callback, this, 0);
}
    
void
USBNode::Buffer::unmap()
{
    if (_transfer)
    {
	libusb_free_transfer(_transfer);
	_transfer = nullptr;
    }
    delete [] _p;
    _p	  = nullptr;
    _node = nullptr;
}
    
void
USBNode::Buffer::enqueue() const
{
    check(libusb_submit_transfer(_transfer), "libusb_submit_transfer failed!!");
}

void
USBNode::Buffer::cancel() const
{
    libusb_cancel_transfer(_transfer);
}
    
void
USBNode::Buffer::callback(libusb_transfer* transfer)
{
    using namespace	std;
    
    if (transfer->status == LIBUSB_TRANSFER_CANCELLED)
    {
#if defined(DEBUG)
	cerr << "USBNode::Buffer::callback(): CANCELLED: " << transfer << endl;
#endif
	return;
    }
    
    auto	buffer = static_cast<const Buffer*>(transfer->user_data);

    if (transfer->status != LIBUSB_TRANSFER_COMPLETED)
    {
#if defined(DEBUG)
	cerr << "USBNode::Buffer::callback(): ERROR: " << transfer << endl;
#endif
	buffer->enqueue();
	return;
    }

#if defined(DEBUG)
    cerr << "USBNode::Buffer::callback(): buffer_length = "
	 << transfer->length
	 << ", autual_lenth = " << transfer->actual_length
	 << endl;
#endif
    std::lock_guard<std::mutex>	lock(buffer->_node->_mutex);
    buffer->_node->_ready.push(buffer);
}
    
}
