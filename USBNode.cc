/*
 *  $Id: USBNode.cc 1655 2014-10-03 01:37:50Z ueshiba $
 */
#include <iostream>
#include "TU/USBNode_.h"

namespace TU
{
/************************************************************************
*  static data								*
************************************************************************/
static struct
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
};
    
/************************************************************************
*  static functions							*
************************************************************************/
static inline int32_t
address_to_request(nodeaddr_t addr)
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

    return -1;
}
    
static inline uint32_t
cycletime_to_subcycle(uint32_t cycletime)
{
    uint32_t	sec	 = (cycletime & 0xe000000) >> 25;
    uint32_t	cycle	 = (cycletime & 0x1fff000) >> 12;
    uint32_t	subcycle = (cycletime & 0x0000fff);

    return subcycle + 3072*(cycle + 8000*sec);
}

#if defined(DEBUG)
static std::ostream&
print_time(std::ostream& out, uint64_t localtime)
{
    uint32_t	usec = localtime % 1000;
    uint32_t	msec = (localtime / 1000) % 1000;
    uint32_t	sec  = localtime / 1000000;
    return out << sec << '.' << msec << '.' << usec;
}
#endif
    
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
USBNode::USBNode(u_int unit_spec_ID, uint64_t uniqId)
    :_handle(nullptr), _buffers(), _ready(), _run(false)
{
    try
    {
	for (DeviceIterator dev(_ctx); *dev; ++dev)	// for each device...
	{
	    using namespace	std;

	    libusb_device_descriptor	desc;
	    exec(libusb_get_device_descriptor(*dev, &desc),
		 "Failed to get device descriptor!!");

	    cerr << endl;
	    cerr << "Bus:\t\t" << dec << int(libusb_get_bus_number(*dev))
		 << endl;
	    cerr << "Address:\t" << int(libusb_get_device_address(*dev))
		 << endl;
	    cerr << "idVendor:\t" << hex << desc.idVendor << endl;
	    cerr << "idProduct:\t" << hex << desc.idProduct << endl;

	    for (const auto& product : usbProducts)
		if (desc.idVendor  == product.idVendor &&
		    desc.idProduct == product.idProduct)
		{
		    exec(libusb_open(*dev, &_handle),
			 "Failed to open device!!");

		    if (unitSpecId() == unit_spec_ID &&
			(uniqId == 0 || globalUniqueId() == uniqId))
		    {
			exec(libusb_set_configuration(_handle, 1),
			     "Failed to set configuration!!");
			exec(libusb_claim_interface(_handle, 0),
			     "Failed to claim interface!!");
			return;
		    }

		    libusb_close(_handle);
		    _handle = nullptr;
		}
	}

	if (!_handle)
	    throw std::runtime_error(
		      "USBNode::USBNode: cannot find specified device!!");
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
	     
USBNode::USBNode(uint16_t idVendor, uint16_t idProduct, uint32_t serialNumber)
    :_handle(nullptr), _buffers(), _ready(), _run(false)
{
    try
    {
	for (DeviceIterator dev(_ctx); *dev; ++dev)	// for each device...
	{
	    using namespace	std;

	    libusb_device_descriptor	desc;
	    exec(libusb_get_device_descriptor(*dev, &desc),
		 "Failed to get device descriptor!!");

	    cerr << endl;
	    cerr << "Bus:\t\t" << dec << int(libusb_get_bus_number(*dev))
		 << endl;
	    cerr << "Address:\t" << int(libusb_get_device_address(*dev))
		 << endl;
	    cerr << "idVendor:\t" << hex << desc.idVendor << endl;
	    cerr << "idProduct:\t" << hex << desc.idProduct << endl;

	    if (desc.idVendor  == idVendor && desc.idProduct == idProduct)
	    {
		exec(libusb_open(*dev, &_handle), "Failed to open device!!");

		if (desc.iSerialNumber)
		{
		    u_char	str[256];
		    exec(libusb_get_string_descriptor_ascii(
			     _handle, desc.iSerialNumber, str, sizeof(str)-1),
			 "Failed to get string descriptor!!");
		    uint32_t	descSerialNumber;
		    sscanf(reinterpret_cast<const char*>(str),
			   "%x", &descSerialNumber);

		    if (serialNumber == 0 || descSerialNumber == serialNumber)
			return;
		}
		else
		    break;
		    
		libusb_close(_handle);
		_handle = nullptr;
	    }
	}

	if (!_handle)
	    throw std::runtime_error(
		      "USBNode::USBNode: cannot find specified device!!");
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
    
    libusb_release_interface(_handle, 0);
    libusb_close(_handle);
}

nodeid_t
USBNode::nodeId() const
{
    return libusb_get_device_address(libusb_get_device(_handle));
}
    
//! ノード内の指定されたアドレスから4byteの値を読み出す
/*!
  \param addr	個々のノード内の絶対アドレス
 */
quadlet_t
USBNode::readQuadlet(nodeaddr_t addr) const
{
    const auto	request = address_to_request(addr);
    quadlet_t	quad;
    exec(libusb_control_transfer(_handle, 0xc0, request,
				 addr & 0xffff, (addr >> 16) & 0xffff,
				 reinterpret_cast<u_char*>(&quad),
				 sizeof(quad), REQUEST_TIMEOUT_MS),
	 "TU::USBNode::readQuadlet: failed to read from port!!");
    return quad;
}

//! ノード内の指定されたアドレスに4byteの値を書き込む
/*!
  \param addr	個々のノード内の絶対アドレス
  \param quad	書き込む4byteの値
 */
void
USBNode::writeQuadlet(nodeaddr_t addr, quadlet_t quad)
{
    const auto	request = address_to_request(addr);
    exec(libusb_control_transfer(_handle, 0x40, request,
				 addr & 0xffff, (addr >> 16) & 0xffff,
				 reinterpret_cast<u_char*>(&quad),
				 sizeof(quad), REQUEST_TIMEOUT_MS),
	 "TU::USBNode::writeQuadlet: failed to write to port!!");
}

//! isochronous受信用のバッファを割り当てる
/*!
  \param packet_size	受信するパケット1つあたりのサイズ(単位: byte)
  \param buf_size	バッファ1つあたりのサイズ(単位: byte)
  \param nb_buffers	割り当てるバッファ数
  \return		割り当てられたisochronous受信用のチャンネル
 */
u_char
USBNode::mapListenBuffer(u_int packet_size, u_int buf_size, u_int nb_buffers)
{
    unmapListenBuffer();
    
    _buffers.resize(nb_buffers);

    for (auto& buffer : _buffers)
    {
	buffer.map(this, buf_size);
	buffer.enqueue();			// 待機queueに入れる
    }

    _run    = true;				// 稼働フラグを立てる
    _thread = std::thread(&mainLoop, this);	// 子スレッドを起動

    return 0;
}

//! isochronousデータが受信されるのを待つ
/*!
  実際にデータが受信されるまで, 本関数は呼び出し側に制御を返さない. 
  \return	データの入ったバッファの先頭アドレス. 
 */
const u_char*
USBNode::waitListenBuffer()
{
    while (_ready.empty())
    {
	std::unique_lock<std::mutex>	lock(_mutex);
	_cond.wait(lock);		// 子スレッドが受信するのを待つ
    }
    return _ready.front()->data();	// 受信済みqueueの先頭データを返す
}

//! データ受信済みのバッファを再びキューイングして次の受信データに備える
void
USBNode::requeueListenBuffer()
{
    if (!_ready.empty())
    {
	_ready.front()->enqueue();	// 受信済みqueueの先頭を待機queueに入れて
	std::lock_guard<std::mutex>	lock(_mutex);
	_ready.pop();			// 受信済みqueueから取り除く
    }
}

//! すべての受信用バッファの内容を空にする
void
USBNode::flushListenBuffer()
{
}

//! ノードに割り当てたすべての受信用バッファを廃棄する
void
USBNode::unmapListenBuffer()
{
    if (_run)				// 子スレッドが走っていたら...
    {
	_run = false;			// 稼働フラグを落とす
	_thread.join();			// 子スレッドの終了を待つ
    }
    
    while (!_ready.empty())
	_ready.pop();
    
    for (auto& buffer : _buffers)
	buffer.unmap();
}

//! capture threadのmain loop
void
USBNode::mainLoop(USBNode* node)
{
    while (node->_run)
    {
	libusb_handle_events(_ctx);

	std::lock_guard<std::mutex>	lock(node->_mutex);
	node->_cond.notify_one();	// イベントが処理されたことを親に伝える
    }
}
    
#if 0
uint64_t
USBNode::cycletimeToLocaltime(uint32_t32_t cycletime) const
{
  // 現在のサイクル時刻と時刻を獲得する．
    uint32_t	cycletime0;
    uint32_t	localtime0;
    raw1394_read_cycle_timer(_handle, &cycletime0, &localtime0);

  // 現時刻と指定された時刻のサイクル時刻をサブサイクル値に直し，
  // 両者のずれを求める．
    uint32_t	subcycle0 = cycletime_to_subcycle(cycletime0);
    uint32_t	subcycle  = cycletime_to_subcycle(cycletime);
    uint64_t	diff	  = (subcycle0 + (128LL*8000LL*3072LL) - subcycle)
			  % (128LL*8000LL*3072LL);

  // ずれをmicro sec単位に直して(1 subcycle = 125/3072 usec)現在時刻から差し引く. 
    return localtime0 - (125LL*diff)/3072LL;
}

uint64_t
USBNode::cycleToLocaltime(uint32_t cycle) const
{
  // 現在のサイクル時刻と時刻を獲得する．
    uint32_t	cycletime0;
    uint64_t	localtime0;
    raw1394_read_cycle_timer(_handle, &cycletime0, &localtime0);

  // 現在のサイクル時刻からサイクル値(周期：8000)を取り出し，与えられた
  // サイクル値とのずれを求める．
    uint32_t	cycle0 = (cycletime0 & 0x1fff000) >> 12;
    uint32_t	diff   = (cycle0 + 8000 - cycle) % 8000;
    
  // ずれをmicro sec単位に直して(1 cycle = 125 micro sec)現在時刻から差し引く. 
    return localtime0 - uint64_t(125*diff);
}
#endif
/************************************************************************
*  class USBNode::Buffer						*
************************************************************************/
inline
USBNode::Buffer::Buffer()
    :_parent(nullptr), _size(0), _p(nullptr), _transfer(nullptr)
{
}

inline
USBNode::Buffer::~Buffer()
{
    unmap();
}
    
void
USBNode::Buffer::map(USBNode* parent, size_t size)
{
    unmap();
    
    _parent   = parent;
    _size     = size;
    _p	      = new u_char[_size];
    _transfer = libusb_alloc_transfer(0);

    libusb_fill_bulk_transfer(_transfer, _parent->_handle, 0x81,
			      _p, _size, &callback, this, 0);
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
    _p	    = nullptr;
    _size   = 0;
    _parent = nullptr;
}
    
inline void
USBNode::Buffer::enqueue() const
{
    exec(libusb_submit_transfer(_transfer), "libusb_submit_transfer failed!!");
}

void
USBNode::Buffer::callback(libusb_transfer* transfer)
{
    using namespace	std;
    
    if (transfer->status == LIBUSB_TRANSFER_CANCELLED)
    {
	cerr << "callback: CANCELLED" << endl;
	return;
    }
    
    Buffer* 	buffer = reinterpret_cast<Buffer*>(transfer->user_data);

    if (transfer->status != LIBUSB_TRANSFER_COMPLETED)
    {
	cerr << "callback: ERROR" << endl;
	buffer->enqueue();
	return;
    }
    else if (transfer->actual_length != transfer->length)
    {
	cerr << "callback: CORRUPT" << endl;
	buffer->enqueue();
	return;
    }

    auto	node = buffer->_parent;
    std::lock_guard<std::mutex>	lock(node->_mutex);
    node->_ready.push(buffer);
}
    
}
