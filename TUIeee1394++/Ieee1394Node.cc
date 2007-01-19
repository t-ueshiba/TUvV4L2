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
 *  $Id: Ieee1394Node.cc,v 1.9 2007-01-19 02:28:07 ueshiba Exp $
 */
#include "Ieee1394++.h"
#include <unistd.h>
#include <errno.h>
#include <string>
#include <libraw1394/csr.h>
#if defined(USE_VIDEO1394)
#  include <fcntl.h>
#  include <sys/ioctl.h>
#  include <sys/mman.h>
#  if !defined(VIDEO1394_IOC_LISTEN_CHANNEL)
#     define VIDEO1394_IOC_LISTEN_CHANNEL	VIDEO1394_LISTEN_CHANNEL
#     define VIDEO1394_IOC_UNLISTEN_CHANNEL	VIDEO1394_UNLISTEN_CHANNEL
#     define VIDEO1394_IOC_LISTEN_QUEUE_BUFFER	VIDEO1394_LISTEN_QUEUE_BUFFER
#     define VIDEO1394_IOC_LISTEN_WAIT_BUFFER	VIDEO1394_LISTEN_WAIT_BUFFER
#     define VIDEO1394_IOC_TALK_CHANNEL		VIDEO1394_TALK_CHANNEL
#     define VIDEO1394_IOC_UNTALK_CHANNEL	VIDEO1394_UNTALK_CHANNEL
#     define VIDEO1394_IOC_TALK_QUEUE_BUFFER	VIDEO1394_TALK_QUEUE_BUFFER
#     define VIDEO1394_IOC_TALK_WAIT_BUFFER	VIDEO1394_TALK_WAIT_BUFFER
#     define VIDEO1394_IOC_LISTEN_POLL_BUFFER	VIDEO1394_LISTEN_POLL_BUFFER
#  endif
#else
#  include <sys/time.h>
#endif

namespace TU
{
#if defined(USE_VIDEO1394)
/************************************************************************
*  static functions							*
************************************************************************/
static int
video1394_get_fd_and_check(int port_number)
{
    using namespace	std;

    char	dev[] = "/dev/video1394/x";
    dev[strlen(dev)-1] = '0' + port_number;
    int		fd = open(dev, O_RDWR);
    if (fd < 0)
	throw runtime_error(string("TU::raw1394_get_fd_and_check: failed to open video1394!! ") + strerror(errno));
    return fd;
}
#endif
#if !defined(__APPLE__)
/************************************************************************
*  class Ieee1394Node::Port						*
************************************************************************/
Ieee1394Node::Port::Port(int portNumber)
    :
#  if defined(USE_VIDEO1394)
     _fd(video1394_get_fd_and_check(portNumber)),
#  endif
     _portNumber(portNumber), _nodes(0)
{
}

Ieee1394Node::Port::~Port()
{
#  if defined(USE_VIDEO1394)
    close(_fd);
#  endif
}

u_char
Ieee1394Node::Port::registerNode(const Ieee1394Node& node)
{
  // Count the number of nodes already registered.
    u_char	nnodes = 0;
    for (int i = 0; i < 8*sizeof(_nodes); ++i)
	if ((_nodes & (0x1ULL << i)) != 0)
	    ++nnodes;

    _nodes |= (0x1ULL << (node.nodeId() & 0x3f));	// Register this node.

    return nnodes;
}

bool
Ieee1394Node::Port::unregisterNode(const Ieee1394Node& node)
{
    return (_nodes &= ~(0x1ULL << (node.nodeId() & 0x3f))) != 0;
}

bool
Ieee1394Node::Port::isRegisteredNode(const Ieee1394Node& node) const
{
    return (_nodes & (0x1ULL << (node.nodeId() & 0x3f))) != 0;
}

/************************************************************************
*  class Ieee1394Node							*
************************************************************************/
std::map<int, Ieee1394Node::Port*>	Ieee1394Node::_portMap;
#endif	// !__APPLE__

//! IEEE1394ノードオブジェクトを生成する
/*!
  \param unit_spec_ID	このノードの種類を示すID(ex. IEEE1394デジタルカメラ
			であれば，0x00a02d)
  \param uniqId		個々の機器固有の64bit ID．同一のIEEE1394 busに
			同一のunit_spec_IDを持つ複数の機器が接続されて
			いる場合，これによって同定を行う．
			0が与えられると，指定されたunit_spec_IDを持ち
			まだ#Ieee1394Nodeオブジェクトを割り当てられて
			いない機器のうち，一番最初にみつかったものがこの
			オブジェクトと結びつけられる．
  \param delay		IEEE1394カードの種類によっては，レジスタの読み書き
			(Ieee1394Node::readQuadlet(),
			Ieee1394Node::writeQuadlet())時に遅延を入れないと
			動作しないことがある．この遅延量をmicro second単位
			で指定する．(例: メルコのIFC-ILP3では1, DragonFly
			付属のボードでは0)
  \param sync_tag	1まとまりのデータを複数のパケットに分割して
			isochronousモードで受信する際に，最初のパケットに
			同期用のtagがついている場合は1を指定．そうでなけれ
			ば0を指定．
  \param flags		video1394のフラグ．VIDEO1394_SYNC_FRAMES, 
			VIDEO1394_INCLUDE_ISO_HEADERS,
			VIDEO1394_VARIABLE_PACKET_SIZEの組合わせ．
*/
Ieee1394Node::Ieee1394Node(u_int unit_spec_ID, u_int64 uniqId, u_int delay
#if defined(USE_VIDEO1394)
			   , int sync_tag, int flags
#endif
			   )
#if defined(__APPLE__)
    :_handle(raw1394_new_handle(unit_spec_ID, uniqId)), 
#else
    :_port(0), _handle(raw1394_new_handle()),
#endif
     _nodeId(0),
#if defined(USE_VIDEO1394)
     _mmap(), _current(0), _buf_size(0),
#else
     _channel(0), _current(0), _end(0),
#endif
     _buf(0), _filltime(), _delay(delay)
{
    using namespace	std;

  // Check whether the handle is valid.
    if (_handle == NULL)
	throw runtime_error(string("TU::Ieee1394Node::Ieee1394Node: failed to get raw1394handle!! ") + strerror(errno));
#if !defined(__APPLE__)
  // Get the number of ports.
    const int	nports = raw1394_get_port_info(_handle, NULL, 0);
    if (nports < 0)
	throw runtime_error(string("TU::Ieee1394Node::Ieee1394Node: failed to get port info!! ") + strerror(errno));
    raw1394_destroy_handle(_handle);

  // Find the specified node yet registered.
    for (int i = 0; i < nports; ++i)		// for each port...
    {
      // Has the i-th port already been created?
	map<int, Port*>::const_iterator	p = _portMap.find(i);
	_port = (p == _portMap.end() ? 0 : p->second);
	
	if ((_handle = raw1394_new_handle_on_port(i)) == NULL)
	    throw runtime_error(string("TU::Ieee1394Node::Ieee1394Node: failed to get raw1394handle and set it to the port!! ") + strerror(errno));
	nodeid_t	localId = raw1394_get_local_id(_handle);
	const int	nnodes  = raw1394_get_nodecount(_handle);
	for (int j  = 0; j < nnodes; ++j)	// for each node....
	{
	    _nodeId = (j | 0xffc0);
	    if (_nodeId != localId				 &&
		readValueFromUnitDirectory(0x12) == unit_spec_ID &&
		(uniqId == 0 || globalUniqueId() == uniqId))
	    {
		if (_port == 0)			// If i-th port is not present,
		{				//
		    _port = new Port(i);	// create
		    _portMap[i] = _port;	// and insert it to the map.
		    goto ok;
		}
		else if (!_port->isRegisteredNode(*this))
		    goto ok;
	    }
	}
	raw1394_destroy_handle(_handle);
    }
    throw runtime_error("TU::Ieee1394Node::Ieee1394Node: node with specified unit_spec_ID (and global_unique_ID) not found!!");

  ok:
#  if defined(USE_VIDEO1394)
    _mmap.channel     = _port->registerNode(*this);
    _mmap.sync_tag    = sync_tag;
    _mmap.nb_buffers  = 0;
    _mmap.buf_size    = 0;
    _mmap.packet_size = 0;
    _mmap.fps	      = 0;
    _mmap.flags	      = flags;
#  else
    _channel = _port->registerNode(*this);  // Register this node to the port.
#  endif
#endif	// !__APPLE__
    _filltime.tv_sec = _filltime.tv_usec = 0;
    raw1394_set_userdata(_handle, this);
}
	     
//! IEEE1394ノードオブジェクトを破壊する
Ieee1394Node::~Ieee1394Node()
{
    unmapListenBuffer();
#if !defined(__APPLE__)
    if (!_port->unregisterNode(*this))		// If no nodes on this port,
    {						//
	_portMap.erase(_port->portNumber());	// erase it from the map
	delete _port;				// and delete.
    }
#endif
    raw1394_destroy_handle(_handle);
}

#if defined(__APPLE__)
nodeaddr_t
Ieee1394Node::cmdRegBase() const
{
    return raw1394_command_register_base(_handle);
}
#endif

//! このノードに結び付けられている機器固有の64bit IDを返す
u_int64
Ieee1394Node::globalUniqueId() const
{
    u_int64	hi = readQuadletFromConfigROM(0x0c),
		lo = readQuadletFromConfigROM(0x10);
    return (hi << 32) | lo;
}

//! 与えられたkeyに対する値をUnit Dependent Directoryから読み出す
/*!
  \param key	keyすなわち4byteの並びのMSB側8bit
 */
u_int
Ieee1394Node::readValueFromUnitDependentDirectory(u_char key) const
{
  // Read length of Bus Info Block and skip it.
    u_int	offset = 0;
    quadlet_t	quad = readQuadletFromConfigROM(offset);
    offset += 4 * (1 + (quad >> 24));

  // Read unit_directory_offset.
    u_int	tmpOffset = readValueFromDirectory(0xd1, offset);
    offset += 4 * tmpOffset;

  // Read unit_dependent_directory_offset.
    tmpOffset = readValueFromDirectory(0xd4, offset);
    offset += 4 * tmpOffset;

    return readValueFromDirectory(key, offset);
}

//! 与えられたkeyに対する値をUnit Directoryから読み出す
/*!
  \param key	keyすなわち4byteの並びのMSB側8bit
 */
u_int
Ieee1394Node::readValueFromUnitDirectory(u_char key) const
{
  // Read length of Bus Info Block and skip it.
    u_int	offset = 0;
    quadlet_t	quad = readQuadletFromConfigROM(offset);
    offset += 4 * (1 + (quad >> 24));

  // Read unit_directory_offset.
    u_int	tmpOffset = readValueFromDirectory(0xd1, offset);
    offset += 4 * tmpOffset;

  // Read unit_spec_ID and return it.
    return readValueFromDirectory(key, offset);
}

u_int
Ieee1394Node::readValueFromDirectory(u_char key, u_int& offset) const
{
  // Read length of the directory in quadlets.
    quadlet_t	quad = readQuadletFromConfigROM(offset);
    u_int	length = quad >> 16;
    offset += 4;

  // Read each field of the directory.
    for (u_int i = 0; i < length; ++i)
    {
	quad = readQuadletFromConfigROM(offset);
	if (u_char(quad >> 24) == key)
	    return (quad & 0xffffff);
	offset += 4;
    }

    throw std::runtime_error("TU::Ieee1394Node::readValueFromDirectory: field with specified key not found!!");

    return ~0;
}

quadlet_t
Ieee1394Node::readQuadletFromConfigROM(u_int offset) const
{
    return readQuadlet(CSR_REGISTER_BASE + CSR_CONFIG_ROM + offset);
}
    
//! ノード内の指定されたアドレスから4byteの値を読み出す
/*!
  \param addr	個々のノード内の絶対アドレス
 */
quadlet_t
Ieee1394Node::readQuadlet(nodeaddr_t addr) const
{
    using namespace	std;

    quadlet_t	quad;
    if (raw1394_read(_handle, _nodeId, addr, 4, &quad) < 0)
	throw runtime_error(string("TU::Ieee1394Node::readQuadlet: failed to read from port!! ") + strerror(errno));
    if (_delay != 0)
	::usleep(_delay);
    return quadlet_t(ntohl(quad));
}

//! ノード内の指定されたアドレスに4byteの値を書き込む
/*!
  \param addr	個々のノード内の絶対アドレス
  \param quad	書き込む4byteの値
 */
void
Ieee1394Node::writeQuadlet(nodeaddr_t addr, quadlet_t quad)
{
    using namespace	std;

    quad = htonl(quad);
    if (raw1394_write(_handle, _nodeId, addr, 4, &quad) < 0)
	throw runtime_error(string("TU::Ieee1394Node::writeQuadlet: failed to write to port!! ") + strerror(errno));
    if (_delay != 0)
	::usleep(_delay);
}

//! isochronous受信用のバッファを割り当てる
/*!
  \param packet_size	受信するパケット1つあたりのサイズ(単位: byte)
  \param buf_size	バッファ1つあたりのサイズ(単位: byte)
  \param nb_buffers	割り当てるバッファ数
  \return		割り当てられたisochronous受信用のチャンネル
 */
u_char
Ieee1394Node::mapListenBuffer(size_t packet_size,
			      size_t buf_size, u_int nb_buffers)
{
    using namespace	std;
    
  // Unmap previously mapped buffer and unlisten the channel.
    unmapListenBuffer();

#if defined(USE_VIDEO1394)
  // Change buffer size and listen to the channel.
  //   *Caution: _mmap.buf_size may be changed by VIDEO1394_LISTEN_CHANNEL.
    _mmap.nb_buffers  = nb_buffers;
    _mmap.buf_size    = _buf_size = buf_size;
    _mmap.packet_size = packet_size;
    if (ioctl(_port->fd(), VIDEO1394_IOC_LISTEN_CHANNEL, &_mmap) < 0)
	throw runtime_error(string("TU::Ieee1394Node::mapListenBuffer: VIDEO1394_IOC_LISTEN_CHANNEL failed!! ") + strerror(errno));
    for (int i = 0; i < _mmap.nb_buffers; ++i)
    {
	video1394_wait	wait;
	wait.channel = _mmap.channel;
	wait.buffer  = i;
	if (ioctl(_port->fd(), VIDEO1394_IOC_LISTEN_QUEUE_BUFFER, &wait) < 0)
	    throw runtime_error(string("Ieee1394Node::mapListenBuffer: VIDEO1394_IOC_LISTEN_QUEUE_BUFFER failed!! ") + strerror(errno));
    }

  // Reset buffer status and re-map new buffer.
    if ((_buf = (u_char*)mmap(0, _mmap.nb_buffers * _mmap.buf_size,
			      PROT_READ, MAP_SHARED, _port->fd(), 0))
	== (u_char*)-1)
    {
	_buf = 0;
	throw runtime_error(string("Ieee1394Node::mapListenBuffer: mmap failed!! ") + strerror(errno));
    }

    usleep(100000);
    return _mmap.channel;
#else
    _buf = _current = new u_char[buf_size];
    _end = _buf + buf_size;

    const u_int	npackets = (buf_size - 1) / packet_size + 1;
    if (raw1394_iso_recv_init(_handle, &Ieee1394Node::receive,
			      nb_buffers * npackets, packet_size, _channel,
			      RAW1394_DMA_BUFFERFILL, npackets) < 0)
	throw runtime_error(string("TU::Ieee1394Node::mapListenBuffer: failed to initialize iso reception!! ") + strerror(errno));
    if (raw1394_iso_recv_start(_handle, -1, -1, 0) < 0)
	throw runtime_error(string("TU::Ieee1394Node::mapListenBuffer: failed to start iso reception!! ") + strerror(errno));
    return _channel;
#endif
}

//! isochronousデータが受信されるのを待つ
/*!
  実際にデータが受信されるまで，本関数は呼び出し側に制御を返さない．
  \return	データの入ったバッファの先頭アドレス．データのサイズは
		#bufferSize()で知ることができる．
 */
const u_char*
Ieee1394Node::waitListenBuffer()
{
    using namespace	std;

#if defined(USE_VIDEO1394)
    video1394_wait	wait;
    wait.channel = _mmap.channel;
    wait.buffer  = _current;
    if (ioctl(_port->fd(), VIDEO1394_IOC_LISTEN_WAIT_BUFFER, &wait) < 0)
	throw runtime_error(string("TU::Ieee1394Node::waitListenBuffer: VIDEO1394_IOC_LISTEN_WAIT_BUFFER failed!! ") + strerror(errno));
    _filltime = wait.filltime;	// time of buffer full.	

    return _buf + _current * _mmap.buf_size;
#else
    while (_current < _end)
	raw1394_loop_iterate(_handle);
    gettimeofday(&_filltime, NULL);

    return _buf;
#endif    
}

//! データ受信済みのバッファを再びキューイングして次の受信データに備える
void
Ieee1394Node::requeueListenBuffer()
{
#if defined(USE_VIDEO1394)
    using namespace	std;

    video1394_wait	wait;
    wait.channel = _mmap.channel;
    wait.buffer	 = _current;
    if (ioctl(_port->fd(), VIDEO1394_IOC_LISTEN_QUEUE_BUFFER, &wait) < 0)
	throw runtime_error(string("TU::Ieee1394Node::requeueListenBuffer: VIDEO1394_IOC_LISTEN_QUEUE_BUFFER failed!! ") + strerror(errno));
    ++_current %= _mmap.nb_buffers;	// next buffer.
#else
    _current = _buf;
#endif
}

//! すべての受信用バッファの内容を空にする
void
Ieee1394Node::flushListenBuffer()
{
#if defined(USE_VIDEO1394)
  // Force flushing by doing unmap and then map buffer.
    if (_buf != 0)
	mapListenBuffer(_mmap.packet_size, _buf_size, _mmap.nb_buffers);
    
  // POLL(kernel-2.4以降のみで有効)してREADY状態のバッファを全てrequeueする．
  // 1つのバッファが一杯になる前にisochronous転送が停止されると，そのバッファ
  // はREADY状態にはならずQUEUED状態のままなので，不完全なデータを持ったまま
  // 残ってしまう．したがって，この方法は良くない．(2002.3.7)
  /*    if (_buf != 0)
	for (;;)
	{
	    video1394_wait	wait;
	    wait.channel = _mmap.channel;
	    wait.buffer  = _current;
	    if (ioctl(_port->fd(), VIDEO1394_IOC_LISTEN_POLL_BUFFER, &wait) < 0)
		break;
	    _nready = 1 + wait.buffer;
	    requeueListenBuffer();
	    }*/
  // "_nready" must be 0 here(no available buffers).
#else
    using namespace	std;

    if (raw1394_iso_recv_flush(_handle) < 0)
	throw runtime_error(string("TU::Ieee1394Node::flushListenBuffer: failed to flush iso receive buffer!! ") + strerror(errno));
    _current = _buf;
#endif
}

//! ノードに割り当てたすべての受信用バッファを廃棄する
void
Ieee1394Node::unmapListenBuffer()
{
    using namespace	std;

    if (_buf != 0)
    {
#if defined(USE_VIDEO1394)
	munmap(_buf, _mmap.nb_buffers * _mmap.buf_size);
	_buf = 0;				// Reset buffer status.
	_buf_size = _current = 0;		// ibid.
	if (ioctl(_port->fd(), VIDEO1394_IOC_UNLISTEN_CHANNEL, &_mmap.channel) < 0)
	    throw runtime_error(string("TU::Ieee1394Node::unmapListenBuffer: VIDEO1394_IOC_UNLISTEN_CHANNEL failed!! ") + strerror(errno));
#else
	raw1394_iso_stop(_handle);
	raw1394_iso_shutdown(_handle);
    
	delete [] _buf;
	_buf = _current = _end = 0;
#endif
    }
}

#if !defined(USE_VIDEO1394)
//! このノードに割り当てられたisochronous受信用バッファを満たす
/*!
  \param data	受信用バッファにセーブするデータ
  \param len	データのバイト数
  \param sy	先頭のデータであれば1, そうでなければ0
*/
raw1394_iso_disposition
Ieee1394Node::receive(raw1394handle_t handle,
		      u_char* data, u_int len,
		      u_char channel, u_char tag, u_char sy,
		      u_int cycle, u_int dropped)
{
    Ieee1394Node*	node = (Ieee1394Node*)raw1394_get_userdata(handle);
    if (sy)
	node->_current = node->_buf;
    u_char* const	next = node->_current + len;
    if (next <= node->_end)
    {
	memcpy(node->_current, data, len);
	node->_current = next;
    }
    return RAW1394_ISO_DEFER;
  /*    return (node->_current < node->_end ? RAW1394_ISO_DEFER
	: RAW1394_ISO_ERROR);*/
}
#endif
 
}
