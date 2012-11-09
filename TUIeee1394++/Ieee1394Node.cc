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
 *  $Id$
 */
#include "TU/Ieee1394++.h"
#include <unistd.h>
#include <errno.h>
#include <string>
#include <algorithm>
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
/************************************************************************
*  static functions							*
************************************************************************/
#if defined(USE_VIDEO1394)
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
static inline u_int32_t
cycletime_to_subcycle(u_int32_t cycletime)
{
    u_int32_t	sec	 = (cycletime & 0xe000000) >> 25;
    u_int32_t	cycle	 = (cycletime & 0x1fff000) >> 12;
    u_int32_t	subcycle = (cycletime & 0x0000fff);

    return subcycle + 3072*(cycle + 8000*sec);
}

#if defined(DEBUG)
static std::ostream&
print_time(std::ostream& out, u_int64_t localtime)
{
    u_int32_t	usec = localtime % 1000;
    u_int32_t	msec = (localtime / 1000) % 1000;
    u_int32_t	sec  = localtime / 1000000;
    return out << sec << '.' << msec << '.' << usec;
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
			であれば, 0x00a02d)
  \param uniqId		個々の機器固有の64bit ID. 同一のIEEE1394 busに
			同一のunit_spec_IDを持つ複数の機器が接続されて
			いる場合, これによって同定を行う. 
			0が与えられると, 指定されたunit_spec_IDを持ち
			まだ#Ieee1394Nodeオブジェクトを割り当てられて
			いない機器のうち, 一番最初にみつかったものがこの
			オブジェクトと結びつけられる. 
  \param delay		IEEE1394カードの種類によっては, レジスタの読み書き
			(Ieee1394Node::readQuadlet(),
			Ieee1394Node::writeQuadlet())時に遅延を入れないと
			動作しないことがある. この遅延量をmicro second単位
			で指定する. (例: メルコのIFC-ILP3では1, DragonFly
			付属のボードでは0)
  \param sync_tag	1まとまりのデータを複数のパケットに分割して
			isochronousモードで受信する際に, 最初のパケットに
			同期用のtagがついている場合は1を指定. そうでなけれ
			ば0を指定. 
  \param flags		video1394のフラグ. VIDEO1394_SYNC_FRAMES, 
			VIDEO1394_INCLUDE_ISO_HEADERS,
			VIDEO1394_VARIABLE_PACKET_SIZEの組合わせ. 
*/
Ieee1394Node::Ieee1394Node(u_int unit_spec_ID, u_int64_t uniqId, u_int delay
#if defined(USE_VIDEO1394)
			   , int sync_tag, int flags
#endif
			   )
#if defined(__APPLE__)
    :_handle(raw1394_new_handle(unit_spec_ID, uniqId)), 
#else
    :_port(0), _handle(raw1394_new_handle()),
#endif
     _nodeId(0), _delay(delay), _buf(0), _arrivaltime(0),
#if defined(USE_VIDEO1394)
     _mmap(), _current(0), _buf_size(0)
#else
     _channel(0), _mid(0), _end(0), _current(0), _arrivaltime_next(0)
#endif

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
	for (int j = 0; j < nnodes; ++j)	// for each node....
	{
	    _nodeId = (j | 0xffc0);

	    try
	    {
		if (_nodeId != localId				     &&
		    readValueFromUnitDirectory(0x12) == unit_spec_ID &&
		    (uniqId == 0 || globalUniqueId() == uniqId))
		{
		    if (_port == 0)		// If i-th port is not present,
		    {				//
			_port = new Port(i);	// create
			_portMap[i] = _port;	// and insert it to the map.
			goto ok;
		    }
		    else if (!_port->isRegisteredNode(*this))
			goto ok;
		}
	    }
	    catch (exception& err)
	    {
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
u_int64_t
Ieee1394Node::globalUniqueId() const
{
    u_int64_t	hi = readQuadletFromConfigROM(0x0c),
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
	    throw runtime_error(string("TU::Ieee1394Node::mapListenBuffer: VIDEO1394_IOC_LISTEN_QUEUE_BUFFER failed!! ") + strerror(errno));
    }

  // Reset buffer status and re-map new buffer.
    if ((_buf = (u_char*)mmap(0, _mmap.nb_buffers * _mmap.buf_size,
			      PROT_READ, MAP_SHARED, _port->fd(), 0))
	== (u_char*)-1)
    {
	_buf = 0;
	throw runtime_error(string("TU::Ieee1394Node::mapListenBuffer: mmap failed!! ") + strerror(errno));
    }

    usleep(100000);
    return _mmap.channel;
#else
  // バッファ1つ分のデータを転送するために必要なパケット数
    const u_int	npackets = (buf_size - 1) / packet_size + 1;

  // raw1394_loop_iterate()は，interval個のパケットを受信するたびにユーザ側に
  // 制御を返す．libraw1394ではこのデフォルト値はパケット数の1/4である．ただし，
  // 512パケットを越える値を指定すると，raw1394_loop_iterate()から帰ってこなく
  // なるようである．
#  if defined(__APPLE__)
    const u_int	interval = npackets;
#  else
    const u_int	interval = std::min(npackets/4, 512U);
#  endif
#  if defined(DEBUG)
    cerr << "mapListenBuffer: npackets = " << npackets
	 << ", interval = " << interval << endl;
#  endif
    _buf     = new u_char[buf_size + interval * packet_size];
    _mid     = _buf + buf_size;
    _end     = _mid + interval * packet_size;
    _current = _buf;

    if (raw1394_iso_recv_init(_handle, &Ieee1394Node::receive,
			      nb_buffers * npackets, packet_size, _channel,
			      RAW1394_DMA_BUFFERFILL, interval) < 0)
	throw runtime_error(string("TU::Ieee1394Node::mapListenBuffer: failed to initialize iso reception!! ") + strerror(errno));
    if (raw1394_iso_recv_start(_handle, -1, -1, 0) < 0)
	throw runtime_error(string("TU::Ieee1394Node::mapListenBuffer: failed to start iso reception!! ") + strerror(errno));
    return _channel;
#endif
}

//! isochronousデータが受信されるのを待つ
/*!
  実際にデータが受信されるまで, 本関数は呼び出し側に制御を返さない. 
  \return	データの入ったバッファの先頭アドレス. 
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

  // wait.filltimeは，全パケットが到着してバッファが一杯になった時刻を表す．
  // これから最初のパケットが到着した時刻を推定するために，バッファあたりの
  // パケット数にパケット到着間隔(125 micro sec)を乗じた数を減じる．
    _arrivaltime = u_int64_t(wait.filltime.tv_sec)*1000000LL
		 + u_int64_t(wait.filltime.tv_usec)
		 - u_int64_t(((_mmap.buf_size - 1)/_mmap.packet_size + 1)*125);

    return _buf + _current * _mmap.buf_size;
#else
#  if defined(DEBUG)
    cerr << "*** BEGIN [waitListenBuffer] ***" << endl;
#  endif
    while (_current < _mid)		// [_buf, _mid)が満たされるまで
    {
	raw1394_loop_iterate(_handle);	// パケットを受信する．
#  if defined(DEBUG)
	cerr << "wait: current = " << _current - _buf << endl;
#  endif
    }
#  if defined(DEBUG)
    cerr << "*** END   [waitListenBuffer] ***" << endl;
#  endif

    return _buf;
#endif
}

//! データ受信済みのバッファを再びキューイングして次の受信データに備える
void
Ieee1394Node::requeueListenBuffer()
{
    using namespace	std;
    
#if defined(USE_VIDEO1394)
    video1394_wait	wait;
    wait.channel = _mmap.channel;
    wait.buffer	 = _current;
    if (ioctl(_port->fd(), VIDEO1394_IOC_LISTEN_QUEUE_BUFFER, &wait) < 0)
	throw runtime_error(string("TU::Ieee1394Node::requeueListenBuffer: VIDEO1394_IOC_LISTEN_QUEUE_BUFFER failed!! ") + strerror(errno));
    ++_current %= _mmap.nb_buffers;	// next buffer.
#else
    if (_current != 0)
    {
      // [_buf, _mid) を廃棄し [_mid, _current)をバッファ領域の先頭に移す
	const size_t	len = _current - _mid;
	memcpy(_buf, _mid, len);
	_current     = _buf + len;
	_arrivaltime = _arrivaltime_next;
#  if defined(DEBUG)
	cerr << "*** BEGIN [requeueListenBuffer] ***" << endl;
	cerr << "      current = " << _current - _buf
	     << " (" << len << " bytes moved...)" << endl;
	cerr << "*** END   [requeueListenBuffer] ***" << endl;
#  endif
    }
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
	_buf = _mid = _end = _current = 0;
#endif
    }
}

u_int64_t
Ieee1394Node::cycletimeToLocaltime(u_int32_t cycletime) const
{
  // 現在のサイクル時刻と時刻を獲得する．
    u_int32_t	cycletime0;
    u_int64_t	localtime0;
    raw1394_read_cycle_timer(_handle, &cycletime0, &localtime0);

  // 現時刻と指定された時刻のサイクル時刻をサブサイクル値に直し，
  // 両者のずれを求める．
    u_int32_t	subcycle0 = cycletime_to_subcycle(cycletime0);
    u_int32_t	subcycle  = cycletime_to_subcycle(cycletime);
    u_int64_t	diff	  = (subcycle0 + (128LL*8000LL*3072LL) - subcycle)
			  % (128LL*8000LL*3072LL);

  // ずれをmicro sec単位に直して(1 subcycle = 125/3072 usec)現在時刻から差し引く. 
    return localtime0 - (125LL*diff)/3072LL;
}

u_int64_t
Ieee1394Node::cycleToLocaltime(u_int32_t cycle) const
{
  // 現在のサイクル時刻と時刻を獲得する．
    u_int32_t	cycletime0;
    u_int64_t	localtime0;
    raw1394_read_cycle_timer(_handle, &cycletime0, &localtime0);

  // 現在のサイクル時刻からサイクル値(周期：8000)を取り出し，与えられた
  // サイクル値とのずれを求める．
    u_int32_t	cycle0 = (cycletime0 & 0x1fff000) >> 12;
    u_int32_t	diff   = (cycle0 + 8000 - cycle) % 8000;
    
  // ずれをmicro sec単位に直して(1 cycle = 125 micro sec)現在時刻から差し引く. 
    return localtime0 - u_int64_t(125*diff);
}

#if !defined(USE_VIDEO1394)
//! このノードに割り当てられたisochronous受信用バッファにパケットデータを転送する
/*!
  本ハンドラは，パケットが1つ受信されるたびに呼び出される．また，mapListenBuffer()
  内の raw1394_iso_recv_init() を呼んだ時点で既にisochronous転送が始まっている
  場合は，waitListenBuffer() 内で raw1394_loop_iterate() を呼ばなくてもこの
  ハンドラが呼ばれることがあるため，buffer overrun を防ぐ方策はこのハンドラ内で
  とっておかなければならない．
  \param data	パケットデータ
  \param len	パケットデータのバイト数
  \param sy	フレームの先頭パケットであれば1, そうでなければ0
*/
raw1394_iso_disposition
Ieee1394Node::receive(raw1394handle_t handle,
		      u_char* data, u_int len,
		      u_char channel, u_char tag, u_char sy,
		      u_int cycle, u_int dropped)
{
    using namespace	std;

    Ieee1394Node* const	node = (Ieee1394Node*)raw1394_get_userdata(handle);

  // [_buf, _mid)の長さをフレームサイズにしてあるので，syパケットは必ず
  // _buf, _midのいずれかに記録される．raw1394_loop_iterate() は，この
  // ハンドラをinterval回呼ぶ．特にintervalが1フレームあたりのパケット数の
  // 約数でない場合は，_bufにsyパケットを読み込んだ後にフレーム全体の受信が
  // 完了した後に，さらに_midに次のフレームのsyパケットが読み込まれる．
    if (sy)				// フレームの先頭パケットならば...
    {
	node->_arrivaltime_next = node->cycleToLocaltime(cycle);
	
	if (node->_current != node->_mid)	// _midが読み込み先でなければ
	{
	    node->_current = node->_buf;	// _bufに読み込む
	    node->_arrivaltime = node->_arrivaltime_next;
	}
#  if defined(DEBUG)
	u_int64_t
	    timestamp = node->cycletimeToLocaltime(ntohl(*(u_int32_t*)data));
	cerr << " (sy: current = " << node->_current - node->_buf
	     << ", cycle = " << cycle << ')'
	     << " diff: ";
	print_time(cerr, node->_arrivaltime_next - timestamp);
	cerr << ", arrivaltime: ";
	print_time(cerr, node->_arrivaltime_next);
	cerr << ", timestamp: ";
	print_time(cerr, timestamp);
	cerr << endl;
#  endif	
    }
#  if DEBUG==2
    else
    {
	cerr << " (    current = ";
	if (node->_current != 0)
	    cerr << node->_current - node->_buf;
	else
	    cerr << "NULL";
	cerr << ", cycle = " << cycle << ')';
	u_int64_t	captime = node->cycleToLocaltime(cycle);
	cerr << " captime: ";
	print_time(cerr, captime);
	cerr << endl;
    }
#  endif
    if (node->_current + len <= node->_end)	// overrunが生じなければ...
    {
	memcpy(node->_current, data, len);	// パケットを読み込む
	node->_current += len;			// 次の読み込み位置に進める
    }
    
    return RAW1394_ISO_OK;
}
#endif
 
}
