/*
 *  $Id: Ieee1394Node.cc,v 1.3 2002-12-09 07:47:50 ueshiba Exp $
 */
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdexcept>
#include <string>
#include "TU/Ieee1394++.h"

namespace TU
{
/************************************************************************
*  class Ieee1394Node							*
************************************************************************/
//! IEEE1394ノードオブジェクトを生成する
/*!
  \param port		このノードが接続されているポート．
  \param unit_spec_ID	このノードの種類を示すID(ex. IEEE1394デジタルカメラ
		       であれば，0x00a02d).
  \param channel	isochronous転送用のチャネル番号(\f$0 \leq
			\f$channel\f$ < 64\f$)
  \param sync_tag	1まとまりのデータを複数のパケットに分割して
			isochronousモードで受信する際に，最初のパケットに
			同期用のtagがついている場合は1を指定．そうでなけれ
			ば0を指定．
  \param flags		video1394のフラグ．VIDEO1394_SYNC_FRAMES, 
			VIDEO1394_INCLUDE_ISO_HEADERS,
			VIDEO1394_VARIABLE_PACKET_SIZEの組合わせ．
  \param uniqId		個々の機器固有の64bit ID．同一のIEEE1394 busに
			同一のunit_spec_IDを持つ複数の機器が接続されて
			いる場合，これによって同定を行う．
			0が与えられると，指定されたunit_spec_IDを持ち
			まだ#Ieee1394Nodeオブジェクトを割り当てられて
			いない機器のうち，一番最初にみつかったものがこの
			オブジェクトと結びつけられる．
*/
Ieee1394Node::Ieee1394Node(Ieee1394Port& port, u_int unit_spec_ID,
			   u_int channel, int sync_tag, int flags,
			   u_int64 uniqId)
    :_port(port), _nodeId(0), _mmap(), _buf_size(0),
     _buf(0), _current(0), _nready(0)
{
  // Find a node yet registered to the port and satisfying the specification.
    u_int	i, nnodes = _port.nnodes();
    for (i = 0; i < nnodes; ++i)
    {
	_nodeId = (i | 0xffc0);		// node on local bus

	if (_nodeId != _port.nodeId()			     &&
	    !_port.isRegisteredNode(*this)		     &&
	    readValueFromUnitDirectory(0x12) == unit_spec_ID &&
	    (uniqId == 0 || globalUniqueId() == uniqId))
	    break;
    }
    if (i == nnodes)
	throw std::runtime_error("TU::Ieee1394Node::Ieee1394Node: node with specified unit_spec_ID (and global_unique_ID) not found!!");

    _mmap.channel     = channel;
    _mmap.sync_tag    = sync_tag;
    _mmap.nb_buffers  = 0;
    _mmap.buf_size    = 0;
    _mmap.packet_size = 0;
    _mmap.fps	      = 0;
    _mmap.flags	      = flags;

  // Register this instance to the port.
    _port.registerNode(*this);
}
	     
//! IEEE1394ノードオブジェクトを破壊する
Ieee1394Node::~Ieee1394Node()
{
    unmapListenBuffer();
    _port.unregisterNode(*this);
}

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
  \param key	keyすなわち4byteの並びのMSB側8bit．
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
  \param key	keyすなわち4byteの並びのMSB側8bit．
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

//! ノード内の指定されたアドレスから4byteの値を読み出す
/*!
  \param addr	個々のノード内の絶対アドレス．
 */
quadlet_t
Ieee1394Node::readQuadlet(nodeaddr_t addr) const
{
    using namespace	std;

    quadlet_t	quad;
    if (raw1394_read(_port.handle(), _nodeId, addr, 4, &quad) < 0)
	throw runtime_error(string("TU::Ieee1394Node::readQuadlet: failed to read from port!! ") + strerror(errno));
    if (_port.delay() != 0)
	::usleep(_port.delay());
    return quadlet_t(ntohl(quad));
}

//! ノード内の指定されたアドレスに4byteの値を書き込む
/*!
  \param addr	個々のノード内の絶対アドレス．
  \param quad	書き込む4byteの値．
 */
void
Ieee1394Node::writeQuadlet(nodeaddr_t addr, quadlet_t quad)
{
    using namespace	std;

    quad = htonl(quad);
    if (raw1394_write(_port.handle(), _nodeId, addr, 4, &quad) < 0)
	throw runtime_error(string("TU::Ieee1394Node::writeQuadlet: failed to write to port!! ") + strerror(errno));
    if (_port.delay() != 0)
	::usleep(_port.delay());
}

//! isochronous受信用のバッファを割り当てる
/*!
  \param packet_size	受信するパケット1つあたりのサイズ(単位: byte).
  \param buf_size	バッファ1つあたりのサイズ(単位: byte)．
  \param nb_buffers	割り当てるバッファ数．
 */
void
Ieee1394Node::mapListenBuffer(size_t packet_size,
			      size_t buf_size, u_int nb_buffers)
{
    using namespace	std;
    
  // Unmap previously mapped buffer and unlisten the channel.
    unmapListenBuffer();

  // Change buffer size and listen to the channel.
  //   *Caution: _mmap.buf_size may be changed by VIDEO1394_LISTEN_CHANNEL.
    _mmap.nb_buffers  = nb_buffers;
    _mmap.buf_size    = _buf_size = buf_size;
    _mmap.packet_size = packet_size;
    if (ioctl(_port.fd(), VIDEO1394_LISTEN_CHANNEL, &_mmap) < 0)
	throw runtime_error(string("TU::Ieee1394Node::mapListenBuffer: VIDEO1394_LISTEN_CHANNEL failed!! ") + strerror(errno));
    for (int i = 0; i < _mmap.nb_buffers; ++i)
    {
	video1394_wait	wait;
	wait.channel = _mmap.channel;
	wait.buffer  = i;
	if (ioctl(_port.fd(), VIDEO1394_LISTEN_QUEUE_BUFFER, &wait) < 0)
	    throw runtime_error(string("Ieee1394Node::mapListenBuffer: VIDEO1394_LISTEN_QUEUE_BUFFER failed!! ") + strerror(errno));
    }

  // Reset buffer status and re-map new buffer.
    if ((_buf = (u_char*)mmap(0, _mmap.nb_buffers * _mmap.buf_size,
			      PROT_READ, MAP_SHARED, _port.fd(), 0))
	== (u_char*)-1)
    {
	_buf = 0;
	throw runtime_error(string("Ieee1394Node::mapListenBuffer: mmap failed!! ") + strerror(errno));
    }

    usleep(100000);
}

//! isochronousデータが受信されるのを待つ
/*!
  実際にデータが受信されるまで，本関数は呼び出し側に制御を返さない．
  \return	データの入ったバッファの先頭アドレス．データのサイズは
		getBufferSize()で知ることができる．
 */
const u_char*
Ieee1394Node::waitListenBuffer()
{
    using namespace	std;

    if (_nready == 0)
    {
	video1394_wait	wait;
	wait.channel = _mmap.channel;
	wait.buffer  = _current;
	if (ioctl(_port.fd(), VIDEO1394_LISTEN_WAIT_BUFFER, &wait) < 0)
	    throw runtime_error(string("TU::Ieee1394Node::waitListenBuffer: VIDEO1394_LISTEN_WAIT_BUFFER failed!! ") + strerror(errno));
	_nready = 1 + wait.buffer;  // current and subsequent ready buffers.
    }
    return _buf + _current * _mmap.buf_size;
}

//! データ受信済みのバッファを再びキューイングして次の受信データに備える
void
Ieee1394Node::requeueListenBuffer()
{
    using namespace	std;

    video1394_wait	wait;
    wait.channel = _mmap.channel;
    wait.buffer	 = _current;
    if (ioctl(_port.fd(), VIDEO1394_LISTEN_QUEUE_BUFFER, &wait) < 0)
	throw runtime_error(string("TU::Ieee1394Node::requeueListenBuffer: VIDEO1394_LISTEN_QUEUE_BUFFER failed!! ") + strerror(errno));
    ++_current %= _mmap.nb_buffers;	// next buffer.
    --_nready;				// # of ready buffers.
}

//! すべての受信用バッファの内容を空にする
void
Ieee1394Node::flushListenBuffer()
{
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
	    if (ioctl(_port.fd(), VIDEO1394_LISTEN_POLL_BUFFER, &wait) < 0)
		break;
	    _nready = 1 + wait.buffer;
	    requeueListenBuffer();
	    }*/
  // "_nready" must be 0 here(no available buffers).
}

//! ノードに割り当てたすべての受信用バッファを廃棄する
void
Ieee1394Node::unmapListenBuffer()
{
    using namespace	std;

    if (_buf != 0)
    {
	munmap(_buf, _mmap.nb_buffers * _mmap.buf_size);
	_buf = 0;				// Reset buffer status.
	_buf_size = _current = _nready = 0;	// ibid.
	if (ioctl(_port.fd(), VIDEO1394_UNLISTEN_CHANNEL, &_mmap.channel) < 0)
	    throw runtime_error(string("TU::Ieee1394Node::unmapListenBuffer: VIDEO1394_UNLISTEN_CHANNEL failed!! ") + strerror(errno));
    }
}
 
}
