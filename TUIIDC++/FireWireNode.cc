/*
 *  $Id: FireWireNode.cc 1655 2014-10-03 01:37:50Z ueshiba $
 */
#include "FireWireNode_.h"
#include <cstring>		// for strerror()

namespace TU
{
/************************************************************************
*  class FireWireNode							*
************************************************************************/
//! FireWireノードオブジェクトを生成する
/*!
  \param unit_spec_ID	このノードの種類を示すID(ex. FireWireデジタルカメラ
			であれば, 0x00a02d)
  \param uniqId		個々の機器固有の64bit ID. 同一のFireWire busに
			同一のunit_spec_IDを持つ複数の機器が接続されて
			いる場合, これによって同定を行う. 
			0が与えられると, 指定されたunit_spec_IDを持ち
			まだ#FireWireNodeオブジェクトを割り当てられて
			いない機器のうち, 一番最初にみつかったものがこの
			オブジェクトと結びつけられる. 
*/
FireWireNode::FireWireNode(u_int unit_spec_ID, uint64_t uniqId)
#if defined(__APPLE__)
    :_handle(raw1394_new_handle(unit_spec_ID, uniqId)),
     _nodeId(raw1394_get_remote_id(_handle)),
#else
    :_handle(raw1394_new_handle()),
     _nodeId(0),
#endif
     _channel(0), _buf_size(0), _buf(nullptr), _len(0), _p(nullptr)
{
#if !defined(__APPLE__)
    try
    {
	check(!_handle,
	      "FireWireNode::FireWireNode: failed to get raw1394handle!!");
	_channel = setHandle(unit_spec_ID, uniqId);
    }
    catch (const std::runtime_error& err)
    {
	if (_handle)
	{
	    raw1394_destroy_handle(_handle);
	    _handle = nullptr;
	}
	throw err;
    }
#endif	// !__APPLE__
    raw1394_set_userdata(_handle, this);
}
	     
//! FireWireノードオブジェクトを破壊する
FireWireNode::~FireWireNode()
{
    unmapListenBuffer();
#if !defined(__APPLE__)
    raw1394_channel_modify(_handle, _channel, RAW1394_MODIFY_FREE);
#endif
    raw1394_destroy_handle(_handle);
}

nodeid_t
FireWireNode::nodeId() const
{
    return _nodeId;
}
    
quadlet_t
FireWireNode::readQuadlet(nodeaddr_t addr) const
{
    quadlet_t	quad;
    check(raw1394_read(_handle, _nodeId, addr, 4, &quad),
	  "FireWireNode::readQuadlet: failed to read from port!!");
    return quadlet_t(ntohl(quad));
}

void
FireWireNode::writeQuadlet(nodeaddr_t addr, quadlet_t quad)
{
    quad = htonl(quad);
    check(raw1394_write(_handle, _nodeId, addr, 4, &quad),
	  "TU::FireWireNode::writeQuadlet: failed to write to port!!");
}

u_char
FireWireNode::mapListenBuffer(u_int packet_size,
			      u_int buf_size, u_int nb_buffers)
{
    unmapListenBuffer();

  // バッファ1つ分のデータを転送するために必要なパケット数
    const u_int	npackets = (buf_size - 1) / packet_size + 1;

  // buf_sizeをpacket_sizeの整数倍にしてからmapする.
    _buf_size = packet_size * npackets;
    _buf      = new u_char[_buf_size];
    _len      = 0;
    _p	      = _buf;
    
  // raw1394_loop_iterate()は，interval個のパケットを受信するたびにユーザ側に
  // 制御を返す．libraw1394ではこのデフォルト値はパケット数の1/4である．ただし，
  // 512パケットを越える値を指定すると，raw1394_loop_iterate()から帰ってこなく
  // なるようである．
#if defined(__APPLE__)
    const u_int	interval = npackets;
#else
    const u_int	interval  = std::min(npackets/4, 512U);
  /*
    const int	speed	  = raw1394_get_speed(_handle, _nodeId);
    u_int	bandwidth = packet_size/4 + 3;
    if (speed > RAW1394_ISO_SPEED_400)
	bandwidth >>= (speed - RAW1394_ISO_SPEED_400);
    else
	bandwidth <<= (RAW1394_ISO_SPEED_400 - speed);
    check(raw1394_bandwidth_modify(_handle, bandwidth, RAW1394_MODIFY_ALLOC),
	  "FireWireNode::mapListenBuffer: failed to allocate bandwidth!!");
  */
#endif
    check(raw1394_iso_recv_init(_handle, receive,
				nb_buffers * npackets, packet_size, _channel,
				RAW1394_DMA_BUFFERFILL, interval),
	  "FireWireNode::mapListenBuffer: failed to initialize iso reception!!");
    check(raw1394_iso_recv_start(_handle, -1, -1, 0),
	  "FireWireNode::mapListenBuffer: failed to start iso reception!!");

    return _channel;
}

void
FireWireNode::unmapListenBuffer()
{
    if (_buf)
    {
	raw1394_iso_stop(_handle);
	raw1394_iso_shutdown(_handle);
      //raw1394_bandwidth_modify(_handle, 4915, RAW1394_MODIFY_FREE);

	delete [] _buf;
	_buf_size = 0;
	_buf	  = nullptr;
	_len	  = 0;
	_p	  = nullptr;
    }
}

const u_char*
FireWireNode::waitListenBuffer()
{
    while (_len < _buf_size)
	raw1394_loop_iterate(_handle);	// パケットを受信する．

    return _buf;
}

void
FireWireNode::requeueListenBuffer()
{
    _len = 0;
    _p   = _buf;
}

void
FireWireNode::flushListenBuffer()
{
}

uint32_t
FireWireNode::getCycleTime(uint64_t& localtime) const
{
    uint32_t	cycletime;
    raw1394_read_cycle_timer(_handle, &cycletime, &localtime);

    return cycletime;
}
    
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
FireWireNode::receive(raw1394handle_t handle,
		      u_char* data, u_int len,
		      u_char channel, u_char tag, u_char sy,
		      u_int cycle, u_int dropped)
{
    if (dropped)
	std::cerr << "recieve: dropped = " << dropped << std::endl;
    
    const auto	node = reinterpret_cast<FireWireNode*>(
			   raw1394_get_userdata(handle));

    if (sy)
    {
	node->_len = 0;
	node->_p   = node->_buf;
    }

    if (node->_p + len <= node->_buf + node->_buf_size)
    {
	memcpy(node->_p, data, len);
	node->_p += len;
    }
    node->_len += len;

    return RAW1394_ISO_OK;
}

void
FireWireNode::check(bool err, const std::string& msg)
{
    if (err)
	throw std::runtime_error(msg + ' ' + strerror(errno));
}
    
#if !defined(__APPLE__)
u_char
FireWireNode::setHandle(uint32_t unit_spec_ID, uint64_t uniqId)
{
  // Get the number of ports.
    const int	nports = raw1394_get_port_info(_handle, NULL, 0);
    check(nports < 0, "FireWireNode::FireWireNode: failed to get port info!!");
    raw1394_destroy_handle(_handle);
    _handle = nullptr;
    
  // Find the specified node yet registered.
    for (int i = 0; i < nports; ++i)		// for each port...
    {
	check(!(_handle = raw1394_new_handle_on_port(i)),
	      "FireWireNode::FireWireNode: failed to get raw1394handle and set it to the port!!");
	nodeid_t	localId = raw1394_get_local_id(_handle);
	const int	nnodes  = raw1394_get_nodecount(_handle);
	for (int j = 0; j < nnodes; ++j)	// for each node....
	{
	    _nodeId = (j | 0xffc0);

	    if (_nodeId != localId && unitSpecId() == unit_spec_ID &&
		(uniqId == 0 || globalUniqueId() == uniqId))
		for (int channel = 0; channel < 64; ++channel)
		{
		    if (raw1394_channel_modify(_handle, channel,
					       RAW1394_MODIFY_ALLOC) == 0)
			return channel;
		}
	}
	raw1394_destroy_handle(_handle);
	_handle = nullptr;
    }

    throw std::runtime_error("No device with specified unit_spec_ID and globalUniqId found!!");
    return 0;
}
#endif
    
}
