/*
 *  $Id: FireWireNode.cc 1655 2014-10-03 01:37:50Z ueshiba $
 */
#include "FireWireNode_.h"
#include <algorithm>		// for std::copy()
#include <cstring>		// for strerror()

namespace TU
{
/************************************************************************
*  class FireWireNode							*
************************************************************************/
//! FireWireノードオブジェクトを生成する
/*!
  \param uniqId		個々の機器固有の64bit ID. 同一のFireWire busに
			同一のunit_spec_IDを持つ複数の機器が接続されて
			いる場合, これによって同定を行う. 
			0が与えられると, 指定されたunit_spec_IDを持ち
			まだ#FireWireNodeオブジェクトを割り当てられて
			いない機器のうち, 一番最初にみつかったものがこの
			オブジェクトと結びつけられる. 
  \param unit_spec_ID	このノードの種類を示すID(ex. FireWireデジタルカメラ
			であれば, 0x00a02d)
*/
FireWireNode::FireWireNode(u_int unit_spec_ID, uint64_t uniqId)
#if defined(__APPLE__)
    :_handle(raw1394_new_handle(unit_spec_ID, uniqId)),
     _nodeId(raw1394_get_remote_id(_handle)),
#else
    :_handle(raw1394_new_handle()),
     _nodeId(0),
#endif
     _channel(0), _buffers(), _ready()
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

    addToList();	// コンストラクタが成功したので、ノードリストに登録
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

IIDCNode::nodeid_t
FireWireNode::nodeId() const
{
    return _nodeId;
}
    
IIDCNode::quadlet_t
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

    _buffers.resize(nb_buffers);
    _ready.initialize(_buffers.begin(), _buffers.end());

  // バッファ1つ分のデータを転送するために必要なパケット数
    const u_int	npackets = (buf_size - 1) / packet_size + 1;

  // buf_sizeをpacket_sizeの整数倍にしてからmapする.
    buf_size = packet_size * npackets;
    for (auto& buffer : _buffers)
	buffer.map(this, buf_size);

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
    if (!_buffers.empty())
    {
	raw1394_iso_stop(_handle);
	raw1394_iso_shutdown(_handle);

	_buffers.clear();
	_ready.initialize(_buffers.begin(), _buffers.end());
    }
}

const void*
FireWireNode::waitListenBuffer()
{
    while (_ready.empty())		// ready queueにバッファが登録されるまで
	raw1394_loop_iterate(_handle);	// パケットを受信する．

    return _ready.front().data();	// 最古のバッファのデータを返す
}

void
FireWireNode::requeueListenBuffer()
{
    if (!_ready.empty())
    {
	_ready.front().clear();		// 最古のバッファを空にして
	_ready.pop();			// ready queueから取り出す
    }
}

void
FireWireNode::flushListenBuffer()
{
    while (!_ready.empty())
    {
	_ready.front().clear();		// バッファを空にして
	_ready.pop();			// ready queueから取り出す
    }
}

uint32_t
FireWireNode::getCycletime(uint64_t& localtime) const
{
    uint32_t	cycletime;
    raw1394_read_cycle_timer(_handle, &cycletime, &localtime);

    return cycletime;
}
    
//! isochronous受信用バッファにパケットデータを転送する
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
		      u_char* data, u_int len, u_char channel,
		      u_char tag, u_char sy, u_int cycle, u_int dropped)
{
    const auto	node = static_cast<FireWireNode*>(
			   raw1394_get_userdata(handle));
    auto&	buffer = node->_ready.back();	// 現在受信中のバッファ
    
    if (sy)					// 新しいフレームが来たら...
	buffer.clear();				// これまでの内容を捨てる

  // 受信データをバッファにコピー
    if (buffer.receive(data, len))		// バッファが一杯になったら...
	node->_ready.push();			// ready queueに移す

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

	    if ((_nodeId      != localId)		    &&
		(unitSpecId() == unit_spec_ID)		    &&
		(uniqId == 0 || globalUniqueId() == uniqId) &&
		!inList())
	    {
		for (int channel = 0; channel < 64; ++channel)
		{
		    if (raw1394_channel_modify(_handle, channel,
					       RAW1394_MODIFY_ALLOC) == 0)
		    {
			return channel;
		    }
		}
	    }
	}
	raw1394_destroy_handle(_handle);
	_handle = nullptr;
    }

    throw std::runtime_error("No device with specified unit_spec_ID and globalUniqId found!!");

    return 0;
}
#endif

/************************************************************************
*  class FireWireNode::Buffer						*
************************************************************************/
FireWireNode::Buffer::Buffer()
    :_node(nullptr), _data(), _ndata(0)
{
}

void
FireWireNode::Buffer::map(FireWireNode* node, u_int size)
{
    _node = node;
    _data.resize(size);
    _ndata = 0;
}
    
bool
FireWireNode::Buffer::receive(u_char* data, u_int len)
{
    const auto	rest = _data.size() - _ndata;

    if (len < rest)
    {
	std::copy_n(data, len, _data.begin() + _ndata);
	_ndata += len;
	return false;
    }
    else
    {
	std::copy_n(data, rest, _data.begin() + _ndata);
	_ndata += rest;
	return true;
    }
}
    
}
