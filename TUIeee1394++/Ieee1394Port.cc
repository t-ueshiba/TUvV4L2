/*
 *  $Id: Ieee1394Port.cc,v 1.1.1.1 2002-07-25 02:14:15 ueshiba Exp $
 */
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdexcept>
#include <string>
#include "TU/Ieee1394++.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static raw1394handle_t
raw1394_get_handle_and_check()
{
    using namespace	std;

    raw1394handle_t	handle = raw1394_new_handle();
    if (handle == NULL)
	throw runtime_error(string("TU::raw1394_get_handle_and_check: failed to get raw1394handle!! ") + strerror(errno));
    return handle;
}

static int
video1394_get_fd_and_check(int port_number)
{
    using namespace	std;

    char	dev[] = "/dev/video1394.x";
    dev[strlen(dev)-1] = '0' + port_number;
    int		fd = open(dev, O_RDWR);
    if (fd < 0)
	throw runtime_error(string("TU::raw1394_get_fd_and_check: failed to open video1394!! ") + strerror(errno));
    return fd;
}

/************************************************************************
*  class Ieee1394Port							*
************************************************************************/
//! IEEE1394ポートオブジェクトを生成する
/*!
  ポートとは，一般にはPCにインストールされているIEEE1394のカードに相当する．
  カードが複数挿入されていれば，port_numberに異なる値を与えることにより
  複数のポートオブジェクトを生成できる．
  \param port_number	ポートの通し番号(0以上の整数)
  \param delay		IEEE1394カードの種類によっては，レジスタの読み書き
			(Ieee1394Node::readQuadlet(),
			Ieee1394Node::writeQuadlet())時に遅延を入れないと
			動作しないことがある．この遅延量をmicro second単位
			で指定する．(例: メルコのIFC-ILP3では1, DragonFly
			付属のボードでは0)
*/
Ieee1394Port::Ieee1394Port(int port_number, u_int delay)
    :_handle(raw1394_get_handle_and_check()),
     _fd(video1394_get_fd_and_check(port_number)), _delay(delay)
{
    using namespace	std;
    
    int	nports = raw1394_get_port_info(_handle, NULL, 0);
    if (nports < 0)
	throw runtime_error(string("TU::Ieee1394Port::Ieee1394Port: failed to get port info!! ") + strerror(errno));
    if (port_number >= nports)
	throw invalid_argument("TU::Ieee1394Port::Ieee1394Port: specified port not found!!");
    if (raw1394_set_port(_handle, port_number) < 0)
	throw runtime_error(string("TU::Ieee1394Port::Ieee1394Port: failed to set port!! ") + strerror(errno));

  // Clear pointer table for nodes.
    for (u_int i = 0; i < sizeof(_node)/sizeof(_node[0]); ++i)
	_node[i] = 0;
}

//! IEEE1394ポートオブジェクトを破壊する
Ieee1394Port::~Ieee1394Port()
{
    close(_fd);
    raw1394_destroy_handle(_handle);
}

void
Ieee1394Port::registerNode(const Ieee1394Node& node)
{
    _node[node.nodeId() & 0x3f] = &node;
}

void
Ieee1394Port::unregisterNode(const Ieee1394Node& node)
{
    _node[node.nodeId() & 0x3f] = 0;
}

bool
Ieee1394Port::isRegisteredNode(const Ieee1394Node& node) const
{
    return (_node[node.nodeId() & 0x3f] != 0);
}
 
}
