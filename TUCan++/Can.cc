/*
 *  $Id: Can.cc,v 1.2 2002-07-25 02:38:01 ueshiba Exp $
 */
/*!
  \mainpage	libTUCan++ - CANおよびManusマニピュレータコントローラ
  \anchor	libTUCan

  libTUCan++は，CAN(Control Area Network)デバイスおよび
  <a href="http://www.exactdynamics.nl/">Exact Dynamics社</a>製のManus
  マニピュレータのためのコントローラライブラリである．

  libTUCan++は，フリーのCANボード用デバイスドライバである
  <a href="http://www.port.de/engl/canprod/content/sw_linux.html">can4linux
  </a>を利用しているので，動作環境もこれに準じる．LINUXのkernel-2.2.18,
  kernel-2.4.18での動作実績がある．おそらくkernel-2.2, kernel-2.4共に
  minor versionへの依存性はないものと推測される．ただし，kernel-2.4.10
  以降では，can4linuxに若干手を加える必要がある(\ref can4linux)．

  \section can4linux can4linuxのインストール
  
  can4linuxを
  <a href="http://www.port.de/engl/canprod/content/sw_linux.html>
  ダウンロード</a>して展開した後，以下の作業を行う．

  まず，以下のようにしてcan4linuxドライバをコンパイルする．

  \verbatim
  % cd can4linux
  % make \endverbatim
  なお，src/can_82c200funcs.c, src/can_core.c, src/can_sysctl.c
  に手を加えてある．

  次に，デバイスノードを作成する．

  \verbatim
  # mknod -m666 /dev/can0 c 91 0 \endverbatim

  さらに，can4linuxドライバ用のconfiguration fileとして，
  /etc/can.confに以下の内容(CANカードによって異なる．以下はManus
  マニピュレータ付属のATカードの例)のファイルを作る．

  \verbatim
  # Channel 0

  Chipset_0=82c200	# Phillips 82c200 can controller.
  Base_0=0x0300		# Base I/O port address of this board is 0x300.
  Irq_0=5		# Must be equal to the jumper pin setting on the board.
  Baud_0=250		# Transmission speed of Manus manipulator is 250 baud.
  AccMask_0=0xffff
  Timeout_0=100
  Outc_0=0

  IOModel_0=p		# Must be "p"(port I/O) but "m"(memory-mapped).
  TxSpeed_0=s		# "s"(slow) is safer than "f"(fast).\endverbatim

  すると，以下の手続きでドライバのロード／アンロードができるようになる．
  \verbatim
  # make load			// ドライバのロード. 
  # make unload			// ドライバのアンロード. 
  % grep . /proc/sys/Can/*	// 正しくドライバがロードされているか確認.\endverbatim

  PCの起動時にドライバが自動的にロードされるためには，まず
  \verbatim
  # mkdir /lib/modules/`uname -r`/kernel/drivers/can
  # cp Can.o /lib/modules/`uname -r`/kernel/drivers/can
  # cp utils/cansetup /etc\endverbatim
  を行い，/etc/rc.d/rc.modulesに以下の内容を追加する．

  \verbatim
  #!/bin/sh

  canpath=/lib/modules/`uname -r`/kernel/drivers/can

  if [ -f "$canpath"/Can.o -a -x /etc/cansetup ]; then
    /sbin/insmod $canpath/Can.o
    /etc/cansetup
  fi\endverbatim

  さらに，
  \verbatim
  # cd /etc
  # ln -s rc.d/rc.modules \endverbatim
  として，/etcからこのファイルが見えるようにしておく(RedHat7.x系のための措置)．

  \section example プログラム例 - manustest

  完全なプログラム例として，キーボードインターフェースによるManus
  マニピュレータのコントロールプログラム: \ref manustestがある．
*/
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <errno.h>
#include <string>
#include "TU/Can++.h"

namespace TU
{
/************************************************************************
*  class CanPort							*
************************************************************************/
//! CANノードを生成する
/*!
  \param dev			このノードのデバイス名(ex. /dev/can0)．
  \exception std::runtime_error	デバイスのopenに失敗．
*/
Can::Can(const char* dev)
    :_fd(open(dev, O_RDWR))
{
    using namespace	std;
    if (_fd < 0)
	throw runtime_error(string("TU::Can::Can: cannot open device!! ") + strerror(errno));
    _msg.flags	= 0;
    _msg.cob	= 0;
    _msg.id	= 0;
    _msg.length	= 0;
}

//! CANノードを破壊する
Can::~Can()
{
    if (_fd >= 0)
	close(_fd);
}

//! メッセージが到着している場合は読み込み，そうでなければ直ちに戻る
/*!
  \return			読み込んだメッセージのID．到着していない
				場合は0xffffffff．
  \exception std::runtime_error	読み込みに失敗．
*/
u_long
Can::nreceive()
{
    using namespace	std;
    Receive_par_t	rx;
    rx.Rx     = &_msg;
    rx.error  = 0;
    rx.retval = 0;
    if (::ioctl(_fd, RECEIVE, &rx) < 0)
	throw runtime_error(string("ioctl (TU::Can::nreceive) ")
			    + strerror(errno));
    return (rx.retval != 0 ? id() : ~0);
}

//! メッセージが到着するまで待って読み込む
/*!
  \return			読み込んだメッセージのID．
  \exception std::runtime_error	読み込みに失敗．
*/
u_long
Can::receive()
{
    using namespace	std;
    Receive_par_t	rx;
    rx.Rx     = &_msg;
    rx.error  = 0;
    rx.retval = 0;
    do
    {
	if (::ioctl(_fd, RECEIVE, &rx) < 0)
	    throw runtime_error(string("ioctl (TU::Can::receive) ")
				+ strerror(errno));
    } while (rx.retval == 0);
    return id();
}

//! メッセージに含まれるデータを返す
/*!
  \param i	データへのindex．\f$0 \leq i < \f$nbytes()でなければならない．
  \return	現在読み込まれているメッセージのi番目のデータ．
  \exception std::out_of_range	iが範囲外．
*/
u_char
Can::get(u_int i) const
{
    if (i < 0 || i >= _msg.length)
	throw std::out_of_range("TU::Can::get: invalid index!!");
    return _msg.data[i];
}

//! 通信速度を設定する
/*!
  \param baud			設定したいbaud rate．
  \return			このCANノードオブジェクト．
  \exception std::runtime_error	ioctlに失敗．
*/
Can&
Can::setBaud(Baud baud)
{
    using namespace		std;
    volatile Command_par_t	cmd;
    cmd.cmd = CMD_STOP;
    if (::ioctl(_fd, COMMAND, &cmd) == -1)
	throw runtime_error(string("ioctl (TU::Can::setBaud) ")
			    + strerror(errno));
    
    Config_par_t		cfg;
    cfg.target = CONF_TIMING;
    cfg.val1   = baud;
    if (::ioctl(_fd, CONFIG, &cfg) == -1)
	throw runtime_error(string("ioctl (TU::Can::setBaud) ")
			    + strerror(errno));
    
    cmd.cmd = CMD_START;
    if (::ioctl(_fd, COMMAND, &cmd) == -1)
	throw runtime_error(string("ioctl (TU::Can::setBaud) ")
			    + strerror(errno));

    return *this;
}

//! 送信メッセージのIDを設定する
/*!
  これまでメッセージ中にあったデータは破棄される．
  \param id	設定したいメッセージID．
  \return	このCANノードオブジェクト．
*/
Can&
Can::setId(u_long id)
{
    _msg.id	= id;
    _msg.length	= 0;
    return *this;
}

//! 送信メッセージにデータを格納する
/*!
  put()を複数回行うと，データはその順番に格納される．ただし，1つの
  メッセージにデータは8つまでしか格納できない．
  \param c			格納したいデータ．
  \return			このCANノードオブジェクト．
  \exception std::out_of_range	既にメッセージバッファが一杯．
*/
Can&
Can::put(u_char c)
{
    if (_msg.length >= CAN_MSG_LENGTH)
	throw std::out_of_range("TU::Can::put: message buffer is full!!");
    _msg.data[_msg.length++] = c;
    return *this;
}

//! メッセージを送信する
/*!
  \return			このCANノードオブジェクト．
  \exception std::runtime_error	送信に失敗．
*/
const Can&
Can::send() const
{
    using namespace	std;
  /*    if (::write(_fd, &_msg, 1) < 0)
	throw runtime_error(string("write (TU::Can::send) ") + strerror(errno)); */
    Send_par_t	tx;
    tx.Tx     = (canmsg_t*)&_msg;
    tx.error  = 0;
    tx.retval = 0;
    if (::ioctl(_fd, SEND, &tx) < 0)
	throw runtime_error(string("ioctl (TU::Can::send) ")
			    + strerror(errno));
    return *this;
}
 
}
