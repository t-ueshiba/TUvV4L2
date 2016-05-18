/*
 *  $Id: FireWireNode_.h 1742 2014-11-14 09:48:38Z ueshiba $
 */
#ifndef __TU_FIREWIRENode__H
#define __TU_FIREWIRENode__H

//#define USE_VIDEO1394

#include "TU/IIDC++.h"
#if defined(HAVE_CONFIG_H)
#  include <config.h>
#endif
#include <libraw1394/raw1394.h>
#if !defined(__APPLE__)
#  include <map>
#  if defined(USE_VIDEO1394)
#    define __user
#    include <video1394.h>
#  endif
#endif

/*!
  \namespace	TU
  \brief	本ライブラリで定義されたクラスおよび関数を納める名前空間
 */
namespace TU
{
/************************************************************************
*  class FireWireNode							*
************************************************************************/
//! FireWireのノードを表すクラス
/*!
  一般には, より具体的な機能を持ったノード(ex. デジタルカメラ)を表す
  クラスの基底クラスとして用いられる. 
*/
class FireWireNode : public IIDCNode
{
#if !defined(__APPLE__)
  private:
    class Port
    {
      public:
	Port(int portNumber)				;
	~Port()						;

#  if defined(USE_VIDEO1394)
	int	fd()				const	{return _fd;}
#  endif
	int	portNumber()			const	{return _portNumber;}
	
	u_char	registerNode(const FireWireNode& node)		;
	bool	unregisterNode(const FireWireNode& node)	;
	bool	isRegisteredNode(const FireWireNode& node) const;
    
      private:
#  if defined(USE_VIDEO1394)
	const int	_fd;		// file desc. for video1394 device.
#  endif
	const int	_portNumber;
	uint64_t	_nodes;		// a bitmap for the registered nodes
    };
#endif	// !__APPLE__
  public:
  //! isochronous転送の速度
    enum Speed
    {
	SPD_100M	= 0,			//!< 100Mbps
	SPD_200M	= 1,			//!< 200Mbps
	SPD_400M	= 2,			//!< 400Mbps
	SPD_800M	= 3,			//!< 800Mbps
	SPD_1_6G	= 4,			//!< 1.6Gbps
	SPD_3_2G	= 5			//!< 3.2Gbps
    };

  public:
    FireWireNode(uint32_t unit_spec_ID, uint64_t uniqId, u_int delay
#if defined(USE_VIDEO1394)
		 , int sync_tag, int flag
#endif
		 )							;
    ~FireWireNode()							;
    FireWireNode(const FireWireNode&)				= delete;
    FireWireNode&	operator =(const FireWireNode&)		= delete;

  //! このノードのisochronous受信用バッファにフレームの先頭パケットが到着した時刻を返す
  /*!
    \return	受信用バッファにフレームの先頭パケットが到着した時刻
   */
    uint64_t		arrivaltime()		const	{return _arrivaltime;}

  //! このノードに割り当てられたisochronousチャンネルを返す
  /*!
    \return	isochronousチャンネル
   */
#if defined(USE_VIDEO1394)
    u_char		channel()		const	{return _mmap.channel;}
#else
    u_char		channel()		const	{return _channel;}
#endif

  //! このノードに割り当てられたasynchronous通信の遅延量を返す
  /*!
    \return	遅延量（単位：micro seconds）
   */
    u_int		delay()			const	{return _delay;}
    
    virtual nodeid_t	nodeId()				const	;
#if defined(__APPLE__)
    nodeaddr_t		commandRegisterBase()			const	;
#endif
    virtual quadlet_t	readQuadlet(nodeaddr_t addr)		const	;
    virtual void	writeQuadlet(nodeaddr_t addr, quadlet_t quad)	;
    virtual u_char	mapListenBuffer(u_int packet_size,
					u_int buf_size,
					u_int nb_buffers)		;
    virtual void	unmapListenBuffer()				;
    virtual const u_char*
			waitListenBuffer()				;
    virtual void	requeueListenBuffer()				;
    virtual void	flushListenBuffer()				;
    uint64_t		cycletimeToLocaltime(uint32_t cycletime) const	;
    uint64_t		cycleToLocaltime(uint32_t cycle)	 const	;
    
  private:
#if !defined(USE_VIDEO1394)
    static raw1394_iso_disposition
			receive(raw1394handle_t handle, u_char* data,
				u_int len, u_char channel,
				u_char tag, u_char sy,
				u_int cycle, u_int dropped)		;
#endif
#if !defined(__APPLE__)    
    static std::map<int, Port*>	_portMap;

    Port*		_port;
#endif
    raw1394handle_t	_handle;
    nodeid_t		_nodeId;
    const u_int		_delay;
    u_char*		_buf;		// mapped buffer
    uint64_t		_arrivaltime;	// time of buffer captured
#if defined(USE_VIDEO1394)
    video1394_mmap	_mmap;		// mmap structure for video1394
    u_int		_current;	// index of current ready buffer
    u_int		_buf_size;	// buffer size excluding header
#else
    u_char		_channel;	// iso receive channel
    u_char*		_mid;		// the middle of the buffer
    u_char*		_end;		// the end of the buffer
    u_char*		_current;	// current insertion point
    uint64_t		_arrivaltime_next;
#endif
};
    

}
#endif	// !__TU_FIREWIRENODE__H
