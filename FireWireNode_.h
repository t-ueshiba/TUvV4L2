/*
 *  $Id: FireWireNode_.h 1742 2014-11-14 09:48:38Z ueshiba $
 */
#ifndef __TU_FIREWIRENode__H
#define __TU_FIREWIRENode__H

#include <libraw1394/raw1394.h>
#include "TU/IIDC++.h"

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
  public:
			FireWireNode(uint32_t unit_spec_ID,
				     uint64_t uniqId)			;
    virtual		~FireWireNode()					;

  //! このノードに割り当てられたisochronousチャンネルを返す
  /*!
    \return	isochronousチャンネル
   */
    u_char		channel()		const	{return _channel;}

    virtual bool	isOpened()				const	;
    virtual nodeid_t	nodeId()				const	;
    virtual quadlet_t	readQuadlet(nodeaddr_t addr)		const	;
    virtual void	writeQuadlet(nodeaddr_t addr, quadlet_t quad)	;
    virtual u_char	mapListenBuffer(u_int packet_size,
					u_int buf_size,
					u_int nb_buffers)		;
    virtual void	unmapListenBuffer()				;
    virtual const void*	waitListenBuffer()				;
    virtual void	requeueListenBuffer()				;
    virtual void	flushListenBuffer()				;
    virtual uint32_t	getCycletime(uint64_t& localtime)	const	;
    
  private:
    static raw1394_iso_disposition
			receive(raw1394handle_t handle, u_char* data,
				u_int len, u_char channel,
				u_char tag, u_char sy,
				u_int cycle, u_int dropped)		;
    static void		check(bool err, const std::string& msg)		;
#if !defined(__APPLE__)
    u_char		setHandle(uint32_t unit_spec_ID,
				  uint64_t uniqId)			;
#endif

  private:
    raw1394handle_t	_handle;
    nodeid_t		_nodeId;
    u_char		_channel;
    size_t		_buf_size;
    u_char*		_buf;
    size_t		_len;
    u_char*		_p;
};
    

}
#endif	// !__TU_FIREWIRENODE__H
