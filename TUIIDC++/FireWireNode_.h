/*
 *  $Id: FireWireNode_.h 1742 2014-11-14 09:48:38Z ueshiba $
 */
#ifndef __TU_FIREWIRENode__H
#define __TU_FIREWIRENode__H

#include <libraw1394/raw1394.h>
#include <vector>
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
  private:
    template <class ITER>
    class RingQueue
    {
      public:
	using iterator	 = ITER;
	using reference	 = typename std::iterator_traits<iterator>::reference;

      public:
	void		initialize(iterator begin, iterator end)
			{
			    _begin = begin;
			    _end   = end;
			    _front = _begin;
			    _back  = _begin;
			}
	bool		empty()		const	{ return _front == _back; }
	reference	front()		const	{ return *_front; }
	reference	back()		const	{ return *_back; }
	void		pop()
			{
			    if (++_front == _end)
				_front = _begin;	// 循環反復子
			}
	void		push()
			{
			    auto	back = _back;
			    if (++back == _end)
				back = _begin;		// 循環反復子

			  // wait状態のバッファが少なくとも1つなければならない
			    if (back != _front)	// wait状態のバッファが残るなら...
				_back = back;	// ready状態のバッファを増やす
			}
	    
      private:
	iterator	_begin;	//!< 記憶領域の先頭
	iterator	_end;	//!< 記憶領域の末尾
	iterator	_front;	//!< [_front, _back) がready状態
	iterator	_back;	//!< [_back, _front) がwait状態
    };
    
    class Buffer
    {
      public:
			Buffer()			;
			~Buffer()			= default;
			Buffer(const Buffer&)		= delete;
	Buffer&		operator =(const Buffer&)	= delete;
			Buffer(Buffer&&)		= default;
	Buffer&		operator =(Buffer&&)		= default;
	
	const void*	data()			const	{ return _data.data(); }
	void		clear()				{ _ndata = 0; }
	void		map(FireWireNode* node, u_int size)		;
	bool		receive(u_char* data, u_int len)		;
	
      private:
	FireWireNode*		_node;
	std::vector<u_char>	_data;	//!< 受信データを収めるバッファ
	u_int			_ndata;	//!< バッファ中の受信データ個数
    };
    
  public:
			FireWireNode(uint32_t unit_spec_ID,
				     uint64_t uniqId)			;
    virtual		~FireWireNode()					;

  //! このノードに割り当てられたisochronousチャンネルを返す
  /*!
    \return	isochronousチャンネル
   */
    u_char		channel()		const	{return _channel;}

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
			receive(raw1394handle_t handle,
				u_char* data, u_int len,
				u_char channel, u_char tag, u_char sy,
				u_int cycle, u_int dropped)		;
    static void		check(bool err, const std::string& msg)		;
#if !defined(__APPLE__)
    u_char		setHandle(uint32_t unit_spec_ID,
				  uint64_t uniqId)			;
#endif

  private:
    using biterator	= std::vector<Buffer>::iterator;
    
    raw1394handle_t		_handle;
    nodeid_t			_nodeId;
    u_char			_channel;
    std::vector<Buffer>		_buffers;
    RingQueue<biterator>	_ready;
};
    

}
#endif	// !__TU_FIREWIRENODE__H
