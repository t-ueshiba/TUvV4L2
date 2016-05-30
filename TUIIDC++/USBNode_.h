/*
 *  $Id: USBNode_.h,v 1.1.1.1 2012-09-15 08:03:09 ueshiba Exp $
 */
#ifndef __TU_USBNODE__H
#define __TU_USBNODE__H

#include <libusb-1.0/libusb.h>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include "TU/IIDC++.h"

/*!
  \namespace	TU
  \brief	本ライブラリで定義されたクラスおよび関数を納める名前空間
 */
namespace TU
{
/************************************************************************
*  class USBNode							*
************************************************************************/
//! USBのノードを表すクラス
/*!
  一般には, より具体的な機能を持ったノード(ex. デジタルカメラ)を表す
  クラスの基底クラスとして用いられる. 
*/
class USBNode : public IIDCNode
{
  private:
    static void
    exec(int err, const std::string& msg)
    {
	if (err < 0)
	    throw std::runtime_error(msg + ' ' + libusb_error_name(err));
    }

    class Context
    {
      public:
	Context()
	    :_ctx(nullptr)
	{
	    exec(libusb_init(&_ctx), "Failed to initialize libusb!!");
	    libusb_set_debug(_ctx, 3);
	}
	~Context()				{ libusb_exit(_ctx); }

	operator libusb_context*()	const	{ return _ctx; }
	    
      private:
	libusb_context*	_ctx;
    };

    class DeviceIterator
    {
      public:
	using value_type = libusb_device*;
	
      public:
	DeviceIterator(libusb_context* ctx)
	    :_devs(nullptr), _dev(_devs)
	{
	    libusb_get_device_list(ctx, &_devs);
	    _dev = _devs;
	}
	~DeviceIterator()
	{
	    if (_devs)
		libusb_free_device_list(_devs, 1);
	}

	value_type	operator *()		const	{ return *_dev; }
	void		operator ++()			{ ++_dev; }

      private:
	value_type*	_devs;		// リストの先頭
	value_type*	_dev;		// 現在位置
    };

    class Buffer
    {
      public:
			Buffer()				;
			~Buffer()				;
			Buffer(const Buffer&)		= delete;
			Buffer(Buffer&& buffer)		= delete;
	Buffer&		operator =(const Buffer&)	= delete;
	Buffer&		operator =(Buffer&& buffer)	= delete;
	
	const u_char*	data()				const	{ return _p; }
	void		map(USBNode* node, u_int size)		;
	void		unmap()					;
	void		enqueue()			const	;
	void		cancel()			const	;
	
      private:
	static void	callback(libusb_transfer* transger)	;
	
      private:
	USBNode*		_node;
	u_char*			_p;
	libusb_transfer*	_transfer;
    };
	
  public:
			USBNode(uint32_t unit_spec_ID, uint64_t uniqId)	;
    virtual		~USBNode()					;
			USBNode(const USBNode&)			= delete;
			USBNode(USBNode&&)			= delete;
    USBNode&		operator =(const USBNode&)		= delete;
    USBNode&		operator =(USBNode&&)			= delete;

    virtual nodeid_t	nodeId()				const	;
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
    
  private:
    void		set_handle(uint32_t unit_spec_ID,
				   uint64_t uniqId)			;
    void		set_endpoint()					;
    void		start_iso()					;
    void		stop_iso()					;
    static void		mainLoop(USBNode* node)				;
    
  private:
    libusb_device_handle*	_handle;
    int				_interface;
    int				_altsetting;
    libusb_device_handle*	_iso_handle;
    size_t			_nbuffers;
    Buffer*			_buffers;
    std::queue<const Buffer*>	_ready;
    std::atomic_bool		_run;
    std::mutex			_mutex;
    std::condition_variable	_cond;
    std::thread			_thread;
    
    static Context		_ctx;
};

}
#endif	// !__TU_USBNODE__H
