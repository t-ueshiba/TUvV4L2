/*
 *  $Id: USBNode_.h,v 1.1.1.1 2012-09-15 08:03:09 ueshiba Exp $
 */
#ifndef __TU_USBNODE__H
#define __TU_USBNODE__H

#include <libusb-1.0/libusb.h>
#include <vector>
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
	}
	~Context()					{ libusb_exit(_ctx); }

	operator libusb_context*()		const	{ return _ctx; }
	    
      private:
	libusb_context*	_ctx;
    };
    
    class DeviceIterator
    {
      public:
	using value_type = libusb_device*;
	
      public:
	DeviceIterator(const Context& ctx)
	    :_devs(nullptr), _dev(_devs)
	{
	    exec(libusb_get_device_list(ctx, &_devs),
		 "Failed to get device list!!");
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
	enum Status	{ FILLED, CORRUPT, ERROR };
	
      public:
	Buffer()					;
	~Buffer()					;
	
	size_t		size()			const	{ return _size; }
	const u_char*	data()			const	{ return _p; }

	void		map(USBNode* parent, size_t size)	;
	void		unmap()					;
	void		enqueue()			const	;
	
      private:
	static void	callback(libusb_transfer* transger)	;
	
      private:
	USBNode*		_parent;
	size_t			_size;
	u_char*			_p;
	libusb_transfer*	_transfer;
    };
	
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
    USBNode(uint32_t unit_spec_ID, uint64_t uniqId)			;
    USBNode(uint16_t idVendor, uint16_t idProduct, uint32_t serialNumber);
    virtual ~USBNode()							;
    USBNode(const USBNode&)					= delete;
    USBNode&	operator =(const USBNode&)			= delete;

  //! このノードのisochronous受信用バッファにフレームの先頭パケットが到着した時刻を返す
  /*!
    \return	受信用バッファにフレームの先頭パケットが到着した時刻
   */
    uint64_t		arrivaltime()		const	{return _arrivaltime;}

    virtual nodeid_t	nodeId()				const	;
    virtual quadlet_t	readQuadlet(nodeaddr_t addr)		const	;
    virtual void	writeQuadlet(nodeaddr_t addr, quadlet_t quad)	;
    virtual u_char	mapListenBuffer(u_int packet_size,
					u_int buf_size,
					u_int nb_buffers)		;
    virtual const u_char*
			waitListenBuffer()				;
    virtual void	requeueListenBuffer()				;
    virtual void	flushListenBuffer()				;
    virtual void	unmapListenBuffer()				;
    uint64_t		cycletimeToLocaltime(uint32_t cycletime)  const	;
    uint64_t		cycleToLocaltime(uint32_t cycle)	  const	;
    
  private:
    static void		mainLoop(USBNode* node)				;
    
  private:
    libusb_device_handle*	_handle;
    std::vector<Buffer>		_buffers;
    std::queue<Buffer*>		_ready;
    std::atomic_bool		_run;
    std::mutex			_mutex;
    std::condition_variable	_cond;
    std::thread			_thread;
    uint64_t			_arrivaltime;	// time of buffer captured
    uint64_t			_arrivaltime_next;

    static constexpr u_int	REQUEST_TIMEOUT_MS = 1000;
    static Context		_ctx;
};

}
#endif	// !__TU_USBNODE__H
