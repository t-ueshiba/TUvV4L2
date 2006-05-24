/*
 * libTUIeee1394++: C++ Library for Controlling IIDC 1394-based Digital Cameras
 * Copyright (C) 2003-2006 Toshio UESHIBA
 *   National Institute of Advanced Industrial Science and Technology (AIST)
 *
 * Written by Toshio UESHIBA <t.ueshiba@aist.go.jp>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  $Id: raw1394.h,v 1.1 2006-05-24 08:06:26 ueshiba Exp $
 */
#ifndef _LIBRAW1394_RAW1394_H
#define _LIBRAW1394_RAW1394_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif
/************************************************************************
*  type definitions							*
************************************************************************/
typedef struct raw1394*		raw1394handle_t;
typedef unsigned short		nodeid_t;
typedef unsigned long long	nodeaddr_t;
typedef unsigned long		quadlet_t;
enum raw1394_iso_disposition
{
    RAW1394_ISO_OK		= 0,
    RAW1394_ISO_DEFER		= 1,
    RAW1394_ISO_ERROR		= 2,
    RAW1394_ISO_STOP		= 3,
    RAW1394_ISO_STOP_NOSYNC	= 4
};
enum raw1394_iso_dma_recv_mode
{
    RAW1394_DMA_DEFAULT		  = -1,	/* default mode(BUFFERFILL for ohci) */
    RAW1394_DMA_BUFFERFILL	  =  1,	/* BUFFER_FILL mode */
    RAW1394_DMA_PACKET_PER_BUFFER =  2	/* PACKET_PER_BUFFER mode */
};

typedef raw1394_iso_disposition	(*raw1394_iso_recv_handler_t)(
    raw1394handle_t	handle,
    unsigned char*	data,
    unsigned int	len,
    unsigned char	channel,
    unsigned char	tag,
    unsigned char	sy,
    unsigned int	cycle,
    unsigned int	dropped);

/************************************************************************
*  wrapper C functions							*
************************************************************************/
raw1394handle_t
	raw1394_new_handle(unsigned int unit_spec_ID,
			   unsigned long long uniqId);
void	raw1394_destroy_handle(raw1394handle_t handle);
void	raw1394_set_userdata(raw1394handle_t handle, void* data);
void*	raw1394_get_userdata(raw1394handle_t handle);
nodeaddr_t
	raw1394_command_register_base(raw1394handle_t handle);
int	raw1394_read(raw1394handle_t handle, nodeid_t node,
		     nodeaddr_t addr, size_t length, quadlet_t* quad);
int	raw1394_write(raw1394handle_t handle, nodeid_t node,
		      nodeaddr_t addr, size_t length, quadlet_t* quad);
int	raw1394_loop_iterate(raw1394handle_t handle);
int	raw1394_iso_recv_init(raw1394handle_t		     handle,
			      raw1394_iso_recv_handler_t     handler,
			      unsigned int		     buf_packets,
			      unsigned int		     max_packet_size,
			      unsigned char		     channel,
			      raw1394_iso_dma_recv_mode	     mode,
			      int			     irq_interval);
void	raw1394_iso_shutdown(raw1394handle_t handle);
int	raw1394_iso_recv_start(raw1394handle_t handle,
			       int start_on_cycle, int tag_mask,
			       int sync);
void	raw1394_iso_stop(raw1394handle_t handle);
int	raw1394_iso_recv_flush(raw1394handle_t handle);

#ifdef __cplusplus
}
#endif

#endif /* _LIBRAW1394_RAW1394_H */
