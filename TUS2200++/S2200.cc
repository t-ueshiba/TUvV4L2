/*
 *  $Id: S2200.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include "TU/S2200++.h"

namespace TU
{
static const u_int	VCON_BASE =  0x40000;
static const u_int	VCON_SIZE =     0x10;
static const u_int	 VIN_BASE =  0x60020;
static const u_int	 VIN_SIZE =      0x8;
static const u_int	VMAP_BASE =  0x60040;
static const u_int	VMAP_SIZE =      0x8;
static const u_int	VOUT_BASE =  0x60000;
static const u_int	VOUT_SIZE =      0x8;
static const u_int	VRAM_BASE = 0x200000;
static const u_int	VRAM_SIZE = 0x200000;

/************************************************************************
*  Public member functions						*
************************************************************************/
S2200::S2200(const char* devname)
    :Array2<Array<ABGR> >(0, 0),
     _fd  (::open(devname, O_RDWR)),
     _vcon(	    mapon(VCON_BASE, VCON_SIZE)),
     _vin (	    mapon( VIN_BASE,  VIN_SIZE)),
     _vmap(	    mapon(VMAP_BASE, VMAP_SIZE)),
     _vout(	    mapon(VOUT_BASE, VOUT_SIZE)),
     _vram((ABGR*)mapon(VRAM_BASE, VRAM_SIZE)),
     _u0(0), _v0(0), _w(DEFAULT_WIDTH), _h(DEFAULT_HEIGHT),
     _channel(GREEN)
{
    if (_fd == -1)
    {
	perror("open (S2200::S2200)");
	return;
    }
    
    _vout[6] = 0x1a;
    
    _vmap[0] = 0x12;
    _vmap[1] = 0x03;
    _vmap[2] = 0x00;
    _vmap[3] = 0xe0;
    
    _vcon[ 0] = 0x05;
    _vcon[ 1] = 0x0c;
    _vcon[ 2] = 0x00;
    _vcon[ 3] = 0x01;
    _vcon[ 4] = 0x00;
    _vcon[ 5] = 0x00;
    _vcon[ 6] = 0x00;
    _vcon[ 7] = 0x02;
    _vcon[ 8] = 0x00;
    _vcon[ 9] = 0x00;
    _vcon[10] = 0x00;
    _vcon[11] = 0x00;
    _vcon[12] = 0x00;
    _vcon[13] = 0x00;
    _vcon[14] = 0x00;
    
    _vin[0] = 0xf0;
    _vin[1] = 110;
    _vin[2] = 80;
    _vin[3] = 80;
    _vin[4] = 200;
    _vin[5] = 0;

    resize((ABGR*)_vram, DEFAULT_HEIGHT, DEFAULT_WIDTH);
}

S2200::~S2200()
{
    if (_fd == -1)
	return;
    ::close(_fd);
}

S2200&
S2200::set_roi(u_int u0, u_int v0, u_int w, u_int h)
{
    _w  = (w > DEFAULT_WIDTH  ? DEFAULT_WIDTH  : w);
    _h  = (h > DEFAULT_HEIGHT ? DEFAULT_HEIGHT : h);
    _u0 = (u0 + width()  > DEFAULT_WIDTH  ? DEFAULT_WIDTH  - width()  : u0);
    _v0 = (v0 + height() > DEFAULT_HEIGHT ? DEFAULT_HEIGHT - height() : v0);
    return *this;
}

S2200&
S2200::set_channel(Channel channel)
{
    _channel = channel;
    return *this;
}

S2200&
S2200::flow()
{
    if (_fd != -1)
    {
	_vcon[0] = 0x05;
	_vcon[1] = 0x0c;
    }
    return *this;
}

S2200&
S2200::freeze()
{
    if (_fd != -1)
    {
	_vcon[1] = 0x08;
	_vcon[0] = 0x04;
    }
    return *this;
}

S2200&
S2200::capture()
{
    if (_fd != -1)
    {
	_vcon[0] = 0x05;
	_vcon[1] = 0x0d;
	while ((_vcon[0] & 0x0c) != 0);
	_vcon[1] = 0x0f;
	while ((_vcon[0] & 0x02) != 0);
    }
    return *this;
}

S2200&
S2200::operator >>(Image<ABGR>& image)
{
    if (_fd == -1)
	return *this;
    
    image.resize(height(), width());
    for (u_int v = 0; v < image.height(); ++v)
    {
	register const ABGR*	p = &(*this)[v0()+v][u0()];
	register ABGR*	q = image[v];
	for (register int n = image.width(); --n >=0; )
	    *q++ = *p++;
    }
    return *this;
}

S2200&
S2200::operator >>(Image<u_char>& image)
{
    if (_fd == -1)
	return *this;

    image.resize(height(), width());
    switch (channel())
    {
      case OVERLAY:
      {
	for (u_int v = 0; v < image.height(); ++v)
	{
	    register const ABGR*	p = &(*this)[v0()+v][u0()];
	    register u_char*		q = image[v];
	    for (register int n = image.width(); --n >=0; )
		*q++ = p++->a;
	}
      }
	break;

      case RED:
      {
	for (u_int v = 0; v < image.height(); ++v)
	{
	    register const ABGR*	p = &(*this)[v0()+v][u0()];
	    register u_char*		q = image[v];
	    for (register int n = image.width(); --n >=0; )
		*q++ = p++->r;
	}
      }
	break;

      case GREEN:
      {
	for (u_int v = 0; v < image.height(); ++v)
	{
	    register const ABGR*	p = &(*this)[v0()+v][u0()];
	    register u_char*		q = image[v];
	    for (register int n = image.width(); --n >=0; )
		*q++ = p++->g;
	}
      }
	break;

      case BLUE:
      {
	for (u_int v = 0; v < image.height(); ++v)
	{
	    register const ABGR*	p = &(*this)[v0()+v][u0()];
	    register u_char*		q = image[v];
	    for (register int n = image.width(); --n >=0; )
		*q++ = p++->b;
	}
      }
	break;
    }
    return *this;
}

/************************************************************************
*  Private member functions						*
************************************************************************/
u_char*
S2200::mapon(u_int base, u_int size)
{
    if (_fd == -1)
	return 0;
    
    static int	pagesize = 0;
    if (pagesize == 0)
#ifdef __SVR4
      	pagesize = ::sysconf(_SC_PAGESIZE);
#else
	pagesize = ::getpagesize();
#endif
    const int		rest = base % pagesize;
    u_char* const	p = (u_char*)mmap(0, size + rest, PROT_READ|PROT_WRITE,
					  MAP_SHARED, _fd,
					  (off_t)(base - rest));
    if (p == (u_char*)-1)
    {
	_fd = -1;
	return 0;
    }
    return p;
}

void
S2200::set_rows()				// called in resize()
{
    for (u_int i = 0; i < nrow(); i++)
    {
	u_int	offset = (i%2 ? nrow()/2 + (i-1)/2 : i/2);

	(*this)[i].resize((ABGR*)*this + offset*ncol(), ncol());
    }
}
 
}
