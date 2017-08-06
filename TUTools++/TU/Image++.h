/*!
  \file		Image++.h
  \author	Toshio UESHIBA
  \brief	画素と画像に関連するクラスの定義と実装
*/
#ifndef	TU_IMAGEPP_H
#define	TU_IMAGEPP_H

#include <cstring>		// for memcpy()
#include <boost/operators.hpp>
#include <sys/types.h>		// for u_int, u_char
#include "TU/pair.h"
#include "TU/Vector++.h"

using s_char	= signed char;	//!< 符号付き8bit整数

namespace TU
{
namespace detail
{
/************************************************************************
*  class detail::ColorConverter						*
************************************************************************/
class ColorConverter
{
  private:
    constexpr static float	_yr = 0.299f;		// ITU-R BT.601, PAL
    constexpr static float	_yb = 0.114f;		// ITU-R BT.601, PAL
    constexpr static float	_yg = 1.0f - _yr - _yb;	// ITU-R BT.601, PAL
    constexpr static float	_ku = 0.4921f;		// ITU-R BT.601, PAL
    constexpr static float	_kv = 0.877314f;	// ITU-R BT.601, PAL
    
  public:
		ColorConverter()					;

    int		r(int y, int v) const
		{
		    return limit(y + _r[v]);
		}
    int		g(int y, int u, int v) const
		{
		    return limit(y - scaleDown(_gu[u] + _gv[v]));
		}
    int		b(int y, int u) const
		{
		    return limit(y + _b[u]);
		}
    template <class T>
    static T	y(int r, int g, int b)
		{
		    return round<T>(_yr*r + _yg*g + _yb*b);
		}
    int		u(int b, int y) const
		{
		    return _u[255 + b - y];
		}
    int		v(int r, int y) const
		{
		    return _v[255 + r - y];
		}
    
  private:
    static int	limit(int val)
		{
		    return (val < 0 ? 0 : val > 255 ? 255 : val);
		}
    static int	scaleUp(float val)
		{
		    return round<int>(val * (1 << 10));
		}
    static int	scaleDown(int val)
		{
		    return val >> 10;
		}
    template <class T>
    static std::enable_if_t<std::is_integral<T>::value, T>
		round(float val)
		{
		    return T(::round(val));
		}
    template <class T>
    static std::enable_if_t<std::is_floating_point<T>::value, T>
		round(float val)
		{
		    return T(val);
		}
    
  private:
    int		_u[255 + 1 + 255];
    int		_v[255 + 1 + 255];

    int		_r[256];
    int		_gu[256];
    int		_gv[256];
    int		_b[256];
};

extern const ColorConverter	colorConverter;
    
/************************************************************************
*  struct detail::[RGB|BGR|RGBA|ARGB|ABGR|BGRA]				*
************************************************************************/
struct RGB
{
    using element_type = u_char;
    
    RGB(element_type rr, element_type gg, element_type bb)
	:r(rr), g(gg), b(bb)						{}
    
    element_type r, g, b;
};

struct BGR
{
    using element_type = u_char;
    
    BGR(element_type rr, element_type gg, element_type bb)
	:b(bb), g(gg), r(rr)						{}
    
    element_type b, g, r;
};

struct RGBA
{
    using element_type = u_char;
    
    RGBA(element_type rr, element_type gg, element_type bb,
	 element_type aa=255)	:r(rr), g(gg), b(bb), a(aa)		{}
    
    element_type r, g, b, a;
};

struct ABGR
{
    using element_type = u_char;
    
    ABGR(element_type rr, element_type gg, element_type bb,
	 element_type aa=255)	:a(aa), b(bb), g(gg), r(rr)		{}
    
    element_type a, b, g, r;
};

struct ARGB
{
    using element_type = u_char;
    
    ARGB(element_type rr, element_type gg, element_type bb,
	 element_type aa=255)	:a(aa), r(rr), g(gg), b(bb)		{}
    
    element_type a, r, g, b;
};

struct BGRA
{
    using element_type = u_char;
    
    BGRA(element_type rr, element_type gg, element_type bb,
	 element_type aa=255)	:b(bb), g(gg), r(rr), a(aa)		{}
    
    element_type b, g, r, a;
};
}	// namespace detail

/************************************************************************
*  struct RGB_<E>							*
************************************************************************/
struct YUV444;
    
template <class E>
struct RGB_ : public E, boost::additive<RGB_<E>,
			boost::multiplicative<RGB_<E>, float,
			boost::equality_comparable<RGB_<E> > > >
{
    using	typename E::element_type;
    
    RGB_()				:E(0, 0, 0)			{}
    RGB_(element_type rr, element_type gg, element_type bb)
					:E(rr, gg, bb)			{}
    RGB_(element_type rr, element_type gg, element_type bb,
	 element_type aa)		:E(rr, gg, bb, aa)		{}
    RGB_(const RGB_<detail::RGB>& p)	:E(p.r, p.g, p.b)		{}
    RGB_(const RGB_<detail::BGR>& p)	:E(p.r, p.g, p.b)		{}
    template <class E_>
    RGB_(const RGB_<E_>& p)		:E(p.r, p.g, p.b, p.a)		{}
    template <class T_,
	      std::enable_if_t<std::is_convertible<T_, element_type>::value>*
	      = nullptr>
    RGB_(const T_& p)
	:E(element_type(p), element_type(p), element_type(p))		{}
    RGB_(const YUV444& p)						;
    
    using	E::r;
    using	E::g;
    using	E::b;

    template <class T_,
	      std::enable_if_t<std::is_arithmetic<T_>::value>* = nullptr>
		operator T_() const
		{
		    return detail::colorConverter.y<T_>(r, g, b);
		}
    RGB_&	operator +=(const RGB_& p)
		{
		    r += p.r; g += p.g; b += p.b;
		    return *this;
		}
    RGB_&	operator -=(const RGB_& p)
		{
		    r -= p.r; g -= p.g; b -= p.b;
		    return *this;
		}
    RGB_&	operator *=(float c)
		{
		    r *= c; g *= c; b *= c;
		    return *this;
		}
    RGB_&	operator /=(float c)
		{
		    r /= c; g /= c; b /= c;
		    return *this;
		}
    bool	operator ==(const RGB_& p) const
		{
		    return (r == p.r && g == p.g && b == p.b && E::a == p.a);
		}
};

template <class E> inline std::istream&
operator >>(std::istream& in, RGB_<E>& p)
{
    u_int	r, g, b;
    in >> r >> g >> b;
    p.r = r;
    p.g = g;
    p.b = b;

    return in;
}

template <class E> inline std::ostream&
operator <<(std::ostream& out, const RGB_<E>& p)
{
    return out << u_int(p.r) << ' ' << u_int(p.g) << ' ' << u_int(p.b) << ' '
	       << u_int(p.a);
}
    
/************************************************************************
*  struct RGB								*
************************************************************************/
//! Red, Green, Blue（各8bit）の順で並んだカラー画素
using RGB = RGB_<detail::RGB>;

template <> template <class E1> inline
RGB_<detail::RGB>::RGB_(const RGB_<E1>& p) :detail::RGB(p.r, p.g, p.b)	{}
    
template <> inline bool
RGB_<detail::RGB>::operator ==(const RGB_& p) const
{
    return (r == p.r && g == p.g && b == p.b);
}

inline std::istream&
operator >>(std::istream& in, RGB& p)
{
    u_int	r, g, b;
    in >> r >> g >> b;
    p.r = r;
    p.g = g;
    p.b = b;

    return in;
}

inline std::ostream&
operator <<(std::ostream& out, const RGB& p)
{
    return out << u_int(p.r) << ' ' << u_int(p.g) << ' ' << u_int(p.b);
}

/************************************************************************
*  struct BGR								*
************************************************************************/
//! Blue, Green, Red（各8bit）の順で並んだカラー画素
using BGR = RGB_<detail::BGR>;

template <> template <class E1> inline
RGB_<detail::BGR>::RGB_(const RGB_<E1>& p) :detail::BGR(p.r, p.g, p.b)	{}
    
template <> inline bool
RGB_<detail::BGR>::operator ==(const RGB_& p) const
{
    return (r == p.r && g == p.g && b == p.b);
}

inline std::istream&
operator >>(std::istream& in, BGR& p)
{
    u_int	r, g, b;
    in >> r >> g >> b;
    p.r = r;
    p.g = g;
    p.b = b;

    return in;
}

inline std::ostream&
operator <<(std::ostream& out, const BGR& p)
{
    return out << u_int(p.r) << ' ' << u_int(p.g) << ' ' << u_int(p.b);
}

/************************************************************************
*  struct RGBA, ABGR, ARGB, BGRA					*
************************************************************************/
//! Red, Green, Blue, Alpha（各8bit）の順で並んだカラー画素
using RGBA = RGB_<detail::RGBA>;
//! Alpha, Blue, Green, Red（各8bit）の順で並んだカラー画素
using ABGR = RGB_<detail::ABGR>;
//! Alpha, Red, Green, Blue（各8bit）の順で並んだカラー画素
using ARGB = RGB_<detail::ARGB>;
//! Blue, Green, Red, Alpha（各8bit）の順で並んだカラー画素
using BGRA = RGB_<detail::BGRA>;

/************************************************************************
*  struct YUV444, YUV422, YUYV422, YUV411				*
************************************************************************/
//! U, Y, V（各8bit）の順で並んだカラー画素
struct YUV444
{
    using element_type = u_char;

    YUV444(element_type yy=0, element_type uu=128, element_type vv=128)
		    :u(uu), y(yy), v(vv)				{}
    template <class E>
    YUV444(const RGB_<E>& p)
		    :y(detail::colorConverter.y<element_type>(p.r, p.g, p.b))
		{
		    u = detail::colorConverter.u(p.b, y);
		    v = detail::colorConverter.v(p.r, y);
		}
    template <class T,
	      std::enable_if_t<std::is_convertible<T, element_type>::value>*
	      = nullptr>
    YUV444(const T& p)	:u(128), y(p), v(128)				{}
    
    template <class T,
	      std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
		operator T()			const	{return T(y);}
    bool	operator ==(const YUV444& yuv)	const	{return (u == yuv.u &&
								 y == yuv.y &&
								 v == yuv.v);}
    bool	operator !=(const YUV444& yuv)	const	{return !(*this==yuv);}

    element_type	u, y, v;
};
    
inline std::istream&
operator >>(std::istream& in, YUV444& yuv)
{
    u_int	y, u, v;
    in >> y >> u >> v;
    yuv.y = y;
    yuv.u = u;
    yuv.v = v;
    
    return in;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV444& yuv)
{
    return out << u_int(yuv.y) << ' ' << u_int(yuv.u) << ' ' << u_int(yuv.v);
}

template <class E> inline
RGB_<E>::RGB_(const YUV444& p)
    :E(detail::colorConverter.r(p.y, p.v),
       detail::colorConverter.g(p.y, p.u, p.v),
       detail::colorConverter.b(p.y, p.u))
{
}

struct YUYV422;

//! [U, Y0], [V, Y1]（各8bit）の順で並んだカラー画素(16bits/pixel)
struct YUV422
{
    using element_type = u_char;

    YUV422(element_type yy=0, element_type xx=128)	:x(xx), y(yy)	{}
    YUV422(const YUYV422& p)						;

    template <class T,
	      std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
		operator T()			const	{return T(y);}
    bool	operator ==(const YUV422& p)	const	{return (x == p.x &&
								 y == p.y);}
    bool	operator !=(const YUV422& p)	const	{return !(*this == p);}
    
    element_type	x, y;
};

inline std::istream&
operator >>(std::istream& in, YUV422& yuv)
{
    u_int	y, x;
    in >> y >> x;
    yuv.y = y;
    yuv.x = x;
    
    return in;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV422& yuv)
{
    return out << u_int(yuv.y) << ' ' << u_int(yuv.x);
}

//! [Y0, U], [Y1, V]（各8bit）の順で並んだカラー画素(16bits/pixel)
struct YUYV422
{
    using element_type = u_char;

    YUYV422(element_type yy=0, element_type xx=128)	:y(yy), x(xx)	{}
    YUYV422(const YUV422& p)				:y(p.y), x(p.x)	{}

    template <class T,
	      std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
		operator T()			const	{return T(y);}
    bool	operator ==(const YUYV422& p)	const	{return (y == p.y &&
								 x == p.x);}
    bool	operator !=(const YUYV422& p)	const	{return !(*this == p);}
    
    element_type	y, x;
};

inline std::istream&
operator >>(std::istream& in, YUYV422& yuv)
{
    u_int	y, x;
    in >> y >> x;
    yuv.y = y;
    yuv.x = x;
    
    return in;
}

inline std::ostream&
operator <<(std::ostream& out, const YUYV422& yuv)
{
    return out << u_int(yuv.y) << ' ' << u_int(yuv.x);
}

inline
YUV422::YUV422(const YUYV422& p)  :x(p.x), y(p.y)	{}

//! [U, Y0, Y1], [V, Y2, Y3]（各8bit）の順で並んだカラー画素(12bits/pixel)
struct YUV411
{
    using element_type = u_char;

    YUV411(element_type yy0=0, element_type yy1=0, element_type xx=128)
	:x(xx), y0(yy0), y1(yy1)					{}
    template <class T,
	      std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
    YUV411(const T& p)	:x(128), y0(p), y1((&p)[1])			{}

    bool	operator ==(const YUV411& p)	const	{return (x  == p.x  &&
								 y0 == p.y0 &&
								 y1 == p.y1);}
    bool	operator !=(const YUV411& p)	const	{return !(*this == p);}
    
    element_type	x, y0, y1;
};

inline std::istream&
operator >>(std::istream& in, YUV411& yuv)
{
    u_int	y0, y1, x;
    in >> y0 >> y1 >> x;
    yuv.y0 = y0;
    yuv.y1 = y1;
    yuv.x  = x;
    
    return in;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV411& yuv)
{
    return out << u_int(yuv.y0) << ' ' << u_int(yuv.y1) << ' ' << u_int(yuv.x);
}

/************************************************************************
*  class pixel_iterator<ITER>						*
************************************************************************/
namespace detail
{
  template <class ITER, class T=iterator_value<ITER> >
  class pixel_proxy
  {
    public:
      using value_type	= T;

      static constexpr size_t	npixels = 1;
      
    public:
      pixel_proxy(const ITER& iter)	:_iter(const_cast<ITER&>(iter))	{}

      template <class ITER_, class T_>
      pixel_proxy&	operator =(const pixel_proxy<ITER_, T_>& proxy)
			{
			    using N = std::integral_constant<
					  size_t,
					  pixel_proxy<ITER_, T_>::npixels>;

			    assign(proxy.value(N()));
			    return *this;
			}
      
      value_type	value(std::integral_constant<size_t, npixels>) const
			{
			    const auto	val = *_iter;
			    ++_iter;
			    return val;
			}
      template <size_t N>
      TU::pair_tree<T, N>
			value(std::integral_constant<size_t, N>) const
			{
			    using N2 = std::integral_constant<size_t, N/2>;
		    
			    const auto	val0 = value(N2());
			    return std::make_pair(val0, value(N2()));
			}

    private:
      template <class T_>
      void		assign(const T_& val)
			{
			    *_iter = val;
			    ++_iter;
			}
      template <class T_>
      void		assign(const std::pair<T_, T_>& val)
			{
			    assign(val.first);
			    assign(val.second);
			}

    private:
      ITER&	_iter;
  };
    
  template <class ITER>
  class pixel_proxy<ITER, YUV422>
  {
    public:
      using value_type		= TU::pair_tree<YUV444, 2>;
      using element_type	= YUV422::element_type;
      
      static constexpr size_t	npixels = 2;
      
    public:
      pixel_proxy(const ITER& iter)	:_iter(const_cast<ITER&>(iter))	{}

      template <class ITER_, class T_>
      pixel_proxy&	operator =(const pixel_proxy<ITER_, T_>& proxy)
			{
			    constexpr size_t
				Np = pixel_proxy<ITER_, T_>::npixels;
			    using N = std::integral_constant<
					  size_t,
					  (npixels > Np ? npixels : Np)>;
		    
			    assign(proxy.value(N()));
			    return *this;
			}

      value_type	value(std::integral_constant<size_t, npixels>) const
			{
			    const auto	val0 = *_iter;
			    const auto	val1 = *(++_iter);
			    ++_iter;
			    return {{val0.y, val0.x, val1.x},
				    {val1.y, val0.x, val1.x}};
			}
      template <size_t N>
      TU::pair_tree<YUV444, N>
			value(std::integral_constant<size_t, N>) const
			{
			    using N2 = std::integral_constant<size_t, N/2>;
		    
			    const auto	val0 = value(N2());
			    return std::make_pair(val0, value(N2()));
			}

    private:
      template <class T_> std::enable_if_t<!is_pair<T_>::value>
			assign(const std::pair<T_, T_>& val)
			{
			    YUV444	val0(val.first);
			    *_iter     = {val0.y, val0.u};
			    *(++_iter) = {element_type(val.second), val0.v};
			    ++_iter;
			}
      template <class T_> std::enable_if_t<is_pair<T_>::value>
			assign(const std::pair<T_, T_>& val)
			{
			    assign(val.first);
			    assign(val.second);
			}

    private:
      ITER&	_iter;
  };

  template <class ITER>
  class pixel_proxy<ITER, YUYV422>
  {
    public:
      using value_type		= TU::pair_tree<YUV444, 2>;
      using element_type	= YUYV422::element_type;
      
      static constexpr size_t	npixels = 2;
      
    public:
      pixel_proxy(const ITER& iter)	:_iter(const_cast<ITER&>(iter))	{}

      template <class ITER_, class T_>
      pixel_proxy&	operator =(const pixel_proxy<ITER_, T_>& proxy)
			{
			    constexpr size_t
				Np = pixel_proxy<ITER_, T_>::npixels;
			    using N = std::integral_constant<
					  size_t,
					  (npixels > Np ? npixels : Np)>;
		    
			    assign(proxy.value(N()));
			    return *this;
			}

      value_type	value(std::integral_constant<size_t, npixels>) const
			{
			    const auto	val0 = *_iter;
			    const auto	val1 = *(++_iter);
			    ++_iter;
			    return {{val0.y, val0.x, val1.x},
				    {val1.y, val0.x, val1.x}};
			}
      template <size_t N>
      TU::pair_tree<YUV444, N>
			value(std::integral_constant<size_t, N>) const
			{
			    using N2 = std::integral_constant<size_t, N/2>;
		    
			    const auto	val0 = value(N2());
			    return std::make_pair(val0, value(N2()));
			}

    private:
      template <class T_> std::enable_if_t<!is_pair<T_>::value>
			assign(const std::pair<T_, T_>& val)
			{
			    YUV444	val0(val.first);
			    *_iter     = {val0.y, val0.u};
			    *(++_iter) = {element_type(val.second), val0.v};
			    ++_iter;
			}
      template <class T_> std::enable_if_t<is_pair<T_>::value>
			assign(const std::pair<T_, T_>& val)
			{
			    assign(val.first);
			    assign(val.second);
			}

    private:
      ITER&	_iter;
  };

  template <class ITER>
  class pixel_proxy<ITER, YUV411>
  {
    public:
      using value_type		= TU::pair_tree<YUV444, 4>;
      using element_type	= YUV411::element_type;
      
      static constexpr size_t	npixels = 4;
      
    public:
      pixel_proxy(const ITER& iter)	:_iter(const_cast<ITER&>(iter))	{}

      template <class ITER_, class T_>
      pixel_proxy&	operator =(const pixel_proxy<ITER_, T_>& proxy)
			{
			    constexpr size_t
				Np = pixel_proxy<ITER_, T_>::npixels;
			    using N = std::integral_constant<
					  size_t,
					  (npixels > Np ? npixels : Np)>;
		    
			    assign(proxy.value(N()));
			    return *this;
			}

      value_type	value(std::integral_constant<size_t, npixels>) const
			{
			    const auto	val0 = *_iter;
			    const auto	val1 = *(++_iter);
			    ++_iter;
			    return {{{val0.y0, val0.x, val1.x},
				     {val0.y1, val0.x, val1.x}},
				    {{val1.y0, val0.x, val1.x},
				     {val1.y1, val0.x, val1.x}}};
			}

    private:
      template <class T_>
      void		assign(const std::pair<std::pair<T_, T_>,
					       std::pair<T_, T_> >& val)
			{
			    YUV444	val0(val.first.first);
			    *_iter     = {val0.y,
					  element_type(val.first.second),
					  val0.u};
			    *(++_iter) = {element_type(val.second.first),
					  element_type(val.second.second),
					  val0.v};
			    ++_iter;
			}

    private:
      ITER&	_iter;
  };
}

template <class ITER>
class pixel_iterator
    : public boost::iterator_adaptor<pixel_iterator<ITER>,
				     ITER,
				     detail::pixel_proxy<ITER>,
				     std::forward_iterator_tag,
				     detail::pixel_proxy<ITER> >
{
  private:
    using super	= boost::iterator_adaptor<pixel_iterator,
					  ITER,
					  detail::pixel_proxy<ITER>,
					  std::forward_iterator_tag,
					  detail::pixel_proxy<ITER> >;
    
  public:
    using reference	= typename super::reference;

    friend class	boost::iterator_core_access;
    
  public:
		pixel_iterator(ITER iter)	:super(iter)		{}

  private:
    reference	dereference() const
		{
		    return reference(super::base());
		}
    void	increment()
		{
		}
    bool	equal(const pixel_iterator& iter) const
		{
		    return super::base() > iter.base() - reference::npixels;
		}
};

template <class ITER> inline pixel_iterator<ITER>
make_pixel_iterator(ITER iter)
{
    return pixel_iterator<ITER>(iter);
}

/************************************************************************
*  Bayer pattern decoding functions					*
************************************************************************/
namespace detail
{
template <class IN, class OUT, class C> void
bayerDecodeRowXGGY(IN inY, IN inYe, IN inXp, IN inXn, OUT out, C X, C Y)
{
    if (inY == inYe)
	return;
    
    auto	x0 = (*inXp + *inXn) >> 1;
    auto	g0 = *inY;
    auto	y1 = *++inY;
    out->*X = x0;
    out->g  = g0;
    out->*Y = y1;
    ++out;
    
    while (++inY != inYe)
    {
	const auto	g2 = *inY;
	out->g  = (*++inXp + *++inXn + g0 + g2) >> 2;
	const auto	x2 = (*++inXp + *++inXn) >> 1;
	out->*X = (x0 + x2) >> 1;
	out->*Y = y1;
	++out;

	const auto	y3 = *++inY;
	out->*X = x2;
	out->g  = g2;
	out->*Y = (y1 + y3) >> 1;
	++out;

	x0 = x2;
	g0 = g2;
	y1 = y3;
    }

    out->*X = x0;
    out->g  = (*++inXp + *++inXn) >> 1;
    out->*Y = y1;
}
    
template <class IN, class OUT, class C> void
bayerDecodeRowGXYG(IN inY, IN inYe, IN inXp, IN inXn, OUT out, C X, C Y)
{
    if (inY == inYe)
	return;
    
    out->g  = (*inXp + *inXn) >> 1;
    auto	y0 = *inY;
    auto	x1 = (*++inXp + *++inXn) >> 1;
    auto	g1 = *++inY;
    out->*X = x1;
    out->*Y = y0;
    ++out;
    
    while (++inY != inYe)
    {
	const auto	y2 = *inY;
	out->*X = x1;
	out->g  = g1;
	out->*Y = (y0 + y2) >> 1;
	++out;

	const auto	g3 = *++inY;
	out->g  = (*++inXp + *++inXn + g1 + g3) >> 2;
	const auto	x3 = (*++inXp + *++inXn) >> 1;
	out->*X = (x1 + x3) >> 1;
	out->*Y = y2;
	++out;

	y0 = y2;
	x1 = x3;
	g1 = g3;
    }

    out->*X = x1;
    out->g  = g1;
    out->*Y = y0;
}
    
template <class IN, class OUT, class C> void
bayerDecodeBorderRowXGGY(IN inX, IN inXe, IN inY, OUT out, C X, C Y)
{
    if (inX == inXe)
	return;
    
    auto	x0 = *inX;
    auto	g1 = *++inX;
    auto	y1 = *++inY;
    out->*X = x0;
    out->g  = g1;
    out->*Y = y1;
    ++out;
    
    while (++inX != inXe)
    {
	const auto	x2 = *inX;
	out->*X = (x0 + x2) >> 1;
	out->g  = g1;
	out->*Y = y1;
	++out;

	const auto	g3 = *++inX;
	++inY;
	const auto	y3 = *++inY;
	out->*X = x2;
	out->g  = (g1 + g3) >> 1;
	out->*Y = (y1 + y3) >> 1;
	++out;

	x0 = x2;
	g1 = g3;
	y1 = y3;
    }

    out->*X = x0;
    out->g  = g1;
    out->*Y = y1;
}

template <class IN, class OUT, class C> void
bayerDecodeBorderRowGXYG(IN inX, IN inXe, IN inY, OUT out, C X, C Y)
{
    if (inX == inXe)
	return;
    
    auto	g0 = *inX;
    auto	x1 = *++inX;
    auto	y0 = *inY;
    out->*X = x1;
    out->g  = g0;
    out->*Y = y0;
    ++out;
    
    while (++inX != inXe)
    {
	const auto	g2 = *inX;
	++inY;
	const auto	y2 = *++inY;
	out->*X = x1;
	out->g  = (g0 + g2) >> 1;
	out->*Y = (y0 + y2) >> 1;
	++out;

	const auto	x3 = *++inX;
	out->*X = (x1 + x3) >> 1;
	out->g  = g2;
	out->*Y = y2;
	++out;

	g0 = g2;
	x1 = x3;
	y0 = y2;
    }

    out->*X = x1;
    out->g  = g0;
    out->*Y = y0;
}

}	// namespace detail
    
template <class IN, class OUT> OUT
bayerDecodeRGGB(IN in, IN ie, OUT out)
{
    using	COLOR = typename iterator_value<OUT>::value_type;

    if (in == ie)
	return out;

    auto	ic = in;
    ++in;
    detail::bayerDecodeBorderRowXGGY(TU::cbegin(*ic), TU::cend(*ic),
				     TU::cbegin(*in), TU::begin(*out),
				     &COLOR::r, &COLOR::b);
    ++out;
    auto	ip = ic;
    for (++ic; ++in != ie; ++ip, ++ic)
    {
	detail::bayerDecodeRowXGGY(TU::cbegin(*ic), TU::cend(*ic),
				   TU::cbegin(*ip), TU::cbegin(*in),
				   TU::begin(*out), &COLOR::r, &COLOR::b);
	++out;
	++ip;
	++ic;
	++in;
	detail::bayerDecodeRowGXYG(TU::cbegin(*ic), TU::cend(*ic),
				   TU::cbegin(*ip), TU::cbegin(*in),
				   TU::begin(*out), &COLOR::b, &COLOR::r);
	++out;
    }
    detail::bayerDecodeBorderRowGXYG(TU::cbegin(*ic), TU::cend(*ic),
				     TU::cbegin(*ip), TU::begin(*out),
				     &COLOR::b, &COLOR::r);

    return ++out;
}

template <class IN, class OUT> OUT
bayerDecodeBGGR(IN in, IN ie, OUT out)
{
    using	COLOR = typename iterator_value<OUT>::value_type;

    if (in == ie)
	return out;

    auto	ic = in;
    ++in;
    detail::bayerDecodeBorderRowXGGY(TU::cbegin(*ic), TU::cend(*ic),
				     TU::cbegin(*in), TU::begin(*out),
				     &COLOR::b, &COLOR::r);
    ++out;
    auto	ip = ic;
    for (++ic; ++in != ie; ++ip, ++ic)
    {
	detail::bayerDecodeRowXGGY(TU::cbegin(*ic), TU::cend(*ic),
				   TU::cbegin(*ip), TU::cbegin(*in),
				   TU::begin(*out), &COLOR::b, &COLOR::r);
	++out;
	++ip;
	++ic;
	++in;
	detail::bayerDecodeRowGXYG(TU::cbegin(*ic), TU::cend(*ic),
				   TU::cbegin(*ip), TU::cbegin(*in),
				   TU::begin(*out), &COLOR::r, &COLOR::b);
	++out;
    }
    detail::bayerDecodeBorderRowGXYG(TU::cbegin(*ic), TU::cend(*ic),
				     TU::cbegin(*ip), TU::begin(*out),
				     &COLOR::r, &COLOR::b);

    return ++out;
}

template <class IN, class OUT> OUT
bayerDecodeGRBG(IN in, IN ie, OUT out)
{
    using	COLOR = typename iterator_value<OUT>::value_type;

    if (in == ie)
	return out;

    auto	ic = in;
    ++in;
    detail::bayerDecodeBorderRowGXYG(TU::cbegin(*ic), TU::cend(*ic),
				     TU::cbegin(*in), TU::begin(*out),
				     &COLOR::r, &COLOR::b);
    ++out;
    auto	ip = ic;
    for (++ic; ++in != ie; ++ip, ++ic)
    {
	detail::bayerDecodeRowGXYG(TU::cbegin(*ic), TU::cend(*ic),
				   TU::cbegin(*ip), TU::cbegin(*in),
				   TU::begin(*out), &COLOR::r, &COLOR::b);
	++out;
	++ip;
	++ic;
	++in;
	detail::bayerDecodeRowXGGY(TU::cbegin(*ic), TU::cend(*ic),
				   TU::cbegin(*ip), TU::cbegin(*in),
				   TU::begin(*out), &COLOR::b, &COLOR::r);
	++out;
    }
    detail::bayerDecodeBorderRowXGGY(TU::cbegin(*ic), TU::cend(*ic),
				     TU::cbegin(*ip), TU::begin(*out),
				     &COLOR::b, &COLOR::r);

    return ++out;
}

template <class IN, class OUT> OUT
bayerDecodeGBRG(IN in, IN ie, OUT out)
{
    using	COLOR = typename iterator_value<OUT>::value_type;

    if (in == ie)
	return out;

    auto	ic = in;
    ++in;
    detail::bayerDecodeBorderRowGXYG(TU::cbegin(*ic), TU::cend(*ic),
				     TU::cbegin(*in), begin(*out),
				     &COLOR::b, &COLOR::r);
    ++out;
    auto	ip = ic;
    for (++ic; ++in != ie; ++ip, ++ic)
    {
	detail::bayerDecodeRowGXYG(TU::cbegin(*ic), TU::cend(*ic),
				   TU::cbegin(*ip), TU::cbegin(*in),
				   TU::begin(*out), &COLOR::b, &COLOR::r);
	++out;
	++ip;
	++ic;
	++in;
	detail::bayerDecodeRowXGGY(TU::cbegin(*ic), TU::cend(*ic),
				   TU::cbegin(*ip), TU::cbegin(*in),
				   TU::begin(*out), &COLOR::r, &COLOR::b);
	++out;
    }
    detail::bayerDecodeBorderRowXGGY(TU::cbegin(*ic), TU::cend(*ic),
				     TU::cbegin(*ip), TU::begin(*out),
				     &COLOR::r, &COLOR::b);

    return ++out;
}

/************************************************************************
*  class ImageBase:	basic image class				*
************************************************************************/
//! 画素の2次元配列として定義されたあらゆる画像の基底となるクラス
class ImageBase
{
  public:
    using matrix34_type	= Matrix<double, 3, 4>;
    
  //! 外部記憶に読み書きする際の画素のタイプ
    enum Type
    {
	DEFAULT = 0,	//!< same as internal type
	U_CHAR	= 5,	//!< unsigned mono	 8bit/pixel
	RGB_24	= 6,	//!< RGB		24bit/pixel	
	SHORT,		//!< signed mono	16bit/pixel
	INT,		//!< signed mono	32bit/pixel	
	FLOAT,		//!< float mono		32bit/pixel 
	DOUBLE,		//!< double mono	64bit/pixel
	YUV_444,	//!< YUV444		24bit/pixel
	YUV_422,	//!< YUV422		16bit/pixel
	YUYV_422,	//!< YUYV422		16bit/pixel
	YUV_411,	//!< YUV411		12bit/pixel
	BMP_8,		//!< BMP indexed color   8bit/pixel
	BMP_24,		//!< BMP BGR		24bit/pixel
	BMP_32		//!< BMP BGRA		32bit/pixel
    };

  //! 外部記憶に読み書きする際の付加情報
    struct TypeInfo
    {
	TypeInfo(Type ty=DEFAULT)	;

	Type	type;		//!< 画素の型
	bool	bottomToTop;	//!< 行が下から上へ収められているならtrue
	size_t	ncolors;	//!< カラーパレットの色数
    };

  protected:
    template <class _T, class _DUMMY=void>
    struct type2type			{static constexpr Type value=RGB_24;};
    template <class _DUMMY>
    struct type2type<u_char, _DUMMY>	{static constexpr Type value=U_CHAR;};
    template <class _DUMMY>
    struct type2type<short, _DUMMY>	{static constexpr Type value=SHORT;};
    template <class _DUMMY>
    struct type2type<int, _DUMMY>	{static constexpr Type value=INT;};
    template <class _DUMMY>
    struct type2type<float, _DUMMY>	{static constexpr Type value=FLOAT;};
    template <class _DUMMY>
    struct type2type<double, _DUMMY>	{static constexpr Type value=DOUBLE;};
    template <class _DUMMY>
    struct type2type<YUV444, _DUMMY>	{static constexpr Type value=YUV_444;};
    template <class _DUMMY>
    struct type2type<YUV422, _DUMMY>	{static constexpr Type value=YUV_422;};
    template <class _DUMMY>
    struct type2type<YUYV422, _DUMMY>	{static constexpr Type value=YUYV_422;};
    template <class _DUMMY>
    struct type2type<YUV411, _DUMMY>	{static constexpr Type value=YUV_411;};
    
  protected:
  //! 画像を生成し投影行列と放射歪曲係数を初期化する．
  /*!
    投影行列は
    \f$\TUbeginarray{cc} \TUvec{I}{3\times 3} & \TUvec{0}{} \TUendarray\f$に，
    2つの放射歪曲係数はいずれも0に初期化される．
  */
    ImageBase()
	:P(), d1(0), d2(0)		{P[0][0] = P[1][1] = P[2][2] = 1.0;}
    virtual ~ImageBase()		;

    size_t		type2nbytes(Type type, bool padding)	const	;
    static size_t	type2depth(Type type)				;
    
  public:
    TypeInfo		restoreHeader(std::istream& in)			;
    Type		saveHeader(std::ostream& out,
				   Type type=DEFAULT)		const	;

  //! 画像の幅を返す．
  /*!
    \return	画像の幅
  */
    size_t		width()			const	{return _width();}

  //! 画像の高さを返す．
  /*!
    \return	画像の高さ
  */
    size_t		height()		const	{return _height();}

    size_t		npixelsToBorder(size_t u, size_t v,
					size_t dir)	const	;
    
  private:
    TypeInfo		restorePBMHeader(std::istream& in)	;
    TypeInfo		restoreBMPHeader(std::istream& in)	;
    Type		savePBMHeader(std::ostream& out,
				      Type type)	const	;
    Type		saveBMPHeader(std::ostream& out,
				      Type type)	const	;
    virtual size_t	_width()			const	= 0;
    virtual size_t	_height()			const	= 0;
    virtual Type	_defaultType()			const	= 0;
    virtual void	_resize(size_t h, size_t w,
				const TypeInfo& typeInfo)	= 0;

  public:
    matrix34_type	P;	//!< カメラの3x4投影行列
    double		d1;	//!< 放射歪曲の第1係数
    double		d2;	//!< 放射歪曲の第2係数
};

//! 指定された向きに沿った与えられた点から画像境界までの画素数を返す．
/*!
  \param u	始点の横座標
  \param v	始点の縦座標
  \param dir	8隣接方向
  \return	画像境界までの画素数(始点を含む)
*/
inline size_t
ImageBase::npixelsToBorder(size_t u, size_t v, size_t dir) const
{
    switch (dir % 8)
    {
      case 0:
	return width() - u;
      case 1:
	return std::min(width() - u, height() - v);
      case 2:
	return height() - v;
      case 3:
	return std::min(u + 1, height() - v);
      case 4:
	return u;
      case 5:
	return std::min(u + 1, v + 1);
      case 6:
	return v + 1;
    }

    return std::min(width() - u, v + 1);
}

/************************************************************************
*  class Image<T, ALLOC>						*
************************************************************************/
//! T型の画素を持つ画像を表すクラス
/*!
  \param T	画素の型
  \param ALLOC	アロケータの型
*/
template <class T, class ALLOC=std::allocator<T> >
class Image : public Array2<T, 0, 0, ALLOC>, public ImageBase
{
  private:
    using super	= Array2<T, 0, 0, ALLOC>;

    template <class L>
    struct Colormap : public Array<L>
    {
	using result_type = L;

	Colormap(size_t d)	:Array<L>(d)		{}
	
	result_type	operator ()(size_t i)	const	{ return (*this)[i]; }
    };
    
  public:
    using	typename super::element_type;
    using	typename super::pointer;

  public:
  //! 幅と高さを指定して画像を生成する．
  /*!
    \param w	画像の幅
    \param h	画像の高さ
  */
    explicit Image(size_t w=0, size_t h=0)
	:super(h, w), ImageBase()				{}

  //! 幅と高さとpaddingを指定して画像を生成する．
  /*!
    \param w	画像の幅
    \param h	画像の高さ
    \param unit	1行あたりのバイト数がこの値の倍数になる
  */
    explicit Image(size_t w, size_t h, size_t unit)
	:super(unit, h, w), ImageBase()				{}

  //! 他の配列と同一要素を持つ画像を作る（コピーコンストラクタの拡張）．
  /*!
    コピーコンストラクタを定義しないと自動的に作られてしまうので，
    このコンストラクタがあってもコピーコンストラクタを別個に定義
    しなければならない．
    \param expr	コピー元の配列
  */
    template <class E_, std::enable_if_t<rank<E_>() == 2>* = nullptr>
    Image(const E_& expr)
	:super(expr), ImageBase()				{}

    Image(pointer p, size_t w, size_t h)
	:super(p, h, w), ImageBase()				{}
    
    Image(pointer p, size_t unit, size_t w, size_t h)
	:super(p, unit, h, w), ImageBase()			{}
    
  //! 他の配列を自分に代入する（標準代入演算子の拡張）．
  /*!
    標準代入演算子を定義しないと自動的に作られてしまうので，この代入演算子が
    あっても標準代入演算子を別個に定義しなければならない．
    \param expr		コピー元の配列
    \return		この配列
  */
    template <class E_> std::enable_if_t<rank<E_>() == 2, Image&>
		operator =(const E_& expr)
		{
		    super::operator =(expr);
		    return *this;
		}

    template <class T_> std::enable_if_t<rank<T_>() == 0, Image&>
		operator =(const T_& c)
		{
		    super::operator =(c);
		    return *this;
		}
    
    using	super::begin;
    using	super::end;
    using	super::rbegin;
    using	super::rend;

//! 指定された位置の画素にアクセスする．
  /*!
    \param p	画素の位置
    \return	指定された画素
  */
    template <class S>
    const T&	operator ()(const Array<S, 2>& p)
					const	{return (*this)[p[1]][p[0]];}

  //! 指定された位置の画素にアクセスする．
  /*!
    \param p	画素の位置
    \return	指定された画素
  */
    template <class S>
    T&		operator ()(const Array<S, 2>& p)
						{return (*this)[p[1]][p[0]];}
    
    size_t	width()			const	{return super::ncol();}
    size_t	height()		const	{return super::nrow();}

    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out,
			     Type type=DEFAULT)			const	;
    std::istream&	restoreData(std::istream& in,
				    const TypeInfo& typeInfo)		;
    std::ostream&	saveData(std::ostream& out,
				 Type type=DEFAULT)		const	;

  private:
    template <class S>
    std::istream&	restoreRows(std::istream& in,
				    const TypeInfo& typeInfo)		;
    template <class S, class L>
    std::istream&	restoreAndLookupRows(std::istream& in,
					     const TypeInfo& typeInfo)	;
    template <class D, class L>
    std::ostream&	saveRows(std::ostream& out, Type type)	const	;
    Type		defaultType()				const	;
    
    virtual size_t	_width()				const	;
    virtual size_t	_height()				const	;
    virtual Type	_defaultType()				const	;
    virtual void	_resize(size_t h, size_t w, const TypeInfo&)	;
};

//! 入力ストリームから画像を読み込む．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, class ALLOC> inline std::istream&
Image<T, ALLOC>::restore(std::istream& in)
{
    return restoreData(in, restoreHeader(in));
}

//! 指定した画素タイプで出力ストリームに画像を書き出す．
/*!
  \param out	出力ストリーム
  \param type	画素タイプ．ただし， #DEFAULT を指定した場合は，
		この画像オブジェクトの画素タイプで書き出される．
  \return	outで指定した出力ストリーム
*/
template <class T, class ALLOC> inline std::ostream&
Image<T, ALLOC>::save(std::ostream& out, Type type) const
{
    return saveData(out, saveHeader(out, type));
}

//! 入力ストリームから画像の画素データを読み込む．
/*!
  \param in		入力ストリーム
  \param typeInfo	ストリーム中のデータの画素タイプ
			(読み込み先の画像の画素タイプではない)
  \return		inで指定した入力ストリーム
*/
template <class T, class ALLOC> std::istream&
Image<T, ALLOC>::restoreData(std::istream& in, const TypeInfo& typeInfo)
{
    switch (typeInfo.type)
    {
      case DEFAULT:
	break;
      case U_CHAR:
	return restoreRows<u_char>(in, typeInfo);
      case SHORT:
	return restoreRows<short>(in, typeInfo);
      case INT:
	return restoreRows<int>(in, typeInfo);
      case FLOAT:
	return restoreRows<float>(in, typeInfo);
      case DOUBLE:
	return restoreRows<double>(in, typeInfo);
      case RGB_24:
	return restoreRows<RGB>(in, typeInfo);
      case YUV_444:
	return restoreRows<YUV444>(in, typeInfo);
      case YUV_422:
	return restoreRows<YUV422>(in, typeInfo);
      case YUYV_422:
	return restoreRows<YUYV422>(in, typeInfo);
      case YUV_411:
	return restoreRows<YUV411>(in, typeInfo);
      case BMP_8:
	return restoreAndLookupRows<u_char, BGRA>(in, typeInfo);
      case BMP_24:
	return restoreRows<BGR>(in, typeInfo);
      case BMP_32:
	return restoreRows<BGRA>(in, typeInfo);
      default:
	throw std::runtime_error("Image<T, ALLOC>::restoreData(): unknown pixel type!!");
    }
    return in;
}

//! 指定した画素タイプで出力ストリームに画像の画素データを書き出す．
/*!
  \param out	出力ストリーム
  \param type	画素タイプ．ただし， #DEFAULT を指定した場合は，
		この画像オブジェクトの画素タイプで書き出される．
  \return	outで指定した出力ストリーム
*/
template <class T, class ALLOC> std::ostream&
Image<T, ALLOC>::saveData(std::ostream& out, Type type) const
{
    if (type == DEFAULT)
	type = defaultType();

    switch (type)
    {
      case U_CHAR:
	return saveRows<u_char, RGB>(out, type);
      case SHORT:
	return saveRows<short, RGB>(out, type);
      case INT:
	return saveRows<int, RGB>(out, type);
      case FLOAT:
	return saveRows<float, RGB>(out, type);
      case DOUBLE:
	return saveRows<double, RGB>(out, type);
      case RGB_24:
	return saveRows<RGB, RGB>(out, type);
      case YUV_444:
	return saveRows<YUV444, RGB>(out, type);
      case YUV_422:
	return saveRows<YUV422, RGB>(out, type);
      case YUYV_422:
	return saveRows<YUYV422, RGB>(out, type);
      case YUV_411:
	return saveRows<YUV411, RGB>(out, type);
      case BMP_8:
	return saveRows<u_char, BGRA>(out, type);
      case BMP_24:
	return saveRows<BGR, BGRA>(out, type);
      case BMP_32:
	return saveRows<BGRA, BGRA>(out, type);
      default:
	throw std::runtime_error("Image<T, ALLOC>::saveData(): unknown pixel type!!");
    }
    return out;
}

template <class T, class ALLOC> template <class S> std::istream&
Image<T, ALLOC>::restoreRows(std::istream& in, const TypeInfo& typeInfo)
{
    const size_t	npads = type2nbytes(typeInfo.type, true);
    Array<S>		buf(width());
    if (typeInfo.bottomToTop)
    {
	for (auto iter = rbegin(); iter != rend(); ++iter)
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    std::copy(make_pixel_iterator(buf.cbegin()),
		      make_pixel_iterator(buf.cend()),
		      make_pixel_iterator(TU::begin(*iter)));
	}
    }
    else
    {
	for (auto&& row : *this)
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    std::copy(make_pixel_iterator(buf.cbegin()),
		      make_pixel_iterator(buf.cend()),
		      make_pixel_iterator(row.begin()));
	}
    }

    return in;
}

template <class T, class ALLOC> template <class S, class L> std::istream&
Image<T, ALLOC>::restoreAndLookupRows(std::istream& in,
				      const TypeInfo& typeInfo)
{
    Colormap<L>	colormap(typeInfo.ncolors);
    colormap.restore(in);
	
    const size_t	npads = type2nbytes(typeInfo.type, true);
    Array<S>		buf(width());
    if (typeInfo.bottomToTop)
    {
	for (auto iter = rbegin(); iter != rend(); ++iter)
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    std::copy(make_pixel_iterator(TU::make_transform_iterator(
					      colormap, buf.cbegin())),
		      make_pixel_iterator(TU::make_transform_iterator(
					      colormap, buf.cend())),
		      make_pixel_iterator(TU::begin(*iter)));
	}
    }
    else
    {
	for (auto&& row : *this)
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    std::copy(make_pixel_iterator(TU::make_transform_iterator(
					      colormap, buf.cbegin())),
		      make_pixel_iterator(TU::make_transform_iterator(
					      colormap, buf.cend())),
		      make_pixel_iterator(row.begin()));
	}
    }

    return in;
}

template <class T, class ALLOC> template <class D, class L> std::ostream&
Image<T, ALLOC>::saveRows(std::ostream& out, Type type) const
{
    TypeInfo	typeInfo(type);

    Colormap<L>	colormap(typeInfo.ncolors);
    for (size_t i = 0; i < colormap.size(); ++i)
	colormap[i] = i;
    colormap.save(out);
    
    Array<u_char>	pad(type2nbytes(type, true));
    Array<D>		buf(width());
    if (typeInfo.bottomToTop)
    {
	for (auto row = rbegin(); row != rend(); ++row)
	{
	    std::copy(make_pixel_iterator(TU::begin(*row)),
		      make_pixel_iterator(TU::end(*row)),
		      make_pixel_iterator(buf.begin()));
	    if (!buf.save(out) || !pad.save(out))
		break;
	}
    }
    else
    {
	for (auto row = begin(); row != end(); ++row)
	{
	    std::copy(make_pixel_iterator(TU::begin(*row)),
		      make_pixel_iterator(TU::end(*row)),
		      make_pixel_iterator(buf.begin()));
	    if (!buf.save(out) || !pad.save(out))
		break;
	}
    }

    return out;
}

template <class T, class ALLOC> size_t
Image<T, ALLOC>::_width() const
{
    return Image::width();		// Don't call ImageBase::width!
}

template <class T, class ALLOC> size_t
Image<T, ALLOC>::_height() const
{
    return Image::height();		// Don't call ImageBase::height!
}

template <class T, class ALLOC> ImageBase::Type
Image<T, ALLOC>::_defaultType() const
{
    return Image::defaultType();
}

template <class T, class ALLOC> inline ImageBase::Type
Image<T, ALLOC>::defaultType() const
{
    return ImageBase::type2type<T>::value;
}

template <class T, class ALLOC> void
Image<T, ALLOC>::_resize(size_t h, size_t w, const TypeInfo&)
{
    Image<T, ALLOC>::resize(h, w);	// Don't call ImageBase::resize!
}

/************************************************************************
*  class GenericImage							*
************************************************************************/
//! 画素の型を問わない総称画像クラス
/*!
  個々の行や画素にアクセスすることはできない．
*/
class GenericImage : public ImageBase
{
  private:
    using array2_type = Array2<char>;

  public:
    using pointer	= array2_type::pointer;
    using const_pointer	= array2_type::const_pointer;
    
  public:
  //! 総称画像を生成する．
    GenericImage() :_a(), _typeInfo(U_CHAR), _colormap(0)		{}

    pointer		data()						;
    const_pointer	data()					const	;
    const TypeInfo&	typeInfo()				const	;
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out)			const	;
    std::istream&	restoreData(std::istream& in)			;
    std::ostream&	saveData(std::ostream& out)		const	;
    
  private:
    virtual size_t	_width()				const	;
    virtual size_t	_height()				const	;
    virtual Type	_defaultType()				const	;
    virtual void	_resize(size_t h, size_t w,
				const TypeInfo& typeInfo)		;

  private:
    array2_type		_a;
    TypeInfo		_typeInfo;
    Array<BGRA>		_colormap;
};

inline GenericImage::pointer
GenericImage::data()
{
    return _a.data();
}
    
inline GenericImage::const_pointer
GenericImage::data() const
{
    return _a.data();
}
    
//! 現在保持している画像のタイプ情報を返す．
/*!
  \return	タイプ情報
*/
inline const ImageBase::TypeInfo&
GenericImage::typeInfo() const
{
    return _typeInfo;
}

//! 入力ストリームから画像を読み込む．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
inline std::istream&
GenericImage::restore(std::istream& in)
{
    restoreHeader(in);
    return restoreData(in);
}

//! 出力ストリームに画像を書き出す．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
inline std::ostream&
GenericImage::save(std::ostream& out) const
{
    saveHeader(out, _typeInfo.type);
    return saveData(out);
}

}
#endif	// !TU_IMAGEPP_H
