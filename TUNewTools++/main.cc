/*
 *  $Id$
 */
#include <vector>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  manipulator sizes_and_strides					*
************************************************************************/
template <class E>
class sizes_and_strides_holder
{
  public:
			sizes_and_strides_holder(const E& expr)	:_expr(expr) {}

    std::ostream&	operator ()(std::ostream& out) const
			{
			    return print_stride(print_size(out, _expr) << ':',
						_expr.begin());
			}

  private:
    template <class E_>
    static typename std::enable_if<!is_range<E_>::value, std::ostream&>::type
			print_x(std::ostream& out, const E_& expr)
			{
			    return out;
			}
    template <class E_>
    static typename std::enable_if<is_range<E_>::value, std::ostream&>::type
			print_x(std::ostream& out, const E_& expr)
			{
			    return out << 'x';
			}
    template <class E_>
    static typename std::enable_if<!is_range<E_>::value, std::ostream&>::type
			print_size(std::ostream& out, const E_& expr)
			{
			    return out;
			}
    template <class E_>
    static typename std::enable_if<is_range<E_>::value, std::ostream&>::type
			print_size(std::ostream& out, const E_& expr)
			{
			    return print_size(print_x(out << std::size(expr),
						      *expr.begin()),
					      *expr.begin());
			}
    template <class ITER_>
    static typename std::enable_if<!is_range<
				       typename std::iterator_traits<ITER_>
				       ::value_type>::value,
				   std::ostream&>::type
			print_stride(std::ostream& out, const ITER_& iter)
			{
			    return out;
			}
    template <class ITER_>
    static typename std::enable_if<is_range<
				       typename std::iterator_traits<ITER_>
				       ::value_type>::value,
				   std::ostream&>::type
			print_stride(std::ostream& out, const ITER_& iter)
			{
			    return print_stride(print_x(out << iter.stride(),
							*iter->begin()),
						iter->begin());
			}

  private:
    const E&	_expr;
};

template <class E> sizes_and_strides_holder<E>
sizes_and_strides(const E& expr)
{
    return sizes_and_strides_holder<E>(expr);
}

template <class E> std::ostream&
operator <<(std::ostream& out, const sizes_and_strides_holder<E>& holder)
{
    return holder(out);
}

template <size_t NROW, size_t NCOL>
struct WindowGenerator
{
};
    
/************************************************************************
*  static functions							*
************************************************************************/
template <class BUF> static void
doJob(BUF& buf)
{
    size_t	size_x, size_y, size_z;

  // buf を2次元行列と見なす
    size_x = 6;
    auto	a2 = make_dense_range(buf.begin(), buf.size()/size_x, size_x);
    std::cout << "--- a2(" << sizes_and_strides(a2) << ") ---\n" << a2
	      << std::endl;

  // buf を3次元行列と見なす
    size_x = 2;
    size_y = 3;
    auto	a3 = make_dense_range(buf.begin(),
				      buf.size()/size_y/size_x, size_y, size_x);
    std::cout << "--- a3 (" << sizes_and_strides(a3) << ") ---\n"
	      << a3
	      << "--- a3[1][2] ---\n" << a3[1][2]
	      << std::endl;

    a3[1][2] = a3[2][1];
  //std::cout << a3;
    
  // buf の一部分を3次元行列と見なす
    size_x = 3;
    size_y = 2;
    size_z = 2;
    auto	b3 = make_range(buf.begin() + 1,
				size_z, size_y, size_y, size_x, 6);
    std::cout << "--- b3(" << sizes_and_strides(b3) << ") ---\n" << b3;

    b3[1][1][2] = 100;
    std::cout << "--- b3(modified) ---\n" << b3;

  // stride = 4 の2次元行列を生成する
    Array2<int>	c(4, 2, 3);
    std::cout << "--- c(" << sizes_and_strides(c) << ") ---\n" << c;
    c[1][2] = 10;

    std::cout << c[1] << std::endl;

  // 2次元行列を複製する
  //new_array<2, BUF>	d;
    Array2<int>	d;
    d = c;
    std::cout << "--- d(" << sizes_and_strides(d) << ") ---\n" << d;

    for (auto iter = c.rbegin(); iter != c.rend(); ++iter)
	std::cout << *iter;

    Array3<float>	e(b3);
    std::cout << "--- e(" << sizes_and_strides(e) << ") ---\n" << e;

    auto	f = make_range<2, 2, 2, 3, 6>(buf.begin());
    std::cout << "--- f(" << sizes_and_strides(f) << ") ---\n" << f;

    auto	g = make_subrange(a2, 1, 2, 2, 3);
    std::cout << "--- a2 (" << sizes_and_strides(a2) << ") ---\n" << a2
	      << "--- subrange(a2, 1, 2, 2, 3) (" << sizes_and_strides(g)
	      << ") ---\n" << g;

    auto	h = make_subrange<2, 3>(a2, 1, 2);
    std::cout << "--- a2 (" << sizes_and_strides(a2) << ") ---\n" << a2
	      << "--- subrange(a2, 1, 2, 2, 3) (" << sizes_and_strides(h)
	      << ") ---\n" << h;

    for (auto iter = a2[0].begin(), end = a2[0].end() - 1; iter != end; ++iter)
	std::cout << make_range<3, 2>(iter, stride<1>(a2));
}

}	// namespace TU

int
main()
{
    std::vector<int>	a{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
			  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
			  20, 21, 22, 23};

    TU::doJob(a);
    
    return 0;
}
