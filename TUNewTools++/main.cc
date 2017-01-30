/*
 *  $Id$
 */
#include <fstream>
#include <vector>
#ifdef CPP11
#  include "TU/Array++11.h"
#else
#  include "TU/Array++.h"
#endif

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
test_range3(BUF buf)
{
    std::cout << "*** 3D range/array test ***" << std::endl;
    
    size_t	size_x = 6, size_y = 2, size_z;
    auto	a3 = make_dense_range(buf.begin(),
				      buf.size()/size_y/size_x, size_y, size_x);
    std::cout << "--- a3(" << sizes_and_strides(a3) << ") ---\n" << a3
	      << "--- a3[1][0] ---\n" << a3[1][0]
	      << std::endl;

  // buf の一部分を3次元配列と見なす
    size_x = 3;
    size_y = 2;
    size_z = 2;
    auto	b3 = make_range(buf.begin() + 1,
				size_z, size_y, size_y, size_x, 6);
    std::cout << "--- b3(" << sizes_and_strides(b3) << ") ---\n" << b3;

    b3[1][1][2] = 100;
    std::cout << "--- b3(modified) ---\n" << b3;

    Array3<double, 2, 2, 3>	c3(b3);
    std::cout << "--- c3(" << sizes_and_strides(c3) << ") ---\n" << c3;
}

static void
test_stride()
{
    std::cout << "*** stride test ***" << std::endl;
    
  // unit = 8 の2次元配列を生成する
    Array2<int>	c(8, 2, 3);
  //Array2<int, 4, 6>	c;

    fill(c, 5);
    std::cout << "--- c(" << sizes_and_strides(c) << ") ---\n" << c;

    c[1][2] = 10;
    std::cout << c[1] << std::endl;
}
    
static void
test_initializer_list()
{
    std::cout << "*** initializer_list<T> test ***" << std::endl;
    
    Array2<int>	a2({{10, 20, 30},{100, 200, 300}});
    std::cout << "--- a2(" << sizes_and_strides(a2) << ") ---\n" << a2;
}
    
template <class BUF> static void
test_subrange(const BUF& buf)
{
    using value_type	= typename BUF::value_type;
    
    std::cout << "*** subrange test ***" << std::endl;

    auto	r = make_range<2, 2, 2, 3, 6>(buf.begin());
    std::cout << "--- make_range<2, 2, 2, 3, 6>(" << sizes_and_strides(r)
	      << ") ---\n" << r;

    size_t		ncol = 6;
    Array2<value_type>	a2(make_dense_range(buf.begin(), buf.size()/ncol, ncol));
    auto		s2 = make_subrange(a2, 1, 2, 2, 3);
    std::cout << "--- a2 (" << sizes_and_strides(a2) << ") ---\n" << a2
	      << "--- subrange(a2, 1, 2, 2, 3) (" << sizes_and_strides(s2)
	      << ") ---\n" << s2;

    auto		s3 = make_subrange<2, 3>(a2, 1, 2);
    std::cout << "--- subrange<2, 3>(a2, 1, 2) (" << sizes_and_strides(s3)
	      << ") ---\n" << s3;
}

template <class BUF> static void
test_window(const BUF& buf)
{
    using value_type	= typename BUF::value_type;
    
    std::cout << "*** window test ***" << std::endl;

    size_t	ncol = 6;
    const auto	a2 = make_dense_range(buf.begin(), buf.size()/ncol, ncol);
    for (auto iter = a2[0].begin(), end = a2[0].end() - 1; iter != end; ++iter)
	std::cout << make_range<3, 2>(iter, stride<1>(a2));
}

template <class BUF> static void
test_binary_io(const BUF& buf)
{
    using value_type	= typename BUF::value_type;

    std::cout << "*** binary I/O test ***" << std::endl;
    
    size_t		ncol = 6;
    Array2<value_type>	a2(make_dense_range(buf.begin(), buf.size()/ncol, ncol));

    std::ofstream	out("tmp.data", std::ios::binary);
    a2.save(out);
    std::cout << "--- save: a2(" << sizes_and_strides(a2) << ") ---\n" << a2;
    out.close();	// 入力ストリームを開く前にcloseしておくこと

    a2.fill(1000);
    std::cout << "--- modified: a2(" << sizes_and_strides(a2) << ") ---\n" << a2;
    
    std::ifstream	in("tmp.data", std::ios::binary);
    a2.restore(in);
    std::cout << "--- restore: a2(" << sizes_and_strides(a2) << ") ---\n" << a2;
}

template <class BUF> static void
test_text_io(const BUF& buf)
{
    using value_type	= typename BUF::value_type;
    
    std::cout << "*** text I/O test ***" << std::endl;
    
    std::ofstream	out("text1.txt");
    out << make_dense_range(buf.begin(), buf.size());
    out.close();

    std::ifstream	in("text1.txt");
    Array<value_type>	a1;
    in >> a1;
    std::cout << "--- a1(" << sizes_and_strides(a1) << ") ---\n" << a1;
    in.close();
    
    out.open("text2.txt");
    out << make_dense_range(buf.begin(), 8, 3);
    out.close();

    in.open("text2.txt");
    Array2<value_type>	a2;
    in >> a2;
    std::cout << "--- a2(" << sizes_and_strides(a2) << ") ---\n" << a2;
    in.close();

    out.open("text3.txt");
    out << make_dense_range(buf.begin(), 3, 4, 2);
    out.close();

    in.open("text3.txt");
    Array3<value_type>	a3;
    in >> a3;
    std::cout << "--- a3(" << sizes_and_strides(a3) << ") ---\n" << a3;
    in.close();
}

template <class BUF> static void
test_external_allocator(BUF buf)
{
    using	value_type = typename BUF::value_type;
    
    std::cout << "*** external allocator test ***" << std::endl;
    
    Array2<value_type, 0, 0, external_allocator<value_type> >
	a2(buf.data(), buf.size()/6, 6);
  //make_subrange(a2[0], 1, 3) = {1000, 2000, 300};
    make_subrange<2, 3>(a2, 1, 2) = {{100, 200, 300}, {400, 500, 600}};
    std::cout << "--- a2(" << sizes_and_strides(a2) << ") ---\n" << a2;

    std::cout << std::is_convertible<range<range_iterator<int*> >, range<range_iterator<const int*> > >::value << std::endl;
}
    
}	// namespace TU

int
main()
{
    std::vector<int>	buf{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
			    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
			    20, 21, 22, 23};

    TU::test_range3(buf);
    TU::test_stride();
    TU::test_initializer_list();
    TU::test_subrange(buf);
    TU::test_window(buf);
    TU::test_binary_io(buf);
    TU::test_text_io(buf);
    TU::test_external_allocator(buf);
    
    return 0;
}
