/*
 *  $Id$
 */
#include <fstream>
#include <vector>
#include <typeinfo>
#ifdef CPP11
#  include "TU/Array++11.h"
#else
#  include "TU/Array++.h"
#endif
#include "TU/functional.h"
#include <boost/core/demangle.hpp>

namespace TU
{
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
    std::cout << "--- a3(" << print_sizes_and_strides(a3) << ") ---\n" << a3
	      << "--- a3[1][0] ---\n" << a3[1][0]
	      << std::endl;

  // buf の一部分を3次元配列と見なす
    size_x = 3;
    size_y = 2;
    size_z = 2;
    auto	b3 = make_range(buf.begin() + 1,
				size_z, size_y, size_y, size_x, 6);
    std::cout << "--- b3(" << print_sizes_and_strides(b3) << ") ---\n" << b3;

    b3[1][1][2] = 100;
    std::cout << "--- b3(modified) ---\n" << b3;

    Array3<double, 2, 2, 3>	c3(b3);
    std::cout << "--- c3(" << print_sizes_and_strides(c3) << ") ---\n" << c3;
}

static void
test_stride()
{
    std::cout << "*** stride test ***" << std::endl;
    
  // unit = 8 の2次元配列を生成する
    Array2<int>	c(8, 2, 3);
  //Array2<int, 4, 6>	c;

    fill(c, 5);
    std::cout << "--- c(" << print_sizes_and_strides(c) << ") ---\n" << c;

    c[1][2] = 10;
    std::cout << c[1] << std::endl;
}
    
static void
test_initializer_list()
{
    std::cout << "*** initializer_list<T> test ***" << std::endl;
    
    Array2<int>	a2({{10, 20, 30},{100, 200, 300}});
    std::cout << "--- a2(" << print_sizes_and_strides(a2) << ") ---\n" << a2;
}
    
template <class BUF> static void
test_subrange(const BUF& buf)
{
    using value_type	= typename BUF::value_type;
    
    std::cout << "*** subrange test ***" << std::endl;

    auto	r = make_range<2, 2, 2, 3, 6>(buf.begin());
    std::cout << "--- make_range<2, 2, 2, 3, 6>(" << print_sizes_and_strides(r)
	      << ") ---\n" << r;
    using	boost::core::demangle;
    
    size_t		ncol = 6;
    Array2<value_type>	a2(make_dense_range(buf.begin(),
					    buf.size()/ncol, ncol));
    auto		s2 = make_subrange(a2, 1, 2, 2, 3);
    std::cout << "--- a2 (" << print_sizes_and_strides(a2) << ") ---\n" << a2
	      << "--- subrange(a2, 1, 2, 2, 3) (" << print_sizes_and_strides(s2)
	      << ") ---\n" << s2;

    auto		s3 = make_subrange<2, 3>(a2, 1, 2);
    std::cout << "--- subrange<2, 3>(a2, 1, 2) (" << print_sizes_and_strides(s3)
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
    Array2<value_type>	a2(make_dense_range(buf.begin(),
					    buf.size()/ncol, ncol));

    std::ofstream	out("tmp.data", std::ios::binary);
    a2.save(out);
    std::cout << "--- save: a2(" << print_sizes_and_strides(a2) << ") ---\n"
	      << a2;
    out.close();	// 入力ストリームを開く前にcloseしておくこと

    a2.fill(1000);
    std::cout << "--- modified: a2(" << print_sizes_and_strides(a2) << ") ---\n"
	      << a2;
    
    std::ifstream	in("tmp.data", std::ios::binary);
    a2.restore(in);
    std::cout << "--- restore: a2(" << print_sizes_and_strides(a2) << ") ---\n"
	      << a2;
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
    std::cout << "--- a1(" << print_sizes_and_strides(a1) << ") ---\n" << a1;
    in.close();
    
    out.open("text2.txt");
    out << make_dense_range(buf.begin(), 8, 3);
    out.close();

    in.open("text2.txt");
    Array2<value_type>	a2;
    in >> a2;
    std::cout << "--- a2(" << print_sizes_and_strides(a2) << ") ---\n" << a2;
    in.close();

    out.open("text3.txt");
    out << make_dense_range(buf.begin(), 3, 4, 2);
    out.close();

    in.open("text3.txt");
    Array3<value_type>	a3;
    in >> a3;
    std::cout << "--- a3(" << print_sizes_and_strides(a3) << ") ---\n" << a3;
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
    std::cout << "--- a2(" << print_sizes_and_strides(a2) << ") ---\n" << a2;
}

template <class BUF> static void
test_numeric(const BUF& buf)
{
    using value_type	= typename BUF::value_type;

    std::cout << "*** numeric test ***" << std::endl;
    
    auto		a = make_range<2, 6, 6>(buf.begin());
    Array2<value_type>	b{{{100, 110, 120, 130, 140, 150},
			   {160, 170, 180, 190, 200, 210}}};
    Array2<float, 2, 6>	c;
    c = a + 2*b;
    std::cout << "--- c(" << print_sizes_and_strides(c) << ") ---\n" << c;

    auto		d = transpose(c);
    std::cout << "--- d(" << print_sizes(d) << ") ---\n" << d;

    using	boost::core::demangle;
    
    auto	x = a + b + b;
    std::cout << demangle(typeid(result_t<decltype(a)>).name()) << std::endl;
    std::cout << demangle(typeid(result_t<decltype(x)>).name()) << std::endl;
}

static void
test_numeric2()
{
    using value_type = int;
#if 0
    std::cout << "*** numeric test2 ***" << std::endl;
#  if 1
    std::array<std::array<int, 6>, 2>	a{{{0, 1, 2, 3, 4, 5},
					   {6, 7, 8, 9, 10, 11}}};
#  else
    int				a[][6]{{0, 1, 2, 3, 4, 5},
					   {6, 7, 8, 9, 10, 11}};
#  endif
    std::array<std::array<float, 6>, 2>	b{{{100, 110, 120, 130, 140, 150},
					   {160, 170, 180, 190, 200, 210}}};
    Array2<float, 2, 6>	c;

    std::cout << is_range<decltype(a)>::value << std::endl;
    c = a + 2*b;
    std::cout << c;
  //std::cout << "--- c(" << print_sizes_and_strides(c) << ") ---\n" << c;

    auto		d = transpose(a);
    std::cout << "--- d(" << print_sizes(d) << ") ---\n" << d;
#  if 0
    int	x[] = {1, 2, 3}, y[] = {4, 5, 6};
    std::cout << is_range<decltype(x)>::value << std::endl;
    Array<float>	z = x;
    std::cout << "--- z(" << print_sizes(z) << ") ---\n" << z;
#  endif
#endif
}
    
static void
test_prod()
{
    using element_type	= float;
    using vector_type	= Array<element_type>;
    using matrix_type	= Array2<element_type>;

    std::cout << "*** prod test ***" << std::endl;
    
  //const vector_type	a = {1, 2, 3};
    const Array<int, 3>	a = {1, 2, 3};
    const vector_type	b = {4, 5, 6};
    vector_type		c;

    std::cout << "--- t = a * a ---" << std::endl;
    element_type	t = a * a;
    std::cout << t << std::endl << std::endl;

    std::cout << "--- c = a * t ---" << std::endl;
    c = a * t;
    std::cout << c << std::endl;

    const matrix_type	A = {{1, 2, 3}, {10, 20, 30}};
  //const Array2<float, 2, 3>	A = {{1, 2, 3}, {10, 20, 30}};
    const matrix_type	B = {{1, 2}, {10, 20}, {100, 200}};
    matrix_type		C, D;

    std::cout << "--- C = -(A + A + A) ---" << std::endl;
  //C = -(A + A + A);
    c = -b;
    using iter = std::decay<decltype((-b).begin())>::type;
    
    std::cout << boost::core::demangle(typeid(iter).name()) << std::endl;
    std::cout << boost::core::demangle(typeid(std::iterator_traits<iter>::iterator_category).name()) << std::endl;

    C = A + A;
    std::cout << C;

    std::cout << "--- c = A * (a + b) ---" << std::endl;
    c = A * (a + b);
    std::cout << c << std::endl;

    C = B;
    std::cout << "--- c = a * (B + C) ---" << std::endl;
    
    c = a * (B + C);
  //c = a * B;
    std::cout << c << std::endl;

    std::cout << "--- C = A * (B + C) ---" << std::endl;
    C = A * (B + C);
    std::cout << C;

    std::cout << "--- C = (a + b) % c ---" << std::endl;
    C = (a + b) % c;
    std::cout << C;

    std::cout << "--- C = A ^ b ---" << std::endl;
    C = A ^ b;
    std::cout << C;
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
    TU::test_numeric(buf);
    TU::test_numeric2();
    TU::test_prod();
    
    return 0;
}
