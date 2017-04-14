/*
 *  $Id$
 */
#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include "TU/DP.h"

namespace TU
{
/************************************************************************
*  class dummy_iterator<T>						*
************************************************************************/
template <class ITER>
class dummy_iterator
    : public boost::iterator_adaptor<dummy_iterator<ITER>, ITER>
{
  private:
    typedef boost::iterator_adaptor<dummy_iterator, ITER>	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    dummy_iterator(const ITER& iter, const value_type& domain)
	:super(iter), _domain(domain)			{}

  private:
    reference	dereference() const			{return _domain;}

  private:
    const value_type&	_domain;
};

/************************************************************************
*  class Energy<T>							*
************************************************************************/
template <class T>
class Energy
{
  public:
    typedef int					argument_type;
    typedef T					result_type;
    typedef Array<result_type>			array_type;
    typedef Array2<result_type>			array2_type;
    
    class Generator
    {
      public:
	typedef array_type	argument_type;
	typedef Energy		result_type;

      public:
	Generator(const array2_type& g)	:_g(g)				{}
	
	result_type	operator ()(const argument_type& f) const
			{
			    return result_type(f, _g);
			}

      private:
	const array2_type&	_g;
    };
    
  public:
    Energy(const array_type& f, const array2_type& g) :_f(f), _g(g)	{}

    result_type	operator()(argument_type x) const
		{
		    return _f[x];
		}
    result_type	operator()(argument_type x, argument_type y) const
		{
		    return _g[x][y];
		}
	
  private:
    const array_type&	_f;
    const array2_type&	_g;
};
    
}

/************************************************************************
*  global functions							*
************************************************************************/
int
main()
{
    using namespace	std;
    using namespace	TU;

    typedef int					value_type;
    typedef Energy<value_type>::array2_type	array2_type;
    typedef Array<u_int>			domain_type;
    typedef dummy_iterator<const domain_type*>	domain_iterator;
    typedef DP<domain_iterator, value_type>	dp_type;
    typedef Energy<value_type>::Generator	generator_type;
    
    try
    {
	array2_type	f, g;
	cin >> f >> g;
	
	if ((g.nrow() != f.ncol()) || (g.ncol() != f.ncol()))
	    throw runtime_error("Corruputed input arrays!");
	
	domain_type	domain(f.ncol());
	for (u_int i = 0; i < domain.size(); ++i)
	    domain[i] = i;

	dp_type		dp;
	dp.initialize(domain_iterator(&domain, domain),
		      domain_iterator(&domain + f.nrow(), domain));

	domain_type	x(f.nrow());
	value_type	val = dp(boost::make_transform_iterator(
				     f.begin(), generator_type(g)),
				 x.rbegin());
	dp.put(cout);
	
	cerr << "Seq = " << x << "Val = " << val << endl;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
