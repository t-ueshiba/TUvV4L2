/*
 *  $Id$
 */
#include <boost/iterator/zip_iterator.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

namespace TU
{
namespace detail
{
  /**********************************************************************
  *  tuple_transform(TUPLE, TUPLE, FUNC)				*
  **********************************************************************/
  template <class FUNC> inline boost::tuples::null_type
  tuple_transform(boost::tuples::null_type, boost::tuples::null_type, FUNC)
  {
      return boost::tuples::null_type();
  }
  template <class TUPLE, class FUNC>
  inline typename boost::detail::tuple_impl_specific::
  tuple_meta_transform<TUPLE, FUNC>::type
  tuple_transform(TUPLE const& t1, TUPLE const& t2, FUNC func)
  { 
      typedef typename boost::detail::tuple_impl_specific::
	  tuple_meta_transform<typename TUPLE::tail_type, FUNC>::type
							transformed_tail_type;

      return boost::tuples::cons<
	  typename boost::mpl::apply<FUNC, typename TUPLE::head_type>::type,
	  transformed_tail_type>(func(t1.get_head(), t2.get_head()),
				 tuple_transform(t1.get_tail(),
						 t2.get_tail(), func));
  }

  /**********************************************************************
  *  struct tuple_meta_transform2<TUPLE1, TUPLE2, BINARY_META_FUN>	*
  **********************************************************************/
  template<class TUPLE1, class TUPLE2, class BINARY_META_FUN>
  struct tuple_meta_transform2;
      
  template<class TUPLE1, class TUPLE2, class BINARY_META_FUN>
  struct tuple_meta_transform2_impl
  {
      typedef boost::tuples::cons<
	  typename boost::mpl::apply2<
	      typename boost::mpl::lambda<BINARY_META_FUN>::type,
	      typename TUPLE1::head_type,
	      typename TUPLE2::head_type>::type,
	  typename tuple_meta_transform2<
	      typename TUPLE1::tail_type,
	      typename TUPLE2::tail_type,
	      BINARY_META_FUN>::type>				type;
  };

  template<class TUPLE1, class TUPLE2, class BINARY_META_FUN>
  struct tuple_meta_transform2
      : boost::mpl::eval_if<
	    boost::is_same<TUPLE1, boost::tuples::null_type>,
	    boost::mpl::identity<boost::tuples::null_type>,
	    tuple_meta_transform2_impl<TUPLE1, TUPLE2, BINARY_META_FUN> >
  {
  };

  /**********************************************************************
  *  tuple_transform2<TUPLE1>(TUPLE2, FUNC)				*
  **********************************************************************/
  template <class, class FUNC> inline boost::tuples::null_type
  tuple_transform2(boost::tuples::null_type const&, FUNC)
  {
      return boost::tuples::null_type();
  }
  template <class TUPLE1, class TUPLE2, class FUNC>
  inline typename tuple_meta_transform2<TUPLE1, TUPLE2, FUNC>::type
  tuple_transform2(TUPLE2 const& t, FUNC func)
  { 
      typedef typename tuple_meta_transform2<
	  typename TUPLE1::tail_type,
	  typename TUPLE2::tail_type, FUNC>::type	transformed_tail_type;

      return boost::tuples::cons<
	  typename boost::mpl::apply2<
	      FUNC,
	      typename TUPLE1::head_type,
	      typename TUPLE2::head_type>::type,
	  transformed_tail_type>(
	      func.template operator ()<typename TUPLE1::head_type>(
		  boost::tuples::get<0>(t)),
	      tuple_transform2<typename TUPLE1::tail_type>(
		  t.get_tail(), func));
  }
}	// namespace detail

/************************************************************************
*  struct is_tuple<T>							*
************************************************************************/
//! 与えられた型がtupleまたはconsであるか判定する．
/*!
  \param T	判定対象となる型
*/
template <class T>
struct is_tuple					: boost::mpl::false_	{};
template<class HT, class TT>
struct is_tuple<boost::tuples::cons<HT, TT> >	: boost::mpl::true_	{};
template<BOOST_PP_ENUM_PARAMS(10, class T)>
struct is_tuple<boost::tuple<BOOST_PP_ENUM_PARAMS(10, T)> >
						: boost::mpl::true_	{};

/************************************************************************
*  struct iterator_value<ITER>						*
************************************************************************/
//! 与えられた反復子が指す値の型を返す．
/*!
  zip_iteratorとfast_zip_iteratorのvalue_typeは参照のtupleとして定義されているが，
  本メタ関数は値のtupleを返す．
  \param ITER	反復子の型
*/
template <class ITER>
struct iterator_value
{
    typedef typename std::iterator_traits<ITER>::value_type	type;
};
template <class TUPLE>
struct iterator_value<boost::zip_iterator<TUPLE> >
    : boost::detail::tuple_impl_specific::
      tuple_meta_transform<TUPLE, iterator_value<boost::mpl::_1> >
{
};
template <class TUPLE>
struct iterator_value<fast_zip_iterator<TUPLE> >
    : boost::detail::tuple_impl_specific::
      tuple_meta_transform<TUPLE, iterator_value<boost::mpl::_1> >
{
};

/************************************************************************
*  struct tuple_head<T>							*
************************************************************************/
//! 与えられた型がtupleまたはconsならばその先頭要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その先頭要素の型を調べるべき型
*/
template <class T>
struct tuple_head : boost::mpl::identity<T>
{
};
template <class HT, class TT>
struct tuple_head<boost::tuples::cons<HT, TT> > : boost::mpl::identity<HT>
{
};
template <BOOST_PP_ENUM_PARAMS(10, class T)>
struct tuple_head<boost::tuple<BOOST_PP_ENUM_PARAMS(10, T)> >
    : tuple_head<typename boost::tuple<BOOST_PP_ENUM_PARAMS(10, T)>::inherited>
{
};
    
/************************************************************************
*  struct tuple2cons<S, T>						*
************************************************************************/
//! 与えられた型がtupleまたはconsならばその全要素の型を，そうでなければ元の型自身を別の型で置き換える．
/*!
  Sがboost::tupleであっても帰される型はboost::tuples::consになることに注意．
  \param S	要素型置換の対象となる型
  \param T	置換後の要素の型．voidならば置換しない．
*/
template <class S, class T=void>
struct tuple2cons : boost::mpl::if_<boost::is_same<T, void>, S, T>
{
};
template <class HT, class TT, class T>
struct tuple2cons<boost::tuples::cons<HT, TT>, T>
    : boost::detail::tuple_impl_specific::
      tuple_meta_transform<boost::tuples::cons<HT, TT>,
			   tuple2cons<boost::mpl::_1, T> >
{
};
template <BOOST_PP_ENUM_PARAMS(10, class S), class T>
struct tuple2cons<boost::tuple<BOOST_PP_ENUM_PARAMS(10, S)>, T>
    : tuple2cons<
	  typename boost::tuple<BOOST_PP_ENUM_PARAMS(10, S)>::inherited, T>
{
};

}
