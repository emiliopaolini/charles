
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_NUM_TRAITS_REP_OF_H)
#define CNL_IMPL_NUM_TRAITS_REP_OF_H

/// compositional numeric library
namespace cnl {
    /// \brief meta-function object that returns the Rep type of a component
    ///
    /// \tparam T component whose representational type to return
    ///
    /// \note The representational or 'Rep' type of a numeric component is central to CNL's method
    /// of encapsulation and abstraction. Typically, this is the type of the member variable in many
    /// of the numeric types. Many of the properties of a component -- such as precision, range and
    /// signedness -- will be established by the Rep type.
    ///
    /// \sa to_rep, from_value, tag_of, set_rep
    template<typename T>
    struct rep_of;

    namespace _impl {
        template<typename T>
        using rep_of_t = typename rep_of<T>::type;
    }
}

#endif  // CNL_IMPL_NUM_TRAITS_REP_OF_H
