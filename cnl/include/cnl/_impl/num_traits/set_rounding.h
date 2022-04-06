
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_NUM_TRAITS_SET_ROUNDING_H)
#define CNL_IMPL_NUM_TRAITS_SET_ROUNDING_H

#include "../type_traits/remove_cvref.h"
#include "is_composite.h"
#include "rounding.h"

#include <type_traits>

namespace cnl {
    /// \brief given a numeric type, defines member `type` as the equivalent type with the given
    /// rounding mode \headerfile cnl/num_traits.h \note User-specializations of this type are
    /// permitted. \note Native numeric types are only convertible to \ref native_rounding_tag.
    /// \sa rounding, set_rounding_t, native_rounding_tag, nearest_rounding_tag
    template<typename Number, rounding_tag RoundingTag>
    struct set_rounding;

    template<typename Number>
    requires(!is_composite_v<Number>) struct set_rounding<Number, rounding_t<Number>>
        : std::type_identity<Number> {
    };

    template<typename Number, rounding_tag RoundingTag>
    struct set_rounding<Number const&, RoundingTag>
        : set_rounding<_impl::remove_cvref_t<Number>, RoundingTag> {
    };

    template<typename Number, rounding_tag RoundingTag>
    struct set_rounding<Number&, RoundingTag> : set_rounding<Number, RoundingTag> {
    };

    template<typename Number, rounding_tag RoundingTag>
    struct set_rounding<Number&&, RoundingTag> : set_rounding<Number, RoundingTag> {
    };

    /// \brief helper alias of \ref set_rounding
    /// \headerfile cnl/num_traits.h
    /// \sa set_rounding, rounding_t, native_rounding_tag, nearest_rounding_tag
    template<typename Number, rounding_tag RoundingTag>
    using set_rounding_t = typename set_rounding<Number, RoundingTag>::type;
}

#endif  // CNL_IMPL_NUM_TRAITS_SET_ROUNDING_H
