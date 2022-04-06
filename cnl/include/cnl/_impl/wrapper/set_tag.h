
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_WRAPPER_SET_TAG_H)
#define CNL_IMPL_WRAPPER_SET_TAG_H

#include "../num_traits/set_tag.h"
#include "declaration.h"

#include <type_traits>

/// compositional numeric library
namespace cnl {
    template<typename Rep, class Tag, class OutTag>
    struct set_tag<_impl::wrapper<Rep, Tag>, OutTag>
        : std::type_identity<_impl::wrapper<Rep, OutTag>> {
    };
}

#endif  // CNL_IMPL_WRAPPER_SET_TAG_H
