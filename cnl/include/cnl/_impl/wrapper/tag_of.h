
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_WRAPPER_TAG_OF_H)
#define CNL_IMPL_WRAPPER_TAG_OF_H

#include "../num_traits/tag_of.h"
#include "declaration.h"

#include <type_traits>

/// compositional numeric library
namespace cnl {
    template<typename Rep, tag Tag>
    struct tag_of<_impl::wrapper<Rep, Tag>> : std::type_identity<Tag> {
    };
}

#endif  // CNL_IMPL_WRAPPER_TAG_OF_H
