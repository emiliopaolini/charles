
//          Copyright John McFarlane 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_WIDE_INTEGER_MAX_DIGITS_H)
#define CNL_IMPL_WIDE_INTEGER_MAX_DIGITS_H

#include "../../numeric_limits.h"
#include "../num_traits/max_digits.h"
#include "definition.h"

/// compositional numeric library
namespace cnl {
    namespace _impl {
        template<int Digits, typename Narrowest>
        inline constexpr auto max_digits<wide_integer<Digits, Narrowest>> = numeric_limits<int>::max();
    }
}

#endif  // CNL_IMPL_WIDE_INTEGER_MAX_DIGITS_H
