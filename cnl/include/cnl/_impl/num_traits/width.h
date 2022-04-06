
//          Copyright John McFarlane 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_TYPE_TRAITS_WIDTH_H)
#define CNL_IMPL_TYPE_TRAITS_WIDTH_H

#include "../numbers/signedness.h"
#include "digits.h"

namespace cnl {
    namespace _impl {
        template<typename T>
        inline constexpr int width = digits_v<T> + numbers::signedness_v<T>;
    }
}

#endif  // CNL_IMPL_TYPE_TRAITS_WIDTH_H
