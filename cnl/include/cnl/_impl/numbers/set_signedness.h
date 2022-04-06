
//          Copyright John McFarlane 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_NUMBERS_SET_SIGNEDNESS_H)
#define CNL_IMPL_NUMBERS_SET_SIGNEDNESS_H

#include "../../numeric_limits.h"
#include "../config.h"
#include "../cstdint/types.h"

#include <type_traits>

/// compositional numeric library, numbers header/namespace
namespace cnl::numbers {
    template<class T, bool IsSigned>
    struct set_signedness;

    template<typename T>
    requires(std::is_integral_v<T>) struct set_signedness<T, false>
        : std::make_unsigned<T> {
    };
    template<typename T>
    requires(std::is_integral_v<T>) struct set_signedness<T, true>
        : std::make_signed<T> {
    };

#if defined(CNL_INT128_ENABLED)
    template<>
    struct set_signedness<int128, false> : std::type_identity<uint128> {
    };
    template<>
    struct set_signedness<int128, true> : std::type_identity<int128> {
    };

    template<>
    struct set_signedness<uint128, false> : std::type_identity<uint128> {
    };
    template<>
    struct set_signedness<uint128, true> : std::type_identity<int128> {
    };
#endif

    template<class T, bool IsSigned>
    using set_signedness_t = typename set_signedness<T, IsSigned>::type;
}

#endif  // CNL_IMPL_NUMBERS_SET_SIGNEDNESS_H
