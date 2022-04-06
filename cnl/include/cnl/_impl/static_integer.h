
//          Copyright John McFarlane 2015 - 2018.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_STATIC_INTEGER_H)
#define CNL_IMPL_STATIC_INTEGER_H

#include "../elastic_integer.h"
#include "../overflow_integer.h"
#include "../rounding_integer.h"
#include "../wide_integer.h"
#include "num_traits/digits.h"

/// compositional numeric library
namespace cnl {
    namespace _impl {
        template<
                int Digits = digits_v<int>, rounding_tag RoundingTag = nearest_rounding_tag,
                overflow_tag OverflowTag = undefined_overflow_tag, class Narrowest = int>
        using static_integer = overflow_integer<
                elastic_integer<
                        Digits,
                        rounding_integer<
                                wide_integer<digits_v<Narrowest>, Narrowest>, RoundingTag>>,
                OverflowTag>;

        template<
                rounding_tag RoundingTag = nearest_rounding_tag,
                overflow_tag OverflowTag = undefined_overflow_tag, class Narrowest = int,
                class Input = int>
        requires(!_impl::is_constant<Input>::value)
                [[nodiscard]] constexpr auto make_static_integer(Input const& input)
                        -> static_integer<
                                numeric_limits<Input>::digits, RoundingTag, OverflowTag,
                                Narrowest>
        {
            return input;
        }

        template<
                rounding_tag RoundingTag = nearest_rounding_tag,
                overflow_tag OverflowTag = undefined_overflow_tag, class Narrowest = int,
                CNL_IMPL_CONSTANT_VALUE_TYPE InputValue = 0>
        [[nodiscard]] constexpr auto make_static_integer(constant<InputValue>)
        {
            return static_integer<used_digits(InputValue), RoundingTag, OverflowTag, Narrowest>{InputValue};
        }
    }
}

#endif  // CNL_IMPL_STATIC_INTEGER_H
