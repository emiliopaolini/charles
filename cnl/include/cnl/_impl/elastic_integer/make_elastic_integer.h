
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_ELASTIC_INTEGER_MAKE_ELASTIC_INTEGER_H)
#define CNL_IMPL_ELASTIC_INTEGER_MAKE_ELASTIC_INTEGER_H

#include "../../constant.h"
#include "../numbers/adopt_signedness.h"
#include "definition.h"
#include "digits.h"

#include <type_traits>

/// compositional numeric library
namespace cnl {
    template<CNL_IMPL_CONSTANT_VALUE_TYPE Value>
    [[nodiscard]] constexpr auto make_elastic_integer(constant<Value>)
            -> elastic_integer<digits_v<constant<Value>>>
    {
        return elastic_integer<digits_v<constant<Value>>>{Value};
    }

    namespace _impl {
        template<integer Rep>
        using narrowest = adopt_signedness_t<int, Rep>;

        template<class Narrowest, class Integral>
        struct make_narrowest {
            using type = Narrowest;
        };

        template<class Integral>
        struct make_narrowest<void, Integral> {
            using type = narrowest<Integral>;
        };

        template<class Narrowest, class Integral>
        using make_narrowest_t = typename make_narrowest<Narrowest, Integral>::type;

        template<class Narrowest, class Integral>
        using make_type =
                elastic_integer<digits_v<Integral>, make_narrowest_t<Narrowest, Integral>>;
    }

    template<class Narrowest = void, class Integral>
    requires(!_impl::is_constant<Integral>::value)
            [[nodiscard]] constexpr auto make_elastic_integer(Integral const& value)
    {
        return _impl::make_type<Narrowest, Integral>{value};
    }
}

#endif  // CNL_IMPL_ELASTIC_INTEGER_MAKE_ELASTIC_INTEGER_H
