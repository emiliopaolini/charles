
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_WRAPPER_UNARY_OPERATOR_H)
#define CNL_IMPL_WRAPPER_UNARY_OPERATOR_H

#include "../config.h"
#include "../custom_operator/definition.h"
#include "../custom_operator/native_tag.h"
#include "../custom_operator/overloads.h"
#include "definition.h"
#include "from_rep.h"
#include "make_wrapper.h"
#include "to_rep.h"

/// compositional numeric library
namespace cnl {
    template<_impl::unary_arithmetic_op Operator, typename Rep, tag Tag>
    struct custom_operator<Operator, op_value<_impl::wrapper<Rep, Tag>>> {
        [[nodiscard]] constexpr auto operator()(_impl::wrapper<Rep, Tag> const& rhs) const
        {
            return _impl::from_rep<_impl::wrapper<Rep, Tag>>(
                    _impl::operate<Operator, Tag>{}(_impl::to_rep(rhs)));
        }
    };
}

#endif  // CNL_IMPL_WRAPPER_UNARY_OPERATOR_H
