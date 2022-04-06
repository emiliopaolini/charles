
//          Copyright John McFarlane 2015 - 2016.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// \brief `cnl::scaled_integer` operators

#if !defined(CNL_IMPL_SCALED_INTEGER_OPERATORS_H)
#define CNL_IMPL_SCALED_INTEGER_OPERATORS_H

#include "../narrow_cast.h"
#include "../scaled/power.h"
#include "definition.h"

#include <numeric>

/// compositional numeric library
namespace cnl {
    ////////////////////////////////////////////////////////////////////////////////
    // heterogeneous operator overloads

    // comparison between operands with different rep and exponent
    template<
            _impl::comparison_op Operator,
            typename LhsRep, int LhsExponent,
            typename RhsRep, int RhsExponent,
            int Radix>
    requires(LhsExponent < RhsExponent) struct custom_operator<
            Operator,
            op_value<scaled_integer<LhsRep, power<LhsExponent, Radix>>>,
            op_value<scaled_integer<RhsRep, power<RhsExponent, Radix>>>> {
        static constexpr int shiftage = RhsExponent - LhsExponent;
        using lhs_type = scaled_integer<LhsRep, power<LhsExponent, Radix>>;
        using rhs_type = scaled_integer<
                decltype(std::declval<RhsRep>() << constant<shiftage>()),
                power<LhsExponent, Radix>>;

        [[nodiscard]] constexpr auto operator()(
                scaled_integer<LhsRep, power<LhsExponent, Radix>> const& lhs,
                scaled_integer<RhsRep, power<RhsExponent, Radix>> const& rhs) const
        {
            return _impl::operate<Operator>{}(lhs_type{lhs}, rhs_type{rhs});
        }
    };

    template<
            _impl::comparison_op Operator,
            typename LhsRep, int LhsExponent,
            typename RhsRep, int RhsExponent,
            int Radix>
    requires(RhsExponent < LhsExponent) struct custom_operator<
            Operator,
            op_value<scaled_integer<LhsRep, power<LhsExponent, Radix>>>,
            op_value<scaled_integer<RhsRep, power<RhsExponent, Radix>>>> {
        static constexpr int shiftage = LhsExponent - RhsExponent;
        using lhs_type = scaled_integer<
                decltype(std::declval<LhsRep>() << constant<shiftage>()),
                power<RhsExponent, Radix>>;
        using rhs_type = scaled_integer<RhsRep, power<RhsExponent, Radix>>;

        [[nodiscard]] constexpr auto operator()(
                scaled_integer<LhsRep, power<LhsExponent, Radix>> const& lhs,
                scaled_integer<RhsRep, power<RhsExponent, Radix>> const& rhs) const
        {
            return _impl::operate<Operator>{}(lhs_type{lhs}, rhs_type{rhs});
        }
    };

    ////////////////////////////////////////////////////////////////////////////////
    // shift operators

    // scaled_integer << constant
    template<typename LhsRep, int LhsExponent, int LhsRadix, CNL_IMPL_CONSTANT_VALUE_TYPE RhsValue>
    struct custom_operator<
            _impl::shift_left_op,
            op_value<scaled_integer<LhsRep, power<LhsExponent, LhsRadix>>>,
            op_value<constant<RhsValue>>> {
        using result_type = scaled_integer<LhsRep, power<LhsExponent + _impl::narrow_cast<int>(RhsValue), LhsRadix>>;
        [[nodiscard]] constexpr auto operator()(
                scaled_integer<LhsRep, power<LhsExponent, LhsRadix>> const& lhs,
                constant<RhsValue>) const
        {
            return _impl::from_rep<result_type>(_impl::to_rep(lhs));
        };
    };

    // scaled_integer >> constant
    template<typename LhsRep, int LhsExponent, int LhsRadix, CNL_IMPL_CONSTANT_VALUE_TYPE RhsValue>
    struct custom_operator<
            _impl::shift_right_op,
            op_value<scaled_integer<LhsRep, power<LhsExponent, LhsRadix>>>,
            op_value<constant<RhsValue>>> {
        using result_type = scaled_integer<LhsRep, power<LhsExponent - _impl::narrow_cast<int>(RhsValue), LhsRadix>>;
        [[nodiscard]] constexpr auto operator()(
                scaled_integer<LhsRep, power<LhsExponent, LhsRadix>> const& lhs,
                constant<RhsValue>) const
        {
            return _impl::from_rep<result_type>(_impl::to_rep(lhs));
        };
    };
}

#endif  // CNL_IMPL_SCALED_INTEGER_OPERATORS_H
