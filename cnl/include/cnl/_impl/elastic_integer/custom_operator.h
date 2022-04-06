
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_ELASTIC_INTEGER_GENERIC_H)
#define CNL_IMPL_ELASTIC_INTEGER_GENERIC_H

#include "../custom_operator/definition.h"
#include "definition.h"

#include <algorithm>

/// compositional numeric library
namespace cnl {
    namespace _impl {
        ////////////////////////////////////////////////////////////////////////////////
        // cnl::_impl::common_signedness

        template<class T1, class T2>
        struct common_signedness {
            static constexpr bool _are_signed =
                    numbers::signedness_v<T1> | numbers::signedness_v<T2>;

            using type = typename std::common_type<
                    numbers::set_signedness_t<T1, _are_signed>,
                    numbers::set_signedness_t<T2, _are_signed>>::type;
        };

        template<class T1, class T2>
        using common_signedness_t = typename common_signedness<T1, T2>::type;

        ////////////////////////////////////////////////////////////////////////////////
        // cnl::_impl::common_elastic_type

        template<typename T1, typename T2>
        struct common_elastic_type;

        template<int Digits1, class Narrowest1, int Digits2, class Narrowest2>
        struct common_elastic_type<
                elastic_integer<Digits1, Narrowest1>, elastic_integer<Digits2, Narrowest2>> {
            using type = elastic_integer<
                    std::max(Digits1, Digits2), common_signedness_t<Narrowest1, Narrowest2>>;
        };

        template<int Digits1, class Narrowest1, class Rhs>
        struct common_elastic_type<elastic_integer<Digits1, Narrowest1>, Rhs>
            : common_elastic_type<
                      elastic_integer<Digits1, Narrowest1>,
                      elastic_integer<numeric_limits<Rhs>::digits, Rhs>> {
        };

        template<class Lhs, int Digits2, class Narrowest2>
        struct common_elastic_type<Lhs, elastic_integer<Digits2, Narrowest2>>
            : common_elastic_type<
                      elastic_integer<numeric_limits<Lhs>::digits, Lhs>,
                      elastic_integer<Digits2, Narrowest2>> {
        };

        template<int FromDigits, class FromNarrowest, int OtherDigits, class OtherNarrowest>
        requires(FromDigits != OtherDigits || !std::is_same<FromNarrowest, OtherNarrowest>::value)
                [[nodiscard]] constexpr auto cast_to_common_type(
                        elastic_integer<FromDigits, FromNarrowest> const& from,
                        elastic_integer<OtherDigits, OtherNarrowest> const&)
        {
            return static_cast<typename common_elastic_type<
                    elastic_integer<FromDigits, FromNarrowest>,
                    elastic_integer<OtherDigits, OtherNarrowest>>::type>(from);
        }
    }

    template<_impl::comparison_op Operator, int LhsDigits, class LhsNarrowest, int RhsDigits, class RhsNarrowest>
    requires(!std::is_same_v<elastic_tag<LhsDigits, LhsNarrowest>, elastic_tag<RhsDigits, RhsNarrowest>>) struct custom_operator<
            Operator,
            op_value<elastic_integer<LhsDigits, LhsNarrowest>>,
            op_value<elastic_integer<RhsDigits, RhsNarrowest>>> {
        [[nodiscard]] constexpr auto operator()(
                elastic_integer<LhsDigits, LhsNarrowest> const& lhs,
                elastic_integer<RhsDigits, RhsNarrowest> const& rhs) const
        {
            return Operator()(cast_to_common_type(lhs, rhs), cast_to_common_type(rhs, lhs));
        }
    };

    // elastic_integer << non-constant
    // elastic_integer >> non-constant
    template<_impl::shift_op Operator, integer LhsRep, int LhsDigits, integer LhsNarrowest, typename Rhs>
    requires(!_impl::is_constant<Rhs>::value) struct custom_operator<
            Operator,
            op_value<_impl::wrapper<LhsRep, elastic_tag<LhsDigits, LhsNarrowest>>>,
            op_value<Rhs>> {
        using lhs_type = _impl::wrapper<LhsRep, elastic_tag<LhsDigits, LhsNarrowest>>;

        [[nodiscard]] constexpr auto operator()(lhs_type const& lhs, Rhs const& rhs) const
        {
            return _impl::from_rep<lhs_type>(Operator{}(_impl::to_rep(lhs), rhs));
        }
    };

    template<int LhsDigits, class LhsNarrowest, CNL_IMPL_CONSTANT_VALUE_TYPE RhsValue>
    struct custom_operator<
            _impl::shift_left_op,
            op_value<elastic_integer<LhsDigits, LhsNarrowest>>,
            op_value<constant<RhsValue>>> {
        [[nodiscard]] constexpr auto operator()(
                elastic_integer<LhsDigits, LhsNarrowest> const& lhs, constant<RhsValue>) const
        {
            return _impl::from_rep<elastic_integer<LhsDigits + int{RhsValue}, LhsNarrowest>>(
                    _impl::to_rep(
                            static_cast<elastic_integer<LhsDigits + int{RhsValue}, LhsNarrowest>>(
                                    lhs))
                    << RhsValue);
        }
    };

    template<integer LhsRep, int LhsDigits, integer LhsNarrowest, CNL_IMPL_CONSTANT_VALUE_TYPE RhsValue>
    struct custom_operator<
            _impl::shift_right_op,
            op_value<_impl::wrapper<LhsRep, elastic_tag<LhsDigits, LhsNarrowest>>>,
            op_value<constant<RhsValue>>> {
        [[nodiscard]] constexpr auto operator()(
                elastic_integer<LhsDigits, LhsNarrowest> const& lhs, constant<RhsValue>) const
        {
            return _impl::from_rep<elastic_integer<LhsDigits - int{RhsValue}, LhsNarrowest>>(
                    _impl::to_rep(static_cast<elastic_integer<LhsDigits, LhsNarrowest>>(lhs))
                    >> RhsValue);
        }
    };

    namespace _impl {
        template<typename T>
        struct tag_narrowest;

        template<int Digits, typename Narrowest>
        struct tag_narrowest<elastic_tag<Digits, Narrowest>> : std::type_identity<Narrowest> {
        };

        template<typename T>
        using tag_narrowest_t = typename tag_narrowest<T>::type;
    }

    // unary +/-
    template<_impl::unary_arithmetic_op Operator, integer RhsRep, int RhsDigits, integer RhsNarrowest>
    requires(!std::is_same_v<_impl::bitwise_not_op, Operator>) struct custom_operator<Operator, op_value<_impl::wrapper<RhsRep, elastic_tag<RhsDigits, RhsNarrowest>>>> {
        using rhs_type = _impl::wrapper<RhsRep, elastic_tag<RhsDigits, RhsNarrowest>>;
        [[nodiscard]] constexpr auto operator()(rhs_type const& rhs) const
        {
            constexpr auto result_digits = digits_v<rhs_type>;
            using rhs_narrowest = _impl::tag_narrowest_t<_impl::tag_of_t<rhs_type>>;
            using result_narrowest = numbers::set_signedness_t<rhs_narrowest, true>;
            using result_type = elastic_integer<result_digits, result_narrowest>;
            return _impl::from_rep<result_type>(Operator{}(_impl::to_rep(static_cast<result_type>(rhs))));
        }
    };

    // unary operator~
    template<integer RhsRep, int RhsDigits, integer RhsNarrowest>
    struct custom_operator<
            _impl::bitwise_not_op, op_value<_impl::wrapper<RhsRep, elastic_tag<RhsDigits, RhsNarrowest>>>> {
        [[nodiscard]] constexpr auto operator()(
                _impl::wrapper<RhsRep, elastic_tag<RhsDigits, RhsNarrowest>> const& rhs)
        {
            using elastic_integer = _impl::wrapper<RhsRep, elastic_tag<RhsDigits, RhsNarrowest>>;
            using rep = _impl::rep_of_t<elastic_integer>;
            return _impl::from_rep<elastic_integer>(static_cast<rep>(
                    _impl::to_rep(rhs)
                    ^ ((static_cast<rep>(~0)) >> (numeric_limits<rep>::digits - RhsDigits))));
        }
    };
}

#endif  // CNL_IMPL_ELASTIC_INTEGER_GENERIC_H
