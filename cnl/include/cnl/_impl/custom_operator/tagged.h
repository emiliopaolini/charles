
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_OPERATORS_TAGGED_H)
#define CNL_IMPL_OPERATORS_TAGGED_H

#include "../../constant.h"
#include "../config.h"
#include "../custom_operator/tag.h"
#include "definition.h"
#include "op.h"

#include <type_traits>
#include <utility>

/// compositional numeric library
namespace cnl {
    /// \brief converts a value from one type to another
    /// \headerfile cnl/all.h
    ///
    /// \tparam DestTag specifies the destination behavior tag, e.g. \ref native_overflow_tag
    /// \tparam Dest specifies the destination type
    /// \tparam SrcTag specifies the source behavior tag, e.g. \ref native_overflow_tag
    ///
    /// \sa native_overflow_tag, saturated_overflow_tag, throwing_overflow_tag,
    /// trapping_overflow_tag, undefined_overflow_tag, nearest_rounding_tag
    template<tag DestTag, typename Dest, tag SrcTag = _impl::native_tag>
    struct convert {
        template<typename Src>
        [[nodiscard]] constexpr auto operator()(Src const& src) const
        {
            return custom_operator<_impl::convert_op, op_value<Src, SrcTag>, op_value<Dest, DestTag>>{}(src);
        }

        template<CNL_IMPL_CONSTANT_VALUE_TYPE Value>
        [[nodiscard]] constexpr auto operator()(constant<Value> const& src) const
        {
            return custom_operator<_impl::convert_op, op_value<decltype(Value), SrcTag>, op_value<Dest, DestTag>>{}(src);
        }
    };

    namespace _impl {
        template<op Operator, tag Tag = native_tag>
        struct operate {
            template<typename... Operands>
            [[nodiscard]] constexpr auto operator()(Operands const&... operands) const
            {
                return custom_operator<Operator, op_value<Operands, Tag>...>{}(operands...);
            }

            template<typename... Operands>
            constexpr auto operator()(Operands&&... operands) const
            {
                return custom_operator<
                        Operator,
                        op_value<std::remove_cvref_t<Operands>, Tag>...>{}(std::forward<Operands>(operands)...);
            }
        };
    }
}

#endif  // CNL_IMPL_OPERATORS_TAGGED_H
