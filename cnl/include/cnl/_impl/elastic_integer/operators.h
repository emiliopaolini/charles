
//          Copyright John McFarlane 2019.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file ../LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if !defined(CNL_IMPL_ELASTIC_INTEGER_OPERATORS_H)
#define CNL_IMPL_ELASTIC_INTEGER_OPERATORS_H

#include "../config.h"
#include "../num_traits/to_rep.h"
#include "definition.h"

#if defined(CNL_IOSTREAMS_ENABLED)
#include <ostream>
#endif

/// compositional numeric library
namespace cnl {
#if defined(CNL_IOSTREAMS_ENABLED)
    template<int Digits, class Narrowest>
    auto& operator<<(std::ostream& o, elastic_integer<Digits, Narrowest> const& i)
    {
        return o << _impl::to_rep(i);
    }
#endif
}

#endif  // CNL_IMPL_ELASTIC_INTEGER_OPERATORS_H
