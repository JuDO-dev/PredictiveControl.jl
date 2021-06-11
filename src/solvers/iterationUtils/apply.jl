struct ApplyIterable{I, F}
    iter::I
    fun::F
    period::UInt
end

function iterate( iter::ApplyIterable, args... )
    next = iterate( iter.iter, args... )

    if ( next != nothing ) && ( ( next[1][1] % iter.period ) == 0 )
        iter.fun( next[1][2] )
    end

    return next
end

"""
```julia
    apply( iter::I, fun::F, period=1 ) where {I, F}
```

Apply the function `fun` to the iterator `iter` at every `period` samples.
`fun` should accept a single argument, which is the state variable for the iterator
`iter`.
"""
apply( iter::I, fun::F, period=1 ) where {I, F} = ApplyIterable{I, F}( iter, fun, period )
