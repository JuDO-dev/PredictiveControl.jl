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

apply( iter::I, fun::F, period=1 ) where {I, F} = ApplyIterable{I, F}( iter, fun, period )
