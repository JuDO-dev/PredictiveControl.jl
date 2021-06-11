"""
    loop( iter )

Loop over `iter` until it finishes.
"""
function loop( iter )
    x = nothing

    n = 0
    for y in iter
        n = n+1
        x = y
    end

    return (x, n)
end
