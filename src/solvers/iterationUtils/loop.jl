"""
    loop( iter )

Loop over `iter` until it finishes.
"""
function loop( iter )
    x = nothing

    for y in iter
        x = y
    end

    return x
end
