# Helper types to represent various combinations of vectors/matrices/scalings
const AbstractNumOrVector  = Union{Number, AbstractVector}
const AbstractNumOrMatrix  = Union{Number, AbstractVector, AbstractMatrix}
const AbstractNumOrUniform = Union{Number, AbstractVector, AbstractMatrix, UniformScaling}

const AbstractNumOrMatrixOrSymbol  = Union{Number, AbstractVector, AbstractMatrix, Symbol}
const AbstractNumOrUniformOrSymbol = Union{Number, AbstractVector, AbstractMatrix, UniformScaling, Symbol}

# Helper type to represent a Hermitian view of a block matrix
const HermitianBlockMatrix = Hermitian{T, M} where {T <: Number, M <: BlockMatrix{<:T}}

# Helper functions to convert various types to a 2d matrix
to_matrix(T, A::AbstractVector) = Matrix{T}(reshape(A, length(A), 1))
to_matrix(T, A::AbstractMatrix) = T.(A)  # Fallback
to_matrix(T, A::Number) = fill(T(A), 1, 1)
to_matrix(T, A::UniformScaling, s::Integer) = Matrix{T}(A*I, s, s)


function dftmatrix( n::Integer; unitary::Bool = false )

    M = fft( 1.0*I(n), 1 )

    # A scaling of 1/(√n) needs to be applied to make the DFT matrix unitary
    if( unitary )
        M = 1 / sqrt( n ) * M
    end

    return M
end
