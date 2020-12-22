import BlockArrays
import LinearAlgebra

# Test a matrix kron into a matrix
A = ones( Float64, (2, 2) )
B = rand( 2, 2 )

K = PredictiveControl.blockkron( A, B )

@test blocksize( K ) == ( 2, 2 )
@test K[Block(1, 1)] == B
@test K[Block(1, 2)] == B
@test K[Block(2, 1)] == B
@test K[Block(2, 2)] == B


# Test a matrix kron into a vector
A = ones( Float64, (2) )
B = rand( 2, 2 )

K = PredictiveControl.blockkron( A, B )

@test blocksize( K ) == ( 2, 1 )
@test K[Block(1, 1)] == B
@test K[Block(2, 1)] == B

# Test a vector kron into a matrix
A = ones( Float64, (2, 2) )
B = rand(4,1)

K = PredictiveControl.blockkron( A, B )

@test blocksize( K ) == ( 2, 2 )
@test K[Block(1, 1)] == B
@test K[Block(1, 2)] == B
@test K[Block(2, 1)] == B
@test K[Block(2, 2)] == B


# Test a vector kron into a vector
A = ones( Float64, (2) )
B = rand(4,1)

K = PredictiveControl.blockkron( A, B )

@test blocksize( K ) == ( 2, 1 )
@test K[Block(1, 1)] == B
@test K[Block(2, 1)] == B

# Test a vector kron into a vector
A = ones( Float64, (2) )
B = rand(4)

K = PredictiveControl.blockkron( A, B )

@test blocksize( K ) == ( 2, )
@test reshape( K[Block(1, 1)], (:) ) == B
@test reshape( K[Block(2, 1)], (:) ) == B
