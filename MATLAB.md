# MATLAB
Helpful things in MATLAB.

## Basic operations
```matlab
% add
5+6

% subtract
3-2

% multiply
5*8

% divide
1/2

% power of
2^6


% false
1 == 2

% true
1 ~= 2

% AND
1 && 0

% OR
1 || 0

% XOR
xor(1,0)


% change promt
PS('>> ');

% assign variable - ; supressing output
a = 3;
b = 'b';
c = (3 >= 1);

a = pi;

% display variable
disp(a);

% display variable with limited decimals
disp(sprintf('2 decimals: %0.2f', a))

% format variable
format short 
format long


% create matrix
A = [1,2; 3,4; 5,6]

% create vector
a = [1; 2; 3;]

% create column vector
a = [1 2 3]

% assign random vector (from 1 with steps of 0.1 to 2)
a = 1:0.1:2

% matrix with ones
ones(2, 3)
2 * ones(2, 3)

% matrix with zeros
zeros(1, 3)

% matrix with random numbers
rand(3, 3)

% histogram
hist(a)
hist(a, 50)

% identity matrix (e.g 4x4 matrix)
I = eye(4)

% help for commands
help(eye)
```