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

% add search path
addpath('C:/smh')
```

## Move data around
```matlab
A = [1 2; 3 4; 5 6]
v = [1 2 3 4]

% Return size of matrix (which itself is a matrix)
az = size(A)
size(az)

% Return size of first dimension
size(A,1)

% Return size of second dimension
size(A,2)

% Return size of longest dimension
length(v)

% Display current path
pwd

% Navigate
cd

% Display files and directories
ls

% Load data from a file
load file.dat
load('file.dat')

% Show variables in current memory
who

% Display file
file.dat

% Show dimension of data
size(featureInFile)

% Return detailed variables in current memory
whos

% Get rid of a feature
clear featureInFile

% Get rid of everything
clear

% Set v to first elements from feature
v = featureInFile(1:10)

% Save data v to file
save hello.mat v;

% Save data v to file as ASCII
save hello.mat v -ascii;

% Element index of matrix
A(3,2)

% Fetch everything in second row
A(2,:)

% Fetch everything in second column
A(:,2)

% Get all elements from first and third row and everything in their columns
A([1 3],:)

% Assign
A(:,2) = [10; 11; 12]

% Append another column vector to right
A = [A, [100; 101; 102]]

% Put all elements of A into a single vector
A(:)

A = [1 2; 3 4; 5 6]
B = [10 11; 12 13; 14 15]

% Puts the matrices together
C = [A B]

% Puts the next matrix to the bottom
C = [A; B]
```

## Compute data
```matlab
A = [1 2; 3 4; 5 6]
B = [10 11; 12 13; 14 15]
C = [1 1; 2 2]

v = [1; 2; 3]

% Element wise operations
A .* B
A ./ B
A .^ B

log(v)

exp(v)

% Returns absolute values
abs(v)

v + ones(length(v), 1) %is same as
v + 1

% Transpose something
A'

% Return max value 
val = max(v)

% Return max value and index
[val, ind] = max(v)

% Element wise comparison
v < 2

% Return index of elements which are bigger than x
find(v < 2)

% Create 3x3 magic matrix with random values
% Magic matrix is where the sum of each column/row  is the same
A = magic(3)

% Return row and index where elements are bigger than x
[r,c] = find(A > 6)

% Add up all elements
sum(v)

% Return product of all elements
prod(v)

% Round down the elements
floor(v)

% Round up the elements
ceil(v)

% Return column wise maximum
max(A,[],1)

% Return row wise maximum
max(A,[],2)

% Return biggest elements
max(max(A)) %or
max(A(:))

% Create magic matrix and check
A = magic(9)
sum(A,1) % Check for sums in columns
sum(A,2) % Check for sums in rows
A .* eye(9) % Wipe everything out besides the diagonal
sum(sum(A.*eye(9))) % Check for sum in diagonal

% Flip matrix
flipud(eye(9))

% Invert matrix
pinv(A)
```

## Plot data
```matlab
t = [0:0.01:0.98]
y1 = sin(2*pi*4*t)
y2 = cos(2*pi*4*t)

% Plot data; horizontal axis being first parameter and vertical being second one
plot(t,y1)

% Plot both functions
hold on;
plot(t,y2)

%  Plot function with different color
plot(t,y2,'r')

% Name horizontal axis
xlabel('time')

% Name vertical axis
ylabel('value')

% Name functions
legend('sin', 'cos')

% Set title of plot
title('cool plot')

% Save plot as PNG
print -dpng 'plot.png'

% Close a plot
close

% Specify figure numbers
figure(1); plot(t,y1)
figure(2); plot(t,y2)

% Divides plot to a 1x2 grid, access n-th element
subplot(1,2,1)
plot(t,y1)
subplot(1,2,2)
plot(t,y2)

% Change axis scales
axis([0.5 1 -1 1])

% Clear figure
clf

% Visualize matrix
A = magic(5)
imagesc(A)
imagesc(A), colorbar, colormap gray

% Entering multiple commands
a = 1, b = 2, c = 3
a = 1; b = 2; c = 3;
```

## Control statements
 * for
 * while
 * if
 
```matlab
v = zeroes(10,1)

% For loop
for i = 1:10, 
  v(i) = 2^i;
end;

% While loop
i = 1;
while i <= 5,
  v(i) = 100;
  i = i+1;
end;

while true,
  v(i) = 999;
  i = i+1;
  if i = 6,
    break;
  end;
end;

% If
v(1) = 2;
if v(1) == 1,
  disp('Value is one');
elseif v(1) == 2,
  disp('Value is two');
else
  disp('Value is neither');
end;
```