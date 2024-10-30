%Yelizaveta Pystina
%Math143M
%Project 2

% Conjugate Gradient Method
H = hilb(20);  %20by20 Hilbert Matrix
b = sum(H, 2); %RHS (vector b) computation accross rows

x0 = [zeros(20, 1)]; %initial guess [00000000000000000000]
r0 = b - H * x0; %calculating residual vector (r0)
v = r0; %v = search direction
iteration = 0; %initialize iteration counter

while norm(r0, inf) >= 1e-3 && iteration < 75
    Hv = H * v; %product of Hilbert Matrix and search direction vector
    a = (r0' * r0) / (v' * Hv);  %optimal step size
    x1 = x0 + a * v;  %update current solution
    r1 = r0 - a * Hv; %update current residual
    
    z = (r1' * r1) / (r0' * r0); %adapt search direction
    v = r1 + z * v;  %updates search direction
    
    x0 = x1; %updated solution
    r0 = r1; %updated residual
    iteration = iteration + 1; %iteration counter
end
%displaying Conjugate Gradient method results
disp('Conjugate Gradient Method:');
disp('Converged Solution (x):');
disp(x0);
disp(['It took the Conjugate Gradient Method ' num2str(iteration) ' iterations']);
disp(' ');

% Jacobi Method
H = hilb(20);   %20by20 Hilbert Matrix
b = sum(H, 2); %RHS (vector b) computation accross rows


x0 = [zeros(20, 1)]; %initial guess [00000000000000000000]
D = diag(diag(H)); %diagonal matrix
L = tril(H, -1); %lower triangular matrix
U = triu(H, 1); %upper triangular matrix
T = inv(D) * (L + U); %transformation matrix
c = inv(D) * b; %constant
iteration = 0; %initialize iteration counter

while norm(H * x0 - b, inf) >= 1e-3 && iteration < 75
    x1 = T * x0 + c;  %update solution
    x0 = x1; %updated solution
    iteration = iteration + 1; %iteration counter
end

%displaying Jacobi method results

disp('Jacobi Method:');
disp('Converged Solution (x):');
disp(x0);
disp(['It took the Jacobi Method ' num2str(iteration) ' iterations']);
