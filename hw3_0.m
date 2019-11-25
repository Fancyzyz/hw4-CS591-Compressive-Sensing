% X: 256 * 1
% 5-sparse, S = 5
% generated a sparse signal, of sparsity 5, 
% randomly choose 5 indexes using randperm, and assign the 5 random indeces
% with value randn(5, 1)

% load("../ps2/ps2_2018.mat")

N = 256;
S = 5;
x = zeros(N, 1);
q = randperm(N);
x(q(1:S)) = randn(S, 1);
epsilon = 10e-10;
M = 50;
sparsity = 5;
countA = [];
countB = [];
countC = [];


alpha = zeros(N, 1);
q = randperm(N);
alpha(q(1:S)) = randn(S, 1);
x = idct(alpha);
A = generateTimeDomainSensingMatrix(M, N);

y=A*(x);
[residue1,x_hat] = OrthogonalMatchingPursuit(A, y, 5, epsilon);
x_hat=(x_hat);



for M=10:10:50
    A = generateTimeDomainSensingMatrix(M, N);
    B = generateFreqDomainSensingMatrix(M, N);
    C = generateRandomGaussianOrthonormalizedMatrix(M, N);
    c1=0;
    c2=0;
    c3=0;
    for i=1:50
        alpha = zeros(N, 1);
        q = randperm(N);
        alpha(q(1:S)) = randn(S, 1);
        x = idct(alpha);
        
        
        
        y = A * dct(x);
        [residue1,x_hat] = OrthogonalMatchingPursuit(A, y, sparsity, epsilon);
        x_hat=idct(x_hat);
%         [residue1,x_hat] = OrthogonalMatchingPursuit(B, y, sparsity, epsilon);
        if norm(x-(x_hat),2)<epsilon
            c1=c1+1;
        end
        y = B * dct(x);
        [residue2,x_hat] = OrthogonalMatchingPursuit(B, y, sparsity, epsilon);
        x_hat=idct(x_hat);
        if norm(x-(x_hat),2)<epsilon
            c2=c2+1;
        end    
        y = C * dct(x);
        [residue3,x_hat] = OrthogonalMatchingPursuit(C, y, sparsity, epsilon);
        x_hat=idct(x_hat);
        if norm(x-(x_hat),2)<epsilon
            c3=c3+1;
            
        end   
    end
    countA = [countA,c1/50];
    countB = [countB,c2/50];
    countC = [countC,c3/50];
end
X=10:10:50;
% y=[countOrth,countSub];
disp(countA);
disp(countB);
disp(countC);
plot(X,countA,X,countB,X,countC);
title('Compare 3 differnet Sensing Matrices');
xlabel('M,numbers of measurements');
ylabel('Probability of Perfect Recovery');

% Random sampling in the time domain: Suppose I is the N by N identity matrix. 
% Create the sensing matrix A by keeping M rows of I at random locations 
% (and deleting the remaining M ? N rows). ?
function [A] = generateTimeDomainSensingMatrix(m, n)
    I = eye(n);
    randIndeces = randperm(n);
    A = I(randIndeces(1:m), :);
end


% (Random sampling in the frequency domain: Suppose F is the N by N DCT matrix 
% (F = dct(eye(N));). Create the sensing matrix A by keeping M rows of F at random locations
% (and deleting the remaining M-N rows). ?
function [A] = generateFreqDomainSensingMatrix(m, n)
    F = dct(eye(n));
    randIndeces = randperm(n);
    A = F(randIndeces(1:m), :);
end


% Sampling with a random matrix: The sensing matrix A is M by N in this case 
% and is generated from a collection of random Gaussian variables, 
% then the rows are orthonor- malized, i.e.,
% A = randn(M, N); A = orth(A?)?
function [A] = generateRandomGaussianOrthonormalizedMatrix(m, n)
    A = randn(m, n);
    A = orth(A')';
end



function [residue_new, x_hat] = OrthogonalMatchingPursuit(A, y, sparsity, epsilon)
    % find best x_hat, such that minimizes ||(A* x_hat - y)||_2 < epsilon
    % y = A*x
    % x = pseudoInverse(A)*y

    [~, xLen] = size(A);
    x_hat = zeros(xLen, 1);
    residue_prev = y;
    residue_new = residue_prev;
    S = [];
    %AInv = pinv(A);
    ATranspose = A';
    itera = 0;
    residues = [];
    
%     while itera < sparsity
    while true
        itera = itera + 1;
        residue_prev = residue_new;
        

        % set the columns from A, that gives the argmax of residue*A to be
        % 0, so that the new argmax cannot choose previously chosen n.
        % Because it is calculating the covariance.
        ATranspose(S, :) = 0;
        [~ ,n] = max(abs(ATranspose * residue_prev));
        % [~ ,n] = max(ATranspose * residue_prev);

        S = [S, n];
        
        temp = zeros(size(A));
        temp(:, S) = A(:, S);
        ADaggerS = pinv(temp);
        
        x_hat = ADaggerS*y;
        residue_new = y - A*x_hat;
        
        residues = [residues, norm(residue_new, 2)];
        
        if norm(residue_new, 2) >= norm(residue_prev, 2)
             
             sprintf("Residue not decreasing, terminating. Iteration: %d, x_hat l0 norm: %d", itera, sum(x_hat ~= 0))
             break
        end
        if norm(residue_new, 2) < epsilon
            sprintf("Residue smaller than epsilon, terminating.. Iteration: %d, x_hat l0 norm: %d", itera, sum(x_hat~= 0))
            break
        end
    end
end



function [residue_new, x_hat] = SubspacePursuit(A, y, sparsity, epsilon)
    % find best x_hat, such that minimizes ||(A* x_hat - y)||_2 < epsilon
    % y = A*x
    % x = pseudoInverse(A)*y

    [~, xLen] = size(A);
    residue_prev = y;
    residue_new = residue_prev;
    S = [];

    ATranspose = A';
    itera = 0;
    residues = [];
    
    while true
        S_prev = S;
        x_hat = zeros(xLen, 1);

        itera = itera + 1;
        residue_prev = residue_new;
   

        % set the columns from A, that gives the argmax of residue*A to be
        % 0, so that the new argmax cannot choose previously chosen n.
        % Because it is calculating the covariance.
        ATransposeTemp = ATranspose;
        ATransposeTemp(S, :) = 0;
        
        
        % find new set of candidates that maximizes the covariance
        [ci ,ciIndeces] = maxk(abs(ATransposeTemp * residue_prev), 5);
        % add new candidate indeces to S set
        
        S = [S, ciIndeces];
        S = S(:);
     
        % pruning for the new best S set
        temp = zeros(size(A));
        temp(:, S) = A(:, S);
        ADaggerS = pinv(temp);
        
        % x_hat = ADaggerS*y;
        [~ ,x_hatIndex] = maxk(abs(ADaggerS*y), 5);
        tempDagger = ADaggerS*y;
        x_hat_val = tempDagger(x_hatIndex);
        S = x_hatIndex;
        S = S(:);
       

        x_hat(x_hatIndex, 1) = x_hat_val;
        
        
        
        residue_new = y - A*x_hat;

        residues = [residues, norm(residue_new, 2)];
        
       

        if (size(S_prev) == size(S)) & (size(intersect(S,S_prev))==[5 1])
            sprintf("S set not changing, terminating. Iteration: %d, x_hat l0 norm: %d", itera, sum(x_hat ~= 0))
            break
        end
        if norm(residue_new, 2) >= norm(residue_prev, 2)
            sprintf("Residue not decreasing, terminating. Iteration: %d, x_hat l0 norm: %d", itera, sum(x_hat ~= 0))
            break
        end
        if norm(residue_new, 2) < epsilon
            sprintf("Residue smaller than epsilon, terminating.. Iteration: %d, x_hat l0 norm: %d", itera, sum(x_hat~= 0))
            break
        end
    end
end







% function x_hat = MatchingPursuit(A, y, epsilon)
%     % find best x_hat, such that minimizes ||(A* x_hat - y)||_2 < epsilon
%     % y = A*x
%     % x = pseudoInverse(A)*y
% 
%     [yLen, xLen] = size(A);
%     x_hat = zeros(xLen, 1);
%     residue_prev = y;
%     residue_new = residue_prev;
%     S = [];
%     AInv = pinv(A);
%     ATranspose = A';
%     itera = 0;
%     
%     while true
%         itera = itera + 1;
%         residue_prev = residue_new;
%         sprintf("residue_new: %f, residue_prev: %f\n", norm(residue_new, 2), norm(residue_prev, 2))
% 
%         % set the columns from A, that gives the argmax of residue*A to be
%         % 0, so that the new argmax cannot choose previously chosen n.
%         % Because it is calculating the covariance.
%         ATranspose(S, :) = 0;
%         [~ ,n] = max(abs(ATranspose * residue_prev));
%         S = [S, n];
%         ADaggerS = zeros(size(AInv));
%         ADaggerS(S, :) = AInv(S, :);
%         x_hat = ADaggerS*y;
%         residue_new = y - A*x_hat;
%         
%     
%         sprintf("residue_new: %f, residue_prev: %f\n", norm(residue_new, 2), norm(residue_prev, 2))
% 
%         if norm(residue_new, 2) >= norm(residue_prev, 2)
%             sprintf("Residue not decreasing, terminating. Iteration: %d, x_hat l0 norm: %d", itera, sum(x_hat >0))
%             break
%         end
%         if norm(residue_new, 2) < epsilon
%             sprintf("Residue smaller than epsilon, terminating.. Iteration: %d, x_hat l0 norm: %d", itera, sum(x_hat> 0))
%             disp("")
%         end
%     end
% end





% 
%  
%  allAfCombinationIndeces = nchoosek(1:100, S);
%  
%  s = size(allAfCombinationIndeces);
%  s = s(1);
%  
%  for i = 1:s
%      rank1 = rank(Af(:, allAfCombinationIndeces(i, :)));
%      rank2 = rank(cat(2, Af(:, allAfCombinationIndeces(i, :)), yf));
%      if rank1 == rank2
%          Xf = linsolve(Af(:, allAfCombinationIndeces(i, :)),yf);
%          ASub = Af(:, allAfCombinationIndeces(i, :));
%          disp("Xf non-zero position: ")
%          disp(allAfCombinationIndeces(i, :))
%          disp("Xf non-zero value: ")
%          disp(Xf)
%      end
%  end
%  
%  for j = 1:s
%      rank1 = rank(Ar(:, allAfCombinationIndeces(j, :)));
%      rank2 = rank(cat(2, Ar(:, allAfCombinationIndeces(j, :)), yr));
%      if rank1 == rank2
%          Xr = linsolve(Ar(:, allAfCombinationIndeces(j, :)),yr);
%          ASub = Ar(:, allAfCombinationIndeces(j, :));
%          disp("Xr non-zero position: ")
%          disp(allAfCombinationIndeces(j, :))
%          disp("Xr non-zero value: ")
%          disp(Xr)
%      end
%  end
%  
%  