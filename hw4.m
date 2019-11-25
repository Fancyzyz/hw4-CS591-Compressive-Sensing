


N=256;
S=10;
S=5;

% M=100; 
% M from 20 to 100 here

epsilon=1e-6;
Matrixindex=3;
countM=[];
for Matrixindex=1:6
    countV=[];
    for M=20:10:100
        count=0;
        for iter=1:100
            q=randperm(N);
            [x]=generateFreqSparseSignal(N,S);
            Psi=dctmtx(N);
            %Psi=eye(N);
            Psi_inv= inv(Psi);
        % generating x.
            if Matrixindex==1
                A=generateTimeDomainSensingMatrix(M,N,q);
            elseif Matrixindex==2
                A=generateUniformTimeDomainSensingMatrix(M,N);
            elseif Matrixindex==3    
                A=generateFreqDomainSensingMatrix(M, N);
            elseif Matrixindex==4
                A=generateLowFreqDomainSensingMatrix(M,N);
            elseif Matrixindex==5    
                A=generateEuqispaceFreqDomainSensingMatrix(M,N);
            elseif Matrixindex==6
                A=generateRandomGaussianOrthonormalizedMatrix(M,N);
            end
            y=A*x;
            
            A1=A*Psi_inv;
            B=[A1 -A1];
            r=ones(1,2*N);
            %x_hat=SimplexAlg(y,A,M,N);
            %x_hat=linprog(r,[],[],A,y);
            lb=zeros(1,2*N);
            z=linprog(r,[],[],B,y,lb,[],[]);
            x_hat=zeros(1,N);
            c=size(x_hat);
            if c(1)~=0
                x_hat=z(1:N)-z(N+1:2*N);
            end
            x_hat=Psi_inv*x_hat;
            
%             sprintf("residue:%f",norm(x-x_hat,2));
%             x0 = A1.'*y;
%             x_hat=l1eq_pd(x0, A1, [], y, 1e-7);
%             x_hat=Psi_inv*x_hat;
            
            if norm(x-x_hat,2)<epsilon
                disp("perfect");
                count=count+1;
            end
        end
        
        countV=[countV,count/100];
    end
    
    countM=[countM,countV];
    
end
X=20:10:100;
h=plot(X,countM(1:9),'b',X,countM(10:18),'r',X,countM(19:27),'y',X,countM(28:36),'g',X,countM(37:45),'m',X,countM(46:54),'k');
h(1).MarkerSize = 8;
h(1).MarkerFaceColor = 'c';
title('Compare 6 differnet Sensing Matrices');
xlabel('M,numbers of measurements');
ylabel('Probability of Perfect Recovery');


function [x] = generateTimeSparseSignal(N,S)
    x=zeros(N,1);
    q=randperm(N);
    x(q(1:S))=randn(S,1);
end 

function [x] = generateFreqSparseSignal(N,S)
    alpha=zeros(N,1);
    q=randperm(N);
    alpha(q(1:S))=randn(S,1);
    x=idct(alpha);
end
    
    
function [x_hat] = SimplexAlg(y,A,M,N)

B=[A -A];

r=-1*ones(1,2*N);
SimM=[B y;r 0];




while true
   if sum(SimM(M+1,:)<0)==0
       u=SimM(M+1,1:N);
       v=SimM(M+1,N+1:2*N);
       x_hat=u-v;
       break
   end 
   
   [~,p_column]=min(SimM(M+1,1:2*N));
   devV=SimM(1:M,2*N+1)./SimM(1:M,p_column);
   SuppdevV=devV>0;
   
   flag=0;
   %[~,p_row]=min(devV);
   for i=1:M 
       if SuppdevV(i)==1
           if flag==0
               p_row=i;flag=1; 
           elseif devV(p_row)<devV(i)
               p_row=i;
           end
       end
    end
   
   pivot=SimM(p_row,p_column);
   temp=SimM;
   
   for i=1:M+1
       for j=1:2*N+1
           
           if i==p_row && j==p_column
               SimM(i,j)=1/pivot;
               % p
            
           elseif i==p_row && j~=p_column
               SimM(i,j)=temp(i,j)/pivot;  
               % r
             
           elseif i~=p_row && j==p_column    
               SimM(i,j)=-temp(i,j)/pivot;
               % c
                
           else
               SimM(i,j)=temp(i,j)-temp(p_row,j)*temp(i,p_column)/pivot;
           end
            
       end              
   end                   
end    
end

function [A] = generateTimeDomainSensingMatrix(m,n,q)
    I = eye(n);
    randIndeces = randperm(n);
    A = I(q(1:m), :);
    % (a) in Hw4
end

function [A] = generateUniformTimeDomainSensingMatrix(m, n)
    I = eye(n);
    Indeces = [];
    k=floor(n/m);
    for i=1:m
        if i==1
            Indeces=[Indeces,1]; 
        else 
            Indeces=[Indeces,i*k];
        end
    end    
    A = I(Indeces(1:m), :);
    % (b) in Hw4
end




% (Random sampling in the frequency domain: Suppose F is the N by N DCT matrix 
% (F = dct(eye(N));). Create the sensing matrix A by keeping M rows of F at random locations
% (and deleting the remaining M-N rows). ?
function [A] = generateFreqDomainSensingMatrix(m, n)
    F = dct(eye(n));
    randIndeces = randperm(n);
    A = F(randIndeces(1:m), :);
    % (c) in hw4
end




function [A] = generateLowFreqDomainSensingMatrix(m, n)
    F = dct(eye(n));
    A = F(1:m, :);    
    % (d) in hw4
end

function [A] = generateEuqispaceFreqDomainSensingMatrix(m, n)
    F = dct(eye(n));
    Indeces = [];
    k=floor(n/m);
    for i=1:m
        if i==1
            Indeces=[Indeces,1]; 
        else 
            Indeces=[Indeces,i*k];
        end
    end   
    A = F(Indeces(1:m), :);
    % (e) in hw4
end


% Sampling with a random matrix: The sensing matrix A is M by N in this case 
% and is generated from a collection of random Gaussian variables, 
% then the rows are orthonor- malized, i.e.,
% A = randn(M, N); A = orth(A?)?
function [A] = generateRandomGaussianOrthonormalizedMatrix(m, n)
    A = randn(m, n);
    A = orth(A')';
    % (f) in hw4
end




