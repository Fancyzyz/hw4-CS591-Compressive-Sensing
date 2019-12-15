
S=3;
N=100;
index=nchoosek(1 : N, S);

for i=1:size(index,1)
    Mat=Ar(:,index(i,:));
    Mat1=[Mat,yr];
    
    if rank(Mat)==rank(Mat1)
        break    
    end    
end    


for j=1:size(index,1)
    Mat=Af(:,index(j,:));
    Mat1=[Mat,yf];
    
    if rank(Mat)==rank(Mat1)
        break    
    end    
end   

disp(index(i,:))
disp(index(j,:))