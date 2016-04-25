%algorithm references:
%CSE 6363 slides
%www.robots.ox.ac.uk/~az/lectures/ml/matlab2.pdf
train_dir = '/Users/VA/Documents/MS/Acads/Fall15/ML/PA2/1st_Dataset/1st_Dataset_Train/';
k=1;
img_matrix=cell(200,1);
%read files from the specified training path
for i=1:40
    folder_name=strcat('s',int2str(i));
    path1=strcat(train_dir,folder_name);
    d=dir(path1);
    file_names = {d.name};
    for j=3:size(d)
        s=strcat(path1,'/',file_names{j});
        %disp(s);
        img=imread(s); 
        img_matrix{k}=reshape(img, 1, []);
        k=k+1;
    end 
end
a = [];
for i=1:200
    a=[a;img_matrix{i}];
end
%training
%create X matrix of dimension 200x10305 including the bias term in the last
%column
X = [a ones(200,1)];
Y=[];
Y_cell = cell(40,1);
%generate Y diagonal matrix of dimension 1x200 with the labels in the leading diagonal
for m=1:40
    for n=1:200
        if (n==(m*5))
            for k=n:-1:n-4
                Y(k)=1;
            end
        else
            Y(n)=-1;
        end
    end
    Y_cell{m}=diag(Y);
end

w_cell = cell(40,1);
n = 10304;
l = 200;
H = eye(n+1);
H(n+1,n+1) = 0;
f = zeros(n+1,1);
c = -ones(l,1);
A = [];
A_resized = [];
for m = 1:40
    %generating the the third parameter of quadprog function
    A = (-double(Y_cell{m}) * double(X));
    %dotproduct = dotproduct;
    %dotproduct(201:10305,:)=0;
    %A_resized = (dotproduct).*(1+dotproduct);
    %A = A_resized(1:200,:);
    %A = (double(dotproduct') * double(dotproduct) + 1);
    %invoke quadprog to retrieve the weight vector w
    w = quadprog(H,f,A,c);
    %if (A*w<=c)
    w_cell{m} = w;
    %end
end

test_path = '/Users/VA/Documents/MS/Acads/Fall15/ML/PA2/1st_Dataset/1st_Dataset_Test/';
test_dir = dir('/Users/VA/Documents/MS/Acads/Fall15/ML/PA2/1st_Dataset/1st_Dataset_Test/*.pgm');

%testing
for i = 1:200
    filename = strcat(test_path,test_dir(i).name);
    test_1 = imread(filename);
    test_img_1 = reshape(test_1,[],1);
    for m = 1:40
        w_inloop = w_cell{m};
        b = w_inloop(10305,1);
        class_pred = sign(double(w_inloop(1:10304)')*double(test_img_1)+b);
        %display(class_pred);
        if (class_pred == 1)
            display(['The image ', test_dir(i).name,' belongs to class s',num2str(m)]);
            break;
        end
    end
end