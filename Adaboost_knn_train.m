function [beta, fxtrain, D_T, k] = AdaboostM2_knn_train(train_x,train_y,test_x,k,T)
%Adaboost.M2 classifier with kNN as the underlying learner

%putting the data labels into a column vector form
if (size(train_y,1)==1 || size(train_y,2)==1)
    train_y=train_y(:); %making the label vector a column vector
else
    error('The Labels must be in vector format.');
end
if (size(train_x,1)~=size(train_y,1))
    train_x = train_x'; %turning the data matrix so that rows correspond to datapoints
    if (size(train_x,1)~=size(train_y,1))
    	error('Data and Labels do not match!');
    end
    if ((size(test_x,2)==size(train_x,2)) || (size(test_x,1)==size(train_x,2)))
        if (size(test_x,1)==size(train_x,2))
            test_x = test_x'; %turning the test data matrix so that rows correspond to datapoints
        end
    else
        error('Training points and testing points do not have same dimensions!');
    end
end
Uc = unique(train_y);
numcls = length(Uc);
if (numcls==2)
    h=((find(train_y==-1)));
    i=((find(train_y==+1)));
    if(length(h)<length(i)) %making the minority class positive for two-class problems
        train_y(h)=+1;
        train_y(i)=-1;
    end
end

%initialization
[n, dims] = size(train_x);

%initialization for boosting
D = (1/n)*ones(n,1);
D_T = zeros(T,n);
w = repmat(D/(numcls-1),1,numcls);
w(repmat(Uc',length(train_y),1) == repmat(train_y,1,numcls)) = 0;
eror = zeros(1,T);
beta = zeros(1,T);

%intialization for C4.5
% tree = cell(1,T);
fxtrain = zeros(n,T);
% inc_node = 2; %Threshold for small number of samples at a node
% confidenceFactor = 0.25; %Default confidence factor for C4.5 is used here
% %Calculate the extra error factor 'coeff'
% Val = [0,   0.001,  0.005,  0.01,   0.05,   0.1,    0.2,    0.4,    1];
% Dev = [100, 3.09,   2.58,   2.33,   1.65,   1.28,   0.84,   0.25,   0];
% indx = sum(confidenceFactor > Val) + 1;
% eta = (confidenceFactor - Val(indx-1))/(Val(indx) - Val(indx-1));
% coeff = Dev(indx-1) + eta*(Dev(indx) - Dev(indx-1));
% coeff = coeff * coeff;
% %Find which of the input patterns are discrete
% Nu = 10; %Threshold for discrete features is 10
% discrete_dim = zeros(1,dims);
% for i = 1:dims
%     Ub = unique([train_x(:,i); test_x(:,i)]);
%     Nb = length(Ub);
%     if (Nb <= Nu)
%         %This is a discrete pattern
%         discrete_dim(i)	= Nb;
%     end
% end

for t = 1:T
    %Boosting parameters
    W = sum(w,2);
    q = w./repmat(W+eps,1,numcls);
    D = W/(sum(W)+eps);
    D_T(t,:) = D*n;
    
    %kNN Classifier
    [pred_y] = k_nn_classifier(train_x,train_y,k,train_x,D_T(t,:));
    %C4.5
%     tree{t} = make_tree_C45(train_x', train_y', D_T', inc_node, discrete_dim, 0);
%     indices = 1:length(train_y);
%     [tree{t}, ~] = prune_tree_C45(train_x', train_y', D_T', indices, tree{t}, discrete_dim, confidenceFactor, coeff, Uc');
%     pred_y = use_tree_C45(train_x', indices, train_y', tree{t}, discrete_dim, Uc');
    pred_y = pred_y';
    fxtrain(:,t) = pred_y;
    
    %Boosting Update
    h = repmat(Uc',length(pred_y),1) == repmat(pred_y,1,numcls);
    corr = pred_y == train_y;
    eror(t) = (1/2)*sum(D.*(ones(size(D)) - corr + sum(q.*h,2)));
    beta(t) = eror(t)/(1 - eror(t) + eps);
    w = w.*(repmat(beta(t),size(w,1),size(w,2)).^((1/2)*(ones(size(w)) + repmat(corr,1,numcls) - h)));
end


end