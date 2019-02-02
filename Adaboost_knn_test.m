function [fintpr,fintnr,precision,gmeansfin,yTestFinal] = AdaboostM2_knn_test(train_x,train_y,test_x,test_y,D_T,beta,k,T)
%Test function for AdaboostM2 classifier with underlying kNN
% Applicable for two-class as well as multi-class problems (returns fintpr=-1 for multi-class case)
%T=length(beta);
hyp_label = zeros(T,length(test_y));

%putting the data labels into a column vector form
if (size(test_y,1)==1 || size(test_y,2)==1)
    test_y=test_y(:); %making the label vector a column vector
else
    error('The Labels must be in vector format.');
end
if (size(test_x,1)~=size(test_y,1))
    test_x = test_x'; %turning the data matrix so that rows correspond to datapoints
    if (size(test_x,1)~=size(test_y,1))
    	error('Test data and Labels do not match!');
    end
    if ((size(train_x,2)==size(test_x,2)) || (size(train_x,1)==size(test_x,2)))
        if (size(train_x,1)==size(test_x,2))
            train_x = train_x'; %turning the test data matrix so that rows correspond to datapoints
        end
    else
        error('Training points and testing points do not have same dimensions!');
    end
end
Uc = unique(test_y);
numcls = length(Uc);
if (numcls==2)
    h=((find(test_y==-1)));
    i=((find(test_y==+1)));
    if(length(h)<length(i)) %making the minority class positive for two-class problems
        test_y(h)=+1;
        test_y(i)=-1;
    end
end

%Find which of the input patterns are discrete
% dims = size(train_x,2);
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
%     indices = 1:length(test_y);
%     hyp_label(t,:) = use_tree_C45(test_x', indices, test_y', tree{t}, discrete_dim, Uc);
% kNN
    hyp_label(t,:) = k_nn_classifier(train_x,train_y,k,test_x,D_T(t,:));
end

fxtest=sum((repmat(log(1./(beta+eps))',1,size(test_x,1)).*hyp_label),1);
yTestFinal = sign(fxtest);
if (numcls==2)
    posMask = (test_y==1); negMask = ~posMask;
    out_posMask = (yTestFinal==1); out_negMask = ~out_posMask;
    tp = sum(posMask.*out_posMask');
    fp = sum(negMask.*out_posMask');
    tn = sum(negMask.*out_negMask');
    fn = sum(posMask.*out_negMask');
    precision = tp/(tp + fp + eps);
    fintpr = tp/(tp + fn + eps);
    fintnr = tn/(tn + fp + eps);
    gmeansfin = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) );
else
    fintnr = -1; %Set flag to denote multi-class case
    confMat = zeros(numcls,numcls); %confusion matrix
    for i = 1:numcls
        idxx = find(test_y==Uc(i));
        confMat(i,:) = sum(repmat(Uc',length(idxx),1) == repmat(yTestFinal(idxx)',1,numcls),1);
    end
    fintpr = diag(confMat)'./sum(confMat,2)';
    precision = (sum(confMat,1) - diag(confMat)')./sum(confMat,1);
    gmeansfin = nthroot(prod(fintpr),numcls);
    
end


end

