function [testGmeans, testAuc, testF1, obj] = LexiBoost_cls2(train_x, train_y, test_x, test_y, beta_orig, D_T, k, T)

%% Finding the training performance for AdaBoost
hyp_label = zeros(length(beta_orig),length(train_y));
% T = length(beta_orig);
for t = 1:T
    % for the kNN classifier
    % k = 3;
    hyp_label(t,:) = k_nn_classifier(train_x, train_y, k, train_x, D_T(t,:));
end

fx_train = hyp_label';

%% Finding the Minimum Pinball losses for the individual classes
unq=unique(train_y);
cls_soln = cell(1,length(unq));
obj = zeros(1,length(unq));
exitflag = zeros(1,length(unq));
alpha_cls = zeros(length(unq),length(beta_orig));

for i = 1:length(unq)
    
    train_y_cls = train_y(train_y==unq(i));
    fx_train_cls = fx_train(train_y==unq(i),:);
    A = [-1*(repmat(train_y_cls,1,length(beta_orig)).*fx_train_cls) -1*eye(size(train_y_cls,1))];
    b = -1*ones(size(train_y_cls,1),1);
    Aeq = [ones(1,length(beta_orig)), zeros(1,size(train_y_cls,1))];%%
    beq = 1;%%    
    LB = zeros(length(beta_orig)+size(train_y_cls,1),1);
    UB = Inf*ones(length(beta_orig)+size(train_y_cls,1),1);
    fcls = [zeros(length(beta_orig),1); ones(size(train_y_cls,1),1)];
    [cls_soln{i},obj(i),exitflag(i)] = linprog(fcls,A,b,Aeq,beq,LB,UB);%%
%     [cls_soln{i},obj(i),exitflag(i)] = linprog(fcls,A,b,[],[],LB,UB);
    alpha_cls_temp = cls_soln{i}; 
    alpha_cls(i,:) = alpha_cls_temp(1:length(beta_orig));
    
end

%% Testing
testtpr_cls = zeros(1,length(unq));
testtnr_cls = zeros(1,length(unq));
testprec_cls = zeros(1,length(unq));
testGmeans = zeros(1,length(unq));
testAuc = zeros(1,length(unq));
testF1 = zeros(1,length(unq));
yTest_cls = zeros(length(unq),length(test_y));
for i = 1:length(unq)
    [testtpr_cls(i),testtnr_cls(i),testprec_cls(i),testGmeans(i),yTest_cls(i,:)] = ...
                                Adaboost_knn_test(train_x,train_y,test_x,test_y,D_T,exp(-alpha_cls(i,:)),k, T);
    testAuc(i) = (testtpr_cls(i)+testtnr_cls(i))/2;
    testF1(i) = (2*testprec_cls(i)*testtpr_cls(i))/(testtpr_cls(i)+testprec_cls(i)+eps);
end

end
