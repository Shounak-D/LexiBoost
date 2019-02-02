function [testGmeans, testAuc, testF1] = LexiBoost_fin2(train_x, train_y, test_x, test_y, beta_orig, D_T, obj_cls, k, T)

%% Finding the training performance for AdaBoost
alpha_orig = log(1./beta_orig); %calculating alphas from the obtained betas
alpha_orig = alpha_orig/sum(alpha_orig); %normalizing for easy comparison with other alphas
hyp_label = zeros(length(beta_orig),length(train_y));
%T = length(beta_orig);
for t = 1:T
    % for the kNN classifier
    %k = 3;
    hyp_label(t,:) = k_nn_classifier(train_x, train_y, k, train_x, D_T(t,:));
end

fx_train = hyp_label';

%% Finding best trade-off alphas such that the maximum deviation from the class-wise minimum average pinball losses is minimized
unq=unique(train_y);
coeffs_t = ones(length(unq),size(train_y,1));
for i = 1:length(unq)
    coeffs_t(i,train_y~=unq(i))=0;
end
coeffs_c = -1*ones(size(obj_cls'));
for i = 1:length(unq)
    coeffs_c(i)=coeffs_c(i)*length(find((train_y==unq(i)))); %multiply weight here for tuning on 1/n_j
end
A = [-1*(repmat(train_y,1,length(alpha_orig)).*fx_train) -1*eye(size(train_y,1)) zeros(size(train_y,1),1);...
            zeros(length(unq),length(alpha_orig)) coeffs_t coeffs_c]; 
b = [-1*ones(size(train_y,1),1); obj_cls'];
Aeq = [ones(1,length(alpha_orig)), zeros(1,size(train_y,1)), 0];%%
beq = 1;%%
LB = [zeros(length(alpha_orig)+size(train_y,1),1); -Inf];%%
% LB = [zeros(length(alpha_orig)+size(train_y,1),1); 0];
UB = Inf*ones(length(alpha_orig)+size(train_y,1)+1,1);
fobj = [zeros(length(alpha_orig)+size(train_y,1),1); 1];
[fin_soln,~,~] = linprog(fobj,A,b,Aeq,beq,LB,UB);%%
% [fin_soln,finobj,exitflag_fin] = linprog(fobj,A,b,[],[],LB,UB);
alpha_fin=fin_soln(1:length(alpha_orig))';
% alpha_fin = mean(alpha_cls,1);

%% Testing
[testtpr_fin, testtnr_fin, testprec_fin, testGmeans,~] = Adaboost_knn_test(train_x, train_y, test_x, test_y, D_T, exp(-alpha_fin), k, T);
testAuc = (testtpr_fin+testtnr_fin)/2;
testF1 = (2*testprec_fin*testtpr_fin)/(testtpr_fin+testprec_fin+eps);

end