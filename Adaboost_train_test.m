function [testGmeans, testAuc, testF1, alphas, betas, D_T] = Adaboost_train_test(train_x, train_y, test_x, test_y, T, k)

%% Training the AdaBoost classifiers
% for the kNN classifier
%k = 3; %number of neighbors for kNN classification
[betas, ~, D_T, ~] = Adaboost_knn_train(train_x,train_y,test_x,k,T); %fx_train : NxT

% finding the alphas for Adaboost
alphas = log(1./betas); %calculating alphas from the obtained betas
alphas = alphas/sum(alphas); %normalizing for easy comparison with other alphas

%% Testing
%for the kNN classifier
[testtpr, testtnr, testprec, testGmeans, ~] = Adaboost_knn_test(train_x, train_y, test_x, test_y, D_T, betas, k, T);

testAuc = (testtpr+testtnr)/2;
testF1 = (2*testprec*testtpr)/(testtpr+testprec+eps);


end
