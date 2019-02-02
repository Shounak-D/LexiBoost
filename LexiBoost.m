%% Required Inputs:
    % data -> matrix of data points, rows correspond to points
    % labels -> column vector of class labels
    
%% Main Loop
cvfold = 5; %input('Enter the value of cvfold: ');
T = 10; %input('Enter the number of iterations: ');
k_knn = 3; %input('Enter the number of neighbors for kNN: ');
C = cvpartition(labels,'k',cvfold);
%initialization
alpha_orig = cell(1,cvfold);
beta_orig = cell(1,cvfold);
D_T = cell(1,cvfold);
testgmeans_orig = zeros(1,cvfold);
testauc_orig = zeros(1,cvfold);
testf1_orig = zeros(1,cvfold);
testgmeans_cls = zeros(cvfold,length(unique(labels)));
testauc_cls = zeros(cvfold,length(unique(labels)));
testf1_cls = zeros(cvfold,length(unique(labels)));
testgmeans_dcls = zeros(cvfold,length(unique(labels)));
testauc_dcls = zeros(cvfold,length(unique(labels)));
testf1_dcls = zeros(cvfold,length(unique(labels)));
testgmeans_fin = zeros(cvfold,1);
testauc_fin = zeros(cvfold,1);
testf1_fin = zeros(cvfold,1);
testgmeans_dfin = zeros(cvfold,1);
testauc_dfin = zeros(cvfold,1);
testf1_dfin = zeros(cvfold,1);
for i = 1:cvfold
    trIdx = C.training(i);
    teIdx = C.test(i);
    train_x = data(trIdx,:);
    test_x = data(teIdx,:);
    train_y = labels(trIdx);
    test_y = labels(teIdx);
    %run Adaboost
    [testgmeans_orig(i), testauc_orig(i), testf1_orig(i), alpha_orig{i}, beta_orig{i}, D_T{i}] = ...
                                                                                        Adaboost_train_test(train_x, train_y, test_x, test_y, T, k_knn);
    %run LexiBoost
    [testgmeans_cls(i,:), testauc_cls(i,:), testf1_cls(i,:), obj(i,:)] = LexiBoost_cls2(train_x, train_y, test_x, test_y, beta_orig{i}, D_T{i}, k_knn, T);
    [testgmeans_fin(i), testauc_fin(i), testf1_fin(i)] = LexiBoost_fin2(train_x, train_y, test_x, test_y, beta_orig{i}, D_T{i}, obj(i,:), k_knn, T);
    
    %run Dual-LexiBoost
    [testgmeans_dcls(i,:), testauc_dcls(i,:), testf1_dcls(i,:), tempt_1, tempt_2] = LexiBoost_dcls(train_x, test_x, train_y, test_y, T, k_knn);
    [testgmeans_dfin(i), testauc_dfin(i), testf1_dfin(i)] = LexiBoost_dfin(train_x, test_x, train_y, test_y, T, k_knn, sum(tempt_1), sum(tempt_2));
    %t_1{j} =  tempt_1;
    %t_2{j} = tempt_2;
    fprintf('Finished runs for partition %d.\n',i);
end

%% Finding the average performances
unq = unique(labels);
%for Adaboost
[cvtestgmeans_orig, cvtestauc_orig, cvtestf1_orig] = average(testgmeans_orig,testauc_orig,testf1_orig);
%for LexiBoost
cvtestgmeans_cls = zeros(1,length(unq));
cvtestauc_cls =  zeros(1,length(unq));
cvtestf1_cls =  zeros(1,length(unq));
for j=1:length(unq)
    [cvtestgmeans_cls(j),cvtestauc_cls(j),cvtestf1_cls(j)] = average(testgmeans_cls(:,j),testauc_cls(:,j),testf1_cls(:,j));
end
[cvtestgmeans_fin, cvtestauc_fin, cvtestf1_fin] = average(testgmeans_fin,testauc_fin,testf1_fin);
%for Dual-LexiBoost
cvtestgmeans_dcls = zeros(1,length(unq));
cvtestauc_dcls = zeros(1,length(unq));
cvtestf1_dcls = zeros(1,length(unq));
for j=1:length(unq)
    [cvtestgmeans_dcls(j),cvtestauc_dcls(j),cvtestf1_dcls(j)] = ...
                                    average(testgmeans_dcls((testgmeans_dcls(:,j)~=-1),j),testauc_dcls((testauc_dcls(:,j)~=-1),j),testf1_dcls((testf1_dcls(:,j)~=-1),j));
end
[cvtestgmeans_dfin, cvtestauc_dfin, cvtestf1_dfin] = average(testgmeans_dfin,testauc_dfin,testf1_dfin);
