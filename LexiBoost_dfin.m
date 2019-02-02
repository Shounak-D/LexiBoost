function [ testGmeans, testAuc, testF1 ] = LexiBoost_dfin(train_x, test_x, train_y, test_y, T, k, t_1, t_2)

% XT = cell(1,T); % cell of T training datasets
% XT{1} = train_x;
% labelT = cell(1,T); % label of T training datasets
% labelT{1} = train_y;
n = size(train_x,1); %Number of points in Training dataset;
nTr = size(train_x,1);
%nTe = size(test_x,1);
nTr1 = sum(train_y==-1); %n1 is number of trainig points having label -1;
nTr2 = sum(train_y==1); %n2 is number of trainig points having label 1;
D = zeros(nTr,T); % D is a n*T matrix of weight
% D(:,1) = (1/nTr)*ones(nTr,1); %initial weight for each training points is equal;
D(:,1) = ones(nTr,1);
D(train_y==-1,1) = (0.5/nTr1)*D(train_y==-1,1);
D(train_y==1,1) = (0.5/nTr2)*D(train_y==1,1);
Z = zeros(1,T);
% pt_error = zeros(nTr,1);%will return 1 for correct classified and -1 for misclassified(calculated in for loop)
nC = length(unique(train_y));
tt = 0;
for t=1:T
    D_T(t,:) = D(:,t)'*n;
    [Q] = k_nn_classifier(train_x,train_y,k,train_x,D_T(t,:)); %using KNN classifier;
    Q = Q'; %Q is predicted label by classifier
    pt_error(:,t) = ((Q==train_y)*2)-1;
    pt_error1 = (Q==train_y);
    if(pt_error1'*D(:,t)< 0.5)
        tt = t-1;
        break;
    end
    %ErrorTr(t) = sum(Q~=labelT{t})/length(labelT{t}); %should be a fraction \in [0,1]
    pt_error_t = pt_error(:,t);
    
    f = [sum(t_1,1)/nTr1; sum(t_2,1)/nTr2; -1*ones(nTr,1); 1]; %multiply weight here with nTr1 and/or nTr2 for tuning on 1/n_j
    A1(t,:) = [zeros(1,nC), pt_error_t(:)', -1];
    A2 = [-(train_y==-1)/nTr1, -(train_y==1)/nTr2, eye(nTr), zeros(nTr,1)]; %multiply weight here with nTr1 and/or nTr2 for tuning on 1/n_j
    AN = [A1; A2];
    bN = zeros(t+nTr,1);
    Aeq = [zeros(1,nC), ones(1,nTr), 0];
    beq = 1;
    LBN = [zeros(nTr+nC,1); -Inf];
    Sol = linprog(f,AN,bN,Aeq,beq,LBN,[]);    
        
    UN = zeros(nTr,1);
    UN = Sol(nC+1:nTr+nC);
    UN(UN<0) = 0; % to handle rounding-off errors
    D(:,t+1) = UN;
    Z(t+1)= sum(D(:,t+1),1);
    D(:,t+1)=D(:,t+1)/Z(t+1); %Normalization

end
if (tt~=0)
    t = tt;
end

unq=unique(train_y);
coeffs_t = ones(nC,nTr);
for i = 1:nC
    coeffs_t(i,train_y~=unq(i))=0;
end
coeffs_c = -1*ones(nC,1);
for i = 1:nC
    coeffs_c(i)=coeffs_c(i)*length(find((train_y==unq(i)))); %multiply weight here for tuning on 1/n_j
end
A = [-1*pt_error(:,1:t) -1*eye(nTr) zeros(nTr,1);...
            zeros(nC,t) coeffs_t coeffs_c]; 
b = [-1*ones(nTr,1); sum(t_1,1); sum(t_2,1)];
Aeq = [ones(1,t), zeros(1,nTr), 0];%%
beq = 1;%%
LB = [zeros(t+nTr,1); -Inf];%%
% LB = [zeros(t+nTr,1); 0];
UB = Inf*ones(t+nTr+1,1);
fobj = [zeros(t+nTr,1); 1];
SolN = linprog(fobj,A,b,Aeq,beq,LB,UB);%%
%SolN = linprog(fobj,A,b,[],[],LB,UB);
alphaN = SolN(1:t,:);

%% 
R = zeros(length(test_y),t);
for tt=1:t
    [R(:,tt)] = k_nn_classifier(train_x,train_y,k,test_x,D_T(tt,:));
end
% yTestFinal = 2*(sum(R,2)>=0)-1;
yTestFinal = sign(R*alphaN);
posMask = (test_y==1); negMask = ~posMask;
out_posMask = (yTestFinal==1); out_negMask = ~out_posMask;
tp = sum(posMask.*out_posMask);
fp = sum(negMask.*out_posMask);
tn = sum(negMask.*out_negMask);
fn = sum(posMask.*out_negMask);
testprec = tp/(tp + fp + eps);
testtpr = tp/(tp + fn + eps);
testtnr = tn/(tn + fp + eps);
testGmeans = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) );
testAuc = (testtpr+testtnr)/2;
testF1 = (2*testprec*testtpr)/(testtpr+testprec+eps);
%Error = sum(yTestFinal~=test_y)/length(test_y);
end

