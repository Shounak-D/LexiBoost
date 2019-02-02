function [testGmeans, testAuc, testF1, t_1, t_2] = LexiBoost_dcls(train_x, test_x, train_y, test_y, T, k)

% XT = cell(1,T); % cell of T training datasets
% XT{1} = train_x;
% labelT = cell(1,T); % label of T training datasets
% labelT{1} = train_y;
n = size(train_x,1); %Number of points in Training dataset;
nTr = size(train_x,1);
nTe = size(test_x,1);
nTr1 = sum(train_y==-1); %n1 is number of trainig points having label -1;
nTr2 = sum(train_y==1); %n2 is number of trainig points having label 1;
D = zeros(nTr,T); % D is a n*T matrix of weight
% D(:,1) = (1/nTr)*ones(nTr,1); %initial weight for each training points is equal;
D(:,1) = ones(nTr,1);
D(train_y==-1,1) = (0.5/nTr1)*D(train_y==-1,1);
D(train_y==1,1) = (0.5/nTr2)*D(train_y==1,1);
Z = zeros(1,T);
% pt_error = zeros(nTr,1);%will return 1 for correct classified and -1 for misclassified(calculated in for loop)
tt = 0;
for t=1:T
    D_T(t,:) = D(:,t)'*n;
    [Q] = k_nn_classifier(train_x,train_y,k,train_x,D_T(t,:)); %using KNN classifier;
    Q = Q'; %Q is predicted label by classifier
    pt_error(:,t) = ((Q==train_y)*2)-1;
    pt_error1 = (Q==train_y);
    if(pt_error1'*D(:,t)< 0.5)
        if(t>1)
            tt = t-1;
        end
        break;
    end
    pt_error_t = pt_error(:,t);
%     ErrorTr1(t) = sum(Q(train_y==-1)~=labelT{t}(train_y==-1))/length(labelT{t}(train_y==-1)); %should be a fraction \in [0,1]
%     ErrorTr2(t) = sum(Q(train_y==1)~=labelT{t}(train_y==1))/length(labelT{t}(train_y==1)); %should be a fraction \in [0,1]

    % for the first class
    f1 = [-1*ones(nTr1,1); 1];
    A_1(t,:) = [pt_error_t(train_y==-1)' -1];
    A1 = [A_1; [eye(nTr1) zeros(nTr1,1)]];
    b1 = [zeros(t,1); (1/nTr1)*ones(nTr1,1)];
    Aeq1 = [ones(1,nTr1) 0];
    beq1 = 1;
    LB1 = [zeros(nTr1,1); -Inf];
%     UB1 = Inf*ones(2*nTr1,1);
    Sol1 = linprog(f1,A1,b1,Aeq1,beq1,LB1,[]);
    
    % for the second class
    f2 = [-1*ones(nTr2,1); 1];
    A_2(t,:) = [pt_error_t(train_y==1)' -1];
    A2 = [A_2; [eye(nTr2) zeros(nTr2,1)]];
    b2 = [zeros(t,1); (1/nTr2)*ones(nTr2,1)];
    Aeq2 = [ones(1,nTr2) 0];
    beq2 = 1;
    LB2 = [zeros(nTr2,1); -Inf];
%     UB2 = Inf*ones(2*nTr2,1);
    Sol2 = linprog(f2,A2,b2,Aeq2,beq2,LB2,[]);
    
    U = zeros(nTr,1); %weight assignment
    U(train_y==-1) = Sol1(1:nTr1);
    U(train_y==1) = Sol2(1:nTr2);
%     for i=1:nTr
%         if(U(i)<0)
%             U(i)=0;
%         end
%     end
    U(U<0) = 0; % to handle round-off errors
    D(:,t+1) = U;
    Z(t+1)= sum(D(:,t+1),1);
    D(:,t+1)=D(:,t+1)/Z(t+1); %Normalization

end
if (tt~=0)
    t = tt;
end

% U1 = mean(D(train_y==-1,1:t),1);
% U2 = mean(D(train_y==1,1:t),1);
fT1 = [zeros(1,t), ones(1,nTr1)/nTr1];
AT1 = [-1*pt_error(train_y==-1,1:t), -1*eye(nTr1)];
bT1 = -1*ones(nTr1,1);
AeqT1 = [ones(1,t), zeros(1,nTr1)];%%
beqT1 = 1;%%
LBT1 = zeros(t+nTr1,1);
UBT1 = Inf*ones(t+nTr1,1);
Ans1 = linprog(fT1,AT1,bT1,AeqT1,beqT1,LBT1,UBT1);%%
% Ans1 = linprog(fT1,AT1,bT1,[],[],LBT1,UBT1);
alpha_cls(1,:) = Ans1(1:t);
t_1 = Ans1(t+1:nTr1+t);

fT2 = [zeros(1,t),ones(1,nTr2)/nTr2];
AT2 = [-1*pt_error(train_y==1,1:t), -1*eye(nTr2)];
bT2 = -1*ones(nTr2,1);
AeqT2 = [ones(1,t), zeros(1,nTr2)];%%
beqT2 = 1;%%
LBT2 = zeros(t+nTr2,1);
UBT2 = Inf*ones(t+nTr2,1);
Ans2 = linprog(fT2,AT2,bT2,AeqT2,beqT2,LBT2,UBT2);%%
% Ans2 = linprog(fT2,AT2,bT2,[],[],LBT2,UBT2);
alpha_cls(2,:) = Ans2(1:t);
t_2 = Ans2(t+1:nTr2+t);

%% Testing

unq=unique(train_y);
testGmeans = zeros(1,length(unq));
testAuc = zeros(1,length(unq));
testF1 = zeros(1,length(unq));
R = zeros(length(test_y),t);
for tt=1:t
    [R(:,tt)] = k_nn_classifier(train_x,train_y,k,test_x,D_T(tt,:));
end
for i = 1:length(unq)
    yTestFinal = sign(R*alpha_cls(i,:)');
    posMask = (test_y==1); negMask = ~posMask;
    out_posMask = (yTestFinal==1); out_negMask = ~out_posMask;
    tp = sum(posMask.*out_posMask);
    fp = sum(negMask.*out_posMask);
    tn = sum(negMask.*out_negMask);
    fn = sum(posMask.*out_negMask);
    testprec = tp/(tp + fp + eps);
    testtpr = tp/(tp + fn + eps);
    testtnr = tn/(tn + fp + eps);
    testGmeans(i) = sqrt( (tp/(tp + fn)) * (tn/(tn + fp)) );
    testAuc(i) = (testtpr+testtnr)/2;
    testF1(i) = (2*testprec*testtpr)/(testtpr+testprec+eps);
end


end

