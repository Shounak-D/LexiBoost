function [finhyp] = k_nn_classifier(train_x,train_y,k,test_x,D_wts)

if (nargin<4)
    error('4 inputs are required!')
else
    if (nargin<5)
        D_wts = ones(1,length(train_y));
    end
    [n1,~]=size(train_x);
    [n2,~]=size(test_x);
    unq = unique(train_y);

    % for i = 1:size(test_x,1)
    %     distt(i,:) = sqrt(sum((train_x - repmat(test_x(i,:),size(train_x,1),1)).^2,2))';
    % end
    dotProduct = test_x*train_x';
    distt = repmat(sqrt(sum(test_x.^2,2).^2),1,n1) - 2*dotProduct + repmat(sqrt(sum(train_x.^2,2)'.^2),n2,1);

    % sort the points according to distance
    [~, closest_neighbour] = sort(distt,2,'ascend');
    
    finhyp = zeros(1,size(test_x,1));
    for i=1:size(test_x,1)
        sum_wts = zeros(1,length(unq));
        hyp = train_y(closest_neighbour(i,1:k));
        if (length(hyp)~=k)
            warning('There is something wrong!');
        end
        hyp_wts = D_wts(closest_neighbour(i,1:k));
        if (length(hyp_wts)~=k)
            warning('There is something wrong!');
        end
        for j=1:length(unq)
            tmp=(hyp==unq(j)); %vector: 1 for points in the current class, 0 otherwise
            if(sum(tmp)~=0)
                sum_wts(j)=sum(hyp_wts(tmp));
            end
        end
        [~, idxxx] = max(sum_wts);
        finhyp(i) = unq(idxxx);
    end

end


end

