function [cvgmeans, cvauc, cvf1] = average(gmeans, auc, f1)

cvgmeans = mean(gmeans);
cvauc = mean(auc);
cvf1 = mean(f1);

end

