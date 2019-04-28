clear

% load data
data = xlsread('C:\Users\susu\Desktop\板栗-matlab程序\原始板栗模型分类结果\原始-528.xlsx');
x_train = data(1:352,1:200);
x_test = data(353:end,1:200);
y_train = data(1:352,201);
y_test = data(353:end,201);
% data = xlsread('.\板栗.xlsx','特征波长数据');
% x_train = data(1:352,1:20);
% x_test = data(353:end,1:20);
% y_train = data(1:352,21);
% y_test = data(353:end,21);


ntr = size(x_train,1);
nte = size(x_test,1);

x_train = x_train';   
x_test = x_test';     
[x_train, inputps] = mapminmax(x_train);   

bestcv = 0;
for i=8:16
    for j=-16:-8
        cmd = ['-t 0 -v 5 -c ', num2str(2^i), ' -g ', num2str(2^j)];
        cv = svmtrain(y_train, x_train', cmd);
        if cv>bestcv
            bestcv = cv;
            bestc = 2^i;
            bestg = 2^j;
        end
    end
end
cmd = ['-t 0 -c ', num2str(bestc), ' -g ', num2str(bestg)];
model = svmtrain(y_train, x_train', cmd);

x_test = mapminmax('apply', x_test, inputps);  
disp('预测的准确率：');
y_pre = svmpredict(y_test, x_test', model); %这是预测结果

disp('回判的准确率：');
y_ret = svmpredict(y_train, x_train', model);   %这是回判结果
cv = svmtrain(y_train, x_train', cmd)