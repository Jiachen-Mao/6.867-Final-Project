function [ERROR,ptest,result] = testSVM(data_all,methodoption)
error=zeros(2,1);
comparison=cell(2,1);
result.error=error;
result.comparison=comparison;

dataset=data_all;
dataset_x=dataset(:,2:end);
dataset_y=dataset(:,1);
%% Scale data
% scale dataset_x
[data_scale_x]=mapminmax(dataset_x',0,1);
dataset_x=data_scale_x';

% scale dataset_y
[data_scale_y,ps]=mapminmax(dataset_y',0,1);
dataset_y=data_scale_y';
dataset=[dataset_y dataset_x];
%% Feature selection
%% train the model and parameter optimization
     
     [m, n] = size(dataset); % m data points, n dimensions
     randomPoints = [];
     mtrain = 0.6*m;
     for i=1:mtrain
          index = random('unid', m-96); % Pick the index at random.
          randomPoints(i,:) = dataset(index,:); % Add random point.
          dataset(index,:) = []; % Delete selected row.
          m = m-1; 
     end

     train_data=randomPoints;
     test_data=dataset;
    
     [row_train, col_train] = size(train_data);
     y_train = train_data(:,1);
     x_train=train_data(:,2:end);
     
     [row_test, col_test] = size(test_data);

    y_test = test_data(:,1);
    x_test=test_data(:,2:end);
    

    x_g = x_train;
    y_g = y_train;

    
  %% parameters optimization
   % grid search
    if methodoption==1
        [bestCVmse,bestc,bestg] = SVMcgForRegress(y_g,x_g,-8,8,-8,8,5,0.4,0.4)
        cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -t 2 -p 0.01'];
    end
     % ga cg
    if methodoption==2
   
       ga_option.maxgen = 100;
       ga_option.sizepop = 20;
       ga_option.cbound = [0,100];
       ga_option.gbound = [0,100];
       ga_option.v = 5;
       ga_option.ggap = 0.9;
       [bestCVmse,bestc,bestg] = ...
        gaSVMcgForRegress(y_g,x_g,ga_option);
        cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -t 2 -p 0.01'];
    end
    
    % pso cg
    if methodoption == 3
    pso_option.c1 = 2;
    pso_option.c2 = 2;
    pso_option.maxgen = 100;
    pso_option.sizepop = 50;
    pso_option.k = 0.8;
    pso_option.wV = 1;
    pso_option.wP = 1;
    pso_option.v = 8;
    pso_option.popcmax = 100;
    pso_option.popcmin = 0.1;
    pso_option.popgmax = 100;
    pso_option.popgmin = 0.1;
    
    [bestCVmse,bestc,bestg] = psoSVMcgForRegress(y_g,x_g,pso_option);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -t 2 -p 0.01'];
    end

  % pso cgp
  if methodoption == 4
    pso_option.c1 = 1.5;
    pso_option.c2 = 1.7;
    pso_option.maxgen = 100;
    pso_option.sizepop = 20;
    pso_option.k = 0.6;
    pso_option.wV = 1;
    pso_option.wP = 1;
    pso_option.v = 3;
    pso_option.popcmax = 100;
    pso_option.popcmin = 0.1;
    pso_option.popgmax = 100;
    pso_option.popgmin = 0.1;
    
    pso_option.poppmax = 10;
    pso_option.poppmin = 0.01;
    
    [bestCVmse,bestc,bestg,bestp] = psoSVMcgpForRegress(y_g,x_g,pso_option);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -p ',num2str(bestp),' -s 3 -t 2'];
  end

 % ga cgp
  if methodoption== 5
    
    ga_option.maxgen = 100;
    ga_option.sizepop = 20;
    ga_option.v = 5;
    ga_option.ggap = 0.9;
    ga_option.cbound = [0,100];
    ga_option.gbound = [0,100];
    ga_option.pbound = [0.01,1];
    
    [bestCVmse,bestc,bestg,bestp] = ...
    gaSVMcgpForRegress(y_g,x_g,ga_option);
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -p ',num2str(bestp),' -s 3 -t 2 '];
  end
  % grid search cgp
    if methodoption == 6
        [bestCVmse,bestc,bestg,bestp] = SVMcgpForRegress(y_g,x_g);
        cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -p ',num2str(bestp),' -s 3 -t 2 '];
    end
    
  %%  Train the model
    model=svmtrain(y_g,x_g,cmd);
    [ptrain train_mse]=svmpredict(y_g,x_g,model);
    
  %% test the model

    [ptest test_mse]=svmpredict(y_test,x_test,model);


 %% Unscale data
 
    ptrain = mapminmax('reverse',ptrain',ps);
    ptrain = ptrain';
    ptest = mapminmax('reverse',ptest,ps);
   
    y_test=mapminmax('reverse',y_test',ps);
    y_test=y_test';
   
    
 %% data storage and plot
    MSE = 0;
    ERROR = 0;
    for i = 1:row_test
        MSE = MSE + (y_test(i)-ptest(i)).^2/row_test;
    end
    RMSE = norm(y_test-ptest)/sqrt(row_test)/mean(y_test);
    error(1,1)=MSE;
    error(2,1)=RMSE;
    comparison{1,1}=y_test;
    comparison{2,1}=ptest;
    result.error=error;
    result.comparison=comparison;
    
    for i = 1:row_test
        ERROR = ERROR + 100*abs(y_test(i)-ptest(i))./y_test(i);
    end
    ERROR = ERROR/row_test;
    %ERROR=100*abs(y_test-ptest)./y_test;
       
    figure;
    hold on;
    j=1:row_test;
    plot(j,y_test,'b-');
    plot(j,ptest','r-.');
    legend('Measured data','Predicted data');
    saveas(gcf,'data','jpg');

