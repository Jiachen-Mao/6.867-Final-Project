
train_y=TrainingSet(:,1);
train_x=TrainingSet(:,2:end);
test_y=TestingSet(:,1);
test_x=TestingSet(:,2:end);
    Method_option.plotOriginal = 1;
    Method_option.xscale = 0;
    Method_option.yscale = 0;
    Method_option.plotScale = 0;
    Method_option.pca = 0;
    Method_option.type = 1;
[predict_Y,mse,r] = SVR(train_y,train_x,test_y,test_x,Method_option);