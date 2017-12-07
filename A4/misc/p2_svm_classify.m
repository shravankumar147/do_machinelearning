clc;close all;
DataSVM = loadDataSVM('Data_SVM.csv');
labels= DataSVM.V3;
data = DataSVM(:,1:end-1);

xdata = table2array(data);
group = labels;
figure; 
svmStruct = svmtrain(xdata,group,'boxconstraint',2,'kernel_function','rbf','rbf_sigma',1,'ShowPlot',true);
title('SVM-using rbf')
figure; 
svmStruct = svmtrain(xdata,group,'boxconstraint',2,'kernel_function','polynomial','polyorder',3,'ShowPlot',true);
title('SVM-using polynomial kernel')