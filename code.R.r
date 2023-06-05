#原始数据缺失比例
setwd("E:/R work/EV")
rawdata<-read.table("3.CSV",header=T, sep=",")
rawdata<-as.matrix(rawdata)
library(mice)
summary(rawdata)
#计算数据缺失比例，原则上，数据缺失比例超过5%，则无法进行插补。
pMiss <- function(x){
  sum(is.na(x))/length(x)*100
}

#apply()函数，1表示行，2表示列
apply(rawdata,2,pMiss)
apply(rawdata,1,pMiss)

#列表显示缺失情况
md.pattern(rawdata)

#画图显示缺失情况
library(VIM)
aggr_plot <- aggr(rawdata,col=c('navyblue','red'),
                  numbers=TRUE, 
                  sortVars=TRUE,
                  labels=names(rawdata),
                  cex.axis=.7,gap=3,
                  ylab=c('Missing data','Pattern'))
#数据插补
library(impute)
knn.rawdata<-impute.knn(rawdata,k=10,rowmax = 0.5,colmax=0.8,maxp =3000, rng.seed=362436069)
View(knn.rawdata)
write.table (knn.rawdata$data, file ="4.csv", sep =",", row.names =FALSE)

#划分训练集验证集
library(tidymodels)
setwd("E:/R work/EV")
rawdata<-read.table("4.CSV",header=T, sep=",")
set.seed(2023)
datasplit <- initial_split(rawdata, prop = 0.70, strata = outcome)
traindata <- training(datasplit)
testdata <- testing(datasplit)
traindata$category<-1
testdata$category<-2
write.table (traindata, file ="traindata.csv", sep =",", row.names =FALSE)
write.table (testdata, file ="testdata.csv", sep =",", row.names =FALSE)
#进行lasso回归
rm(list = ls())
library(glmnet)
library(rms)
library(foreign)
setwd("E:/R work/EV")
rawdata1<-read.table("4set.CSV",header=T, sep=",")
View(rawdata1)
dev=rawdata1[rawdata1$category==1,]
View(dev)
setwd("E:/R work/EV/results/lasso")
x <- as.matrix(dev[,c(3:19)])
y <- dev[,2]
set.seed(2023)
fit <- glmnet(x, y, family = "binomial")
pdf("lambda.pdf")
plot(fit, xvar = "lambda")
dev.off()

cvfit <- cv.glmnet(x, y, family = "binomial", nfolds=10)
pdf("cvfit.pdf")
plot(cvfit)
abline(v=log(c(cvfit$lambda.min,cvfit$lambda.1se)),lty="dashed")
dev.off()

coef=coef(fit, s = cvfit$lambda.1se)
variable=as.matrix(coef[which(coef[,1]!=0),])
variable
#####################################################
#构建模型（logistic/xgboost/randomforest/决策树）
rm(list = ls())
library(tidymodels)
setwd("E:/R work/EV")
EV<-read.table("4set.CSV",header=T, sep=",")
colnames(EV)

# 修正变量类型
# 将分类变量转换为factor,结局也是
for(i in c(2,3,17,19)){ #,3,14,15,22,23,24,25
  EV[[i]] <- factor(EV[[i]])
}

EV$outcome<- factor(EV$outcome,levels=c(0,1),labels=c("No","Yes")) #结局一定要转换！
EV$Ascites <- factor(EV$Ascites,levels=c(1,2,3),labels=c("None","Mild","Moderate to severe"))
EV$CTP <- factor(EV$CTP,levels=c(1,2,3),labels=c("A","B","C"))
#EV$tc <- factor(EV$tc,levels=c(1,2,3,4),labels=c("≤2.92","2.92-3.42","3.42-4.04",">4.04"))
#EV$hdlc <- factor(EV$hdlc,levels=c(1,2,3,4),labels=c("≤0.56","0.56-0.80","0.80-1.09",">1.09"))
#EV$sd <- factor(EV$sd,levels=c(1,2,3,4),labels=c("≤114.28","114.28-131.00","131.00-152.60",">152.60"))
#EV$pc <- factor(EV$pc,levels=c(1,2,3,4),labels=c("≤53","54-78","79-113","≥114"))

# 变量类型修正后数据概况
skimr::skim(EV)
#划分训练集与验证集
dev = EV[EV$category==1,] #dev和vad之前就已经区分了
vad = EV[EV$category==2,]
# 数据预处理
# 先对照训练集写配方
datarecipe <- recipe(outcome ~ tc+hdlc+PC+Ascites+SD+CTP, dev) %>%
  #step_rm(Id) %>% 剔除模型中的无关变量
  step_naomit(all_predictors(), skip = F) %>% #剔除缺失值
  step_dummy(all_nominal_predictors()) %>% #对所有分类变量进行独热编码
  prep()
datarecipe #因为这个公式已经代表了上述6个变量，所以下面可以直接用.
#按方处理训练集和测试集
traindata <- bake(datarecipe, new_data = NULL) %>%
  select(outcome, everything())
testdata <- bake(datarecipe, new_data =vad) %>%
  select(outcome, everything())
# 数据预处理后数据概况，这一步转化以后只有outcome是factor，其他的是numeric变量
skimr::skim(traindata)
skimr::skim(testdata)

#############################训练logistic模型
# 设定模型
model_logistic <- logistic_reg(
  mode = "classification", #默认是分类
  engine = "glm"
)
model_logistic

# 拟合模型
fit_logistic <- model_logistic %>%
  fit(outcome~., traindata) #使用.是因为上面已经设置了模型
fit_logistic
fit_logistic$fit
summary(fit_logistic$fit)

# 系数输出
fit_logistic %>%
  tidy()

# 应用模型-预测训练集
predtrain_logistic <- fit_logistic %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "logistic")
predtrain_logistic #pred_Yes和No是根据自己定的
# 评估模型ROC曲线-训练集上
contrasts(traindata$outcome) #首先根据这个函数看参考水平是哪个，0所对的是参考水平默认是first，1所对的是second。如果结局是0，就写first。
levels(traindata$outcome)
roctrain_logistic <- predtrain_logistic %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_logistic
#训练集roc曲线
autoplot(roctrain_logistic) 
# 约登法则对应的p值，概率阈值点
yueden_logistic <- roctrain_logistic %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_logistic
# 预测概率+约登法则=预测分类
predtrain_logistic2 <- predtrain_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_logistic, "Yes", "No")))
predtrain_logistic2
# 混淆矩阵
cmtrain_logistic <- predtrain_logistic2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_logistic

autoplot(cmtrain_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
eval_train_logistic <- cmtrain_logistic %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_logistic %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_logistic

###################################
# 应用模型-预测测试集
predtest_logistic <- fit_logistic %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "logistic")
predtest_logistic

# 评估模型ROC曲线-测试集上
roctest_logistic <- predtest_logistic %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
autoplot(roctest_logistic)

# 预测概率+约登法则=预测分类
predtest_logistic2 <- predtest_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_logistic, "Yes", "No")))
predtest_logistic2

# 混淆矩阵
cmtest_logistic <- predtest_logistic2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_logistic

autoplot(cmtest_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
eval_test_logistic <- cmtest_logistic %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_logistic %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_logistic

##############################################################
# 合并训练集和测试集上ROC曲线
roctrain_logistic %>%
  bind_rows(roctest_logistic) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))
  
# 合并训练集和测试集上性能指标
eval_logistic <- eval_train_logistic %>%
  bind_rows(eval_test_logistic) %>%
  mutate(model = "logistic")
eval_logistic

setwd("E:/R work/EV/results/eval")
write.table (eval_logistic, file ="eval_logistic.csv", sep =",", row.names =FALSE)

#################################################################
###交叉验证，产生10折图
# 当样本量很小的时候可以设定n折交叉验证,n折交叉验证就可以不用训练集验证集，而是把所有都拿来建模。
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# workflow
wf_logistic <- 
  workflow() %>%
  add_model(model_logistic) %>%
  add_formula(outcome ~ .)
wf_logistic

# 交叉验证
set.seed(2023)
cv_logistic <- 
  wf_logistic %>%
  fit_resamples(folds,
                metrics = metric_set(accuracy, roc_auc, pr_auc),
                control = control_resamples(save_pred = T))
cv_logistic

# 交叉验证指标平均结果
eval_cv_logistic <- collect_metrics(cv_logistic)
eval_cv_logistic

# 交叉验证指标具体结果，每一折
eval_cv10_logistic <- collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  mutate(model = "logistic") %>%
  left_join(eval_cv_logistic[c(1,3,5)]) #135代表了什么
eval_cv10_logistic

# 保存评估结果
setwd("E:/R work/EV/cls2")
save(fit_logistic,
     predtrain_logistic,
     predtest_logistic,
     eval_logistic,
     eval_cv10_logistic, 
     file = "evalresult_logistic.RData")

# 交叉验证指标折线图图示
eval_cv10_logistic %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 交叉验证ROC曲线图示
collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(size = 1) +
  theme_classic()

#####################################训练randomforest模型
# 设定模型
model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_args(importance = T)
model_rf

# workflow
wk_rf <- 
  workflow() %>%
  add_model(model_rf) %>%
  add_formula(outcome ~ .)#不需要写出来具体指标
wk_rf

# 重抽样设定-n折交叉验证，寻找最好的超参数，也是生成特殊的ROC曲线所需
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# 超参数寻优范围
hpset_rf <- parameters(
  mtry(range = c(2, 10)), 
  trees(range = c(200, 500)),
  min_n(range = c(20, 50))
)
hpgrid_rf <- grid_regular(hpset_rf, levels = c(3, 2, 2)) #322是指mtry,tree,min_n分别有3,2,2个备选值
hpgrid_rf

# 交叉验证网格搜索过程
set.seed(2023)
tune_rf <- wk_rf %>%
  tune_grid(resamples = folds,
            grid = hpgrid_rf,
            metrics = metric_set(accuracy, roc_auc, pr_auc), #前面的网站上可查
            control = control_grid(save_pred = T, verbose = T)) #打印中间信息

# 图示交叉验证结果
autoplot(tune_rf)#图像解读？
eval_tune_rf <- tune_rf %>%
  collect_metrics()
eval_tune_rf #3*2*2=12个超参数可能性，每个进行5折交叉验证，最后得到平均值（12*3,3个参数）

# 经过交叉验证得到的最优超参数
hpbest_rf <- tune_rf %>%
  select_best(metric = "roc_auc")
hpbest_rf

# 采用最优超参数组合训练最终模型
set.seed(2023)
final_rf <- wk_rf %>%
  finalize_workflow(hpbest_rf) %>%
  fit(traindata)
final_rf

# 提取最终的算法模型
final_rf2 <- final_rf %>%
  extract_fit_engine()
plot(final_rf2, main = "随机森林树的棵树与误差演变")
legend("top", 
       legend = colnames(final_rf2$err.rate),
       lty = 1:3,
       col = 1:3,
       horiz = T)

# 变量重要性
importance(final_rf2)
varImpPlot(final_rf2, main = "变量重要性")

# 偏依赖图
partialPlot(final_rf2, 
            pred.data = as.data.frame(traindata), 
            x.var = tc, #指定一个自变量
            which.class = "Yes")

##############################################################

# 应用模型-预测训练集
predtrain_rf <- final_rf %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "rf")
predtrain_rf

# 评估模型ROC曲线-训练集上
contrasts(traindata$outcome)
roctrain_rf <- predtrain_rf %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
autoplot(roctrain_rf)

# 约登法则对应的p值
yueden_rf <- roctrain_rf %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_rf
# 预测概率+约登法则=预测分类
predtrain_rf2 <- predtrain_rf %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_rf, "Yes", "No")))
predtrain_rf2
# 混淆矩阵
cmtrain_rf <- predtrain_rf2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_rf

autoplot(cmtrain_rf, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_rf <- cmtrain_rf %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_rf %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_rf

##############################################################

# 应用模型-预测测试集
predtest_rf <- final_rf %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "rf")
predtest_rf
View(predtest_rf)
# 评估模型ROC曲线-测试集上
roctest_rf <- predtest_rf %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
autoplot(roctest_rf)

# 预测概率+约登法则=预测分类
predtest_rf2 <- predtest_rf %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_rf, "Yes", "No")))
predtest_rf2
# 混淆矩阵
cmtest_rf <- predtest_rf2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_rf

autoplot(cmtest_rf, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_rf <- cmtest_rf %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_rf %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_rf

##############################################################

# 合并训练集和测试集上ROC曲线
roctrain_rf %>%
  bind_rows(roctest_rf) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))


# 合并训练集和测试集上性能指标
eval_rf <- eval_train_rf %>%
  bind_rows(eval_test_rf) %>%
  mutate(model = "rf")
eval_rf
setwd("E:/R work/EV/results/eval")
write.table (eval_rf, file ="eval_rf.csv", sep =",", row.names =FALSE)
#############################################################
# 最优超参数的交叉验证指标平均结果（提取一些中间结果）
eval_best_cv_rf <- eval_tune_rf %>%
  inner_join(hpbest_rf[, 1:3])
eval_best_cv_rf

# 最优超参数的交叉验证指标具体结果
eval_best_cv10_rf <- tune_rf %>%
  collect_predictions() %>%
  inner_join(hpbest_rf[, 1:3]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes, event_level = "second") %>% #之前的有错，少了event_level
  ungroup() %>%
  mutate(model = "rf") %>%
  inner_join(eval_best_cv_rf[c(4,6,8)])
eval_best_cv10_rf

# 保存评估结果
setwd("E:/R work/EV/cls2")
save(final_rf,
     predtrain_rf,
     predtest_rf,
     eval_rf,
     eval_best_cv10_rf, 
     file = "evalresult_rf.RData")

# 最优超参数的交叉验证折线图图示
eval_best_cv10_rf %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证ROC图示

tune_rf %>%
  collect_predictions() %>%
  inner_join(hpbest_rf[, 1:3]) %>%
  group_by(id) %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(size = 1) +
  theme_classic()

#################xgboost
# 训练模型
# 设定模型
model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25 #提前终止迭代次数
) %>%
  set_args(validation = 0.2)
model_xgboost

# workflow
wk_xgboost <- 
  workflow() %>%
  add_model(model_xgboost) %>%
  add_formula(outcome ~ .)
wk_xgboost
# 重抽样设定-n折交叉验证
set.seed(2023)
folds <- vfold_cv(traindata, v =10)
folds

# 超参数寻优范围
hpset_xgboost <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
)
#hpgrid_xgboost <- 
#grid_regular(hpset_xgboost, levels = c(3, 2, 2, 3, 2, 2)) #每个超参数的备选值个数，这一步可以省略，因为下一步的存在。
set.seed(2023)
hpgrid_xgboost <- grid_random(hpset_xgboost, size = 5) #生成五组超参数
hpgrid_xgboost

# 交叉验证随机搜索过程
set.seed(2023)
tune_xgboost <- wk_xgboost %>%
  tune_grid(resamples = folds,
            grid = hpgrid_xgboost,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_xgboost)
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost
# 经过交叉验证得到的最优超参数
hpbest_xgboost <- tune_xgboost %>%
  select_best(metric = "roc_auc")
hpbest_xgboost

# 采用最优超参数组合训练最终模型
set.seed(2023)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(traindata)
final_xgboost

# 提取最终的算法模型
final_xgboost2 <- final_xgboost %>%
  extract_fit_engine()

importance_matrix <- xgb.importance(model = final_xgboost2)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Cover", #也可以通过gain,frequency
                    col = "skyblue")
# SHAP
colnames(traindata)
xgb.plot.shap(data = as.matrix(traindata[,-1]), #去除traindata中因变量
              model = final_xgboost2,
              top_n = 20)
###############################################################

# 应用模型-预测训练集
predtrain_xgboost <- final_xgboost %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "xgboost")
predtrain_xgboost 
# 评估模型ROC曲线-训练集上
levels(traindata$outcome)
roctrain_xgboost <- predtrain_xgboost %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_xgboost
autoplot(roctrain_xgboost)

# 约登法则对应的p值
yueden_xgboost <- roctrain_xgboost %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_xgboost
# 预测概率+约登法则=预测分类
predtrain_xgboost2 <- predtrain_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_xgboost, "Yes", "No")))
predtrain_xgboost2
# 混淆矩阵
cmtrain_xgboost <- predtrain_xgboost2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_xgboost

autoplot(cmtrain_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_xgboost <- cmtrain_xgboost %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_xgboost %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_xgboost

###############################################################

# 应用模型-预测测试集
predtest_xgboost <- final_xgboost %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "xgboost")
predtest_xgboost

# 评估模型ROC曲线-测试集上
roctest_xgboost <- predtest_xgboost %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
roctest_xgboost
autoplot(roctest_xgboost)

setwd("E:/R work/EV/results/eval")
write.table (roctest_xgboost, file ="roctest_xgboost.csv", sep =",", row.names =FALSE)

# 预测概率+约登法则=预测分类
predtest_xgboost2 <- predtest_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_xgboost, "Yes", "No")))
predtest_xgboost2
# 混淆矩阵
cmtest_xgboost <- predtest_xgboost2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_xgboost

autoplot(cmtest_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_xgboost <- cmtest_xgboost %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_xgboost %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_xgboost

##############################################################
# 合并训练集和测试集上ROC曲线
roctrain_xgboost %>%
  bind_rows(roctest_xgboost) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()

# 合并训练集和测试集上性能指标
eval_xgboost <- eval_train_xgboost %>%
  bind_rows(eval_test_xgboost) %>%
  mutate(model = "xgboost")
eval_xgboost

setwd("E:/R work/EV/results/eval")
write.table (eval_xgboost, file ="eval_xgboost.csv", sep =",", row.names =FALSE)
#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_xgboost <- eval_tune_xgboost %>%
  inner_join(hpbest_xgboost[, 1:6])
eval_best_cv_xgboost

# 最优超参数的交叉验证指标具体结果
eval_best_cv10_xgboost <- tune_xgboost %>%
  collect_predictions() %>%
  inner_join(hpbest_xgboost[, 1:6]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes, event_level="second") %>%
  ungroup() %>%
  mutate(model = "xgboost") %>%
  inner_join(eval_best_cv_xgboost[c(7,9,11)])
eval_best_cv10_xgboost

# 保存评估结果
setwd("E:/R work/EV/cls2")
save(final_xgboost,
     predtrain_xgboost,
     predtest_xgboost,
     eval_xgboost,
     eval_best_cv10_xgboost, 
     file = "evalresult_xgboost.RData")

# 最优超参数的交叉验证指标图示
eval_best_cv10_xgboost %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证图示
tune_xgboost %>%
  collect_predictions() %>%
  inner_join(hpbest_xgboost[, 1:6]) %>%
  group_by(id) %>%
  roc_curve(outcome, .pred_Yes, event_level="second") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(size = 1) +
  theme_bw()

###################################################################
####################决策树模型
# 训练模型
# 设定模型
model_dt <- decision_tree(
  mode = "classification",
  engine = "rpart",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_args(model=TRUE)
model_dt

# workflow
wk_dt <- 
  workflow() %>%
  add_model(model_dt) %>%
  add_formula(outcome~ .) #因为前面已经传递进来了
wk_dt

# 重抽样设定-n折交叉验证
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# 超参数寻优范围
hpset_dt <- parameters(tree_depth(range = c(3, 7)),
                       min_n(range = c(5, 10)),
                       cost_complexity(range = c(-6, -1))) 
# hpgrid_dt <- grid_regular(hpset_dt, levels = c(3, 2, 4)) #method1这是设置每个超参数的取值数量，最后生成3*2*4个模型
set.seed(2023)
hpgrid_dt <- grid_random(hpset_dt, size = 10)#method2这是生成10组超参数
hpgrid_dt #并不在（-6，-1）范围内
log10(hpgrid_dt$cost_complexity) #对数转换后在该范围内


# 交叉验证网格搜索过程
set.seed(2023)
tune_dt <- wk_dt %>%
  tune_grid(resamples = folds,
            grid = hpgrid_dt,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_dt)
eval_tune_dt <- tune_dt %>%
  collect_metrics()
eval_tune_dt

# 经过交叉验证得到的最优超参数
hpbest_dt <- tune_dt %>%
  select_by_one_std_err(metric = "roc_auc", desc(cost_complexity))#所指定参数的一个标注误以内使试模型最简单的的那个超参数，并且按照复杂度降序排列，复杂度越大模型越简单。
hpbest_dt

# 采用最优超参数组合训练最终模型
final_dt <- wk_dt %>%
  finalize_workflow(hpbest_dt) %>%
  fit(traindata)
final_dt

# 提取最终的算法模型
final_dt2 <- final_dt %>%
  extract_fit_engine()
library(rpart.plot)
rpart.plot(final_dt2) #画决策树的图

final_dt2$variable.importance
par(mar = c(10, 3, 1, 1))
barplot(final_dt2$variable.importance, las = 1) #las=2使横坐标的标签竖直

##############################################################

# 应用模型-预测训练集
predtrain_dt <- final_dt %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "dt")
predtrain_dt #pred.no和pred.yes要根据自己的数据改。

# 评估模型ROC曲线-训练集上
contrasts(traindata$outcome)#横轴上为0对应的是基准水平，为first。
roctrain_dt <- predtrain_dt %>%
  # roc_curve(AHD, .pred_Yes, event_level = "second") %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_dt
autoplot(roctrain_dt)

# 约登法则对应的p值
yueden_dt <- roctrain_dt %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_dt
# 预测概率+约登法则=预测分类
predtrain_dt2 <- predtrain_dt %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_dt, "Yes", "No")))
predtrain_dt2

# 混淆矩阵
cmtrain_dt <- predtrain_dt2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_dt

autoplot(cmtrain_dt, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
eval_train_dt <- cmtrain_dt %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_dt %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_dt

##############################################################

# 应用模型-预测测试集
predtest_dt <- final_dt %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "dt")
predtest_dt
# 评估模型ROC曲线-测试集上
roctest_dt <- predtest_dt %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
roctest_dt
autoplot(roctest_dt)

# 预测概率+约登法则=预测分类
predtest_dt2 <- predtest_dt %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_dt, "Yes", "No")))
predtest_dt2

# 混淆矩阵
cmtest_dt <- predtest_dt2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_dt

autoplot(cmtest_dt, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_dt <- cmtest_dt %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_dt %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_dt

##############################################################
# 合并训练集和测试集上ROC曲线
roctrain_dt %>%
  bind_rows(roctest_dt) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()

# 合并训练集和测试集上性能指标
eval_dt <- eval_train_dt %>%
  bind_rows(eval_test_dt) %>%
  mutate(model = "dt")
eval_dt
View(eval_dt)

setwd("E:/R work/EV/results/eval")
write.table (eval_dt, file ="eval_dt.csv", sep =",", row.names =FALSE)
#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_dt <- eval_tune_dt %>%
  inner_join(hpbest_dt[, 1:3])
eval_best_cv_dt

# 最优超参数的交叉验证指标具体结果
eval_best_cv10_dt <- tune_dt %>%
  collect_predictions() %>%
  inner_join(hpbest_dt[, 1:3]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes, event_level="second") %>%
  ungroup() %>%
  mutate(model = "dt") %>%
  inner_join(eval_best_cv_dt[c(4,6,8)]) #某一列
eval_best_cv10_dt

# 保存评估结果
setwd("E:/R work/EV/cls2")
save(final_dt,
     predtrain_dt,
     predtest_dt,
     eval_dt,
     eval_best_cv10_dt, 
     file = "evalresult_dt.RData")

# 最优超参数的交叉验证指标图示
eval_best_cv10_dt %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证图示
tune_dt %>%
  collect_predictions() %>%
  inner_join(hpbest_dt[, 1:3]) %>%
  group_by(id) %>%
  roc_curve(outcome, .pred_Yes, event_level="second") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(size = 1) +
  theme_bw()

#########################SVM
# 训练模型
# 设定模型
model_rsvm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
)
model_rsvm

# workflow
wk_rsvm <- 
  workflow() %>%
  add_model(model_rsvm) %>%
  add_formula(outcome ~ .)
wk_rsvm

# 重抽样设定-10折交叉验证
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# 超参数寻优范围
hpset_rsvm <- parameters(cost(range = c(-5, 5)), 
                         rbf_sigma(range = c(-4, -1)))
hpgrid_rsvm <- grid_regular(hpset_rsvm, levels = c(2,3))
hpgrid_rsvm
# 交叉验证网格搜索过程
library(kernlab)
set.seed(2023)
tune_rsvm <- wk_rsvm %>%
  tune_grid(resamples = folds,
            grid = hpgrid_rsvm,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_rsvm)
eval_tune_rsvm <- tune_rsvm %>%
  collect_metrics()
eval_tune_rsvm

# 经过交叉验证得到的最优超参数
hpbest_rsvm <- tune_rsvm %>%
  select_best(metric = "roc_auc")
hpbest_rsvm

# 采用最优超参数组合训练最终模型
final_rsvm <- wk_rsvm %>%
  finalize_workflow(hpbest_rsvm) %>%
  fit(traindata)
final_rsvm

# 提取最终的算法模型
final_rsvm %>%
  extract_fit_engine()

###############################################################

# 应用模型-预测训练集
predtrain_rsvm <- final_rsvm %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "rsvm")
predtrain_rsvm
# 评估模型ROC曲线-训练集上
levels(traindata$outcome)
roctrain_rsvm <- predtrain_rsvm %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_rsvm
autoplot(roctrain_rsvm)

# 约登法则对应的p值
yueden_rsvm <- roctrain_rsvm %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_rsvm

# 预测概率+约登法则=预测分类
predtrain_rsvm2 <- predtrain_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_rsvm, "Yes", "No")))
predtrain_rsvm2
# 混淆矩阵
cmtrain_rsvm <- predtrain_rsvm2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)

cmtrain_rsvm

autoplot(cmtrain_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_rsvm <- cmtrain_rsvm %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_rsvm %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_rsvm

##############################################################
# 应用模型-预测测试集
predtest_rsvm <- final_rsvm %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "rsvm")
predtest_rsvm
# 评估模型ROC曲线-测试集上
roctest_rsvm <- predtest_rsvm %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
roctest_rsvm
autoplot(roctest_rsvm)


# 预测概率+约登法则=预测分类
predtest_rsvm2 <- predtest_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_rsvm, "Yes", "No")))
predtest_rsvm2
# 混淆矩阵
cmtest_rsvm <- predtest_rsvm2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_rsvm

autoplot(cmtest_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_rsvm <- cmtest_rsvm %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_rsvm %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_rsvm
##############################################################

# 合并训练集和测试集上ROC曲线
roctrain_rsvm %>%
  bind_rows(roctest_rsvm) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))

# 合并训练集和测试集上性能指标
eval_rsvm <- eval_train_rsvm %>%
  bind_rows(eval_test_rsvm) %>%
  mutate(model = "rsvm")
eval_rsvm

setwd("E:/R work/EV/results/eval")
write.table (eval_rsvm, file ="eval_rsvm.csv", sep =",", row.names =FALSE)

View(eval_rsvm)
#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_rsvm <- eval_tune_rsvm %>%
  inner_join(hpbest_rsvm[, 1:2])
eval_best_cv_rsvm

# 最优超参数的交叉验证指标具体结果
eval_best_cv10_rsvm <- tune_rsvm %>%
  collect_predictions() %>%
  inner_join(hpbest_rsvm[, 1:2]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes,event_level="second") %>%
  ungroup() %>%
  mutate(model = "rsvm") %>%
  inner_join(eval_best_cv_rsvm[c(3,5,7)])
eval_best_cv10_rsvm

# 保存评估结果
setwd("E:/R work/EV/cls2")
save(final_rsvm,
     predtrain_rsvm,
     predtest_rsvm,
     eval_rsvm,
     eval_best_cv10_rsvm, 
     file = "evalresult_rsvm.RData")
# 最优超参数的交叉验证指标图示
eval_best_cv10_rsvm %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证图示
tune_rsvm %>%
  collect_predictions() %>%
  inner_join(hpbest_rsvm[, 1:2]) %>%
  group_by(id) %>%
  roc_curve(outcome, .pred_Yes,event_level="second") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(size = 1) +
  theme_bw()

###############################mlp
#############################################################
# 训练模型
# 设定模型
model_mlp <- mlp(
  mode = "classification",
  engine = "nnet",
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_args(MaxNWts = 5000)
model_mlp

# workflow
wk_mlp <- 
  workflow() %>%
  add_model(model_mlp) %>%
  add_formula(outcome ~ .)
wk_mlp

# 重抽样设定-10折交叉验证
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# 超参数寻优范围
hpset_mlp <- parameters(hidden_units(range = c(15, 24)),
                        penalty(range = c(-3, 0)),
                        epochs(range = c(50, 150)))
hpgrid_mlp <- grid_regular(hpset_mlp, levels = 2)
hpgrid_mlp

# 交叉验证网格搜索过程
set.seed(2023)
tune_mlp <- wk_mlp %>%
  tune_grid(resamples = folds,
            grid = hpgrid_mlp,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_mlp)
eval_tune_mlp <- tune_mlp %>%
  collect_metrics()
eval_tune_mlp

# 经过交叉验证得到的最优超参数
hpbest_mlp <- tune_mlp %>%
  select_best(metric = "roc_auc")
hpbest_mlp

# 采用最优超参数组合训练最终模型
set.seed(2023)
final_mlp <- wk_mlp %>%
  finalize_workflow(hpbest_mlp) %>%
  fit(traindata)
final_mlp

# 提取最终的算法模型
final_mlp2 <- final_mlp %>%
  extract_fit_engine()

library(NeuralNetTools)
plotnet(final_mlp2)
garson(final_mlp2) +
  coord_flip()
olden(final_mlp2) +
  coord_flip()

#############################################################

# 应用模型-预测训练集
predtrain_mlp <- final_mlp %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "mlp")
predtrain_mlp
# 评估模型ROC曲线-训练集上
levels(traindata$outcome)
roctrain_mlp <- predtrain_mlp %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_mlp
autoplot(roctrain_mlp)

# 约登法则对应的p值
yueden_mlp <- roctrain_mlp %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_mlp
# 预测概率+约登法则=预测分类
predtrain_mlp2 <- predtrain_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_mlp, "Yes", "No")))
predtrain_mlp2
# 混淆矩阵
cmtrain_mlp <- predtrain_mlp2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_mlp

autoplot(cmtrain_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_mlp <- cmtrain_mlp %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_mlp %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_mlp

#############################################################

# 应用模型-预测测试集
predtest_mlp <- final_mlp %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "mlp")
predtest_mlp
# 评估模型ROC曲线-测试集上
roctest_mlp <- predtest_mlp %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
roctest_mlp
autoplot(roctest_mlp)


# 预测概率+约登法则=预测分类
predtest_mlp2 <- predtest_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_mlp, "Yes", "No")))
predtest_mlp2
# 混淆矩阵
cmtest_mlp <- predtest_mlp2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_mlp

autoplot(cmtest_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_mlp <- cmtest_mlp %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_mlp %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_mlp

#############################################################

# 合并训练集和测试集上ROC曲线
roctrain_mlp %>%
  bind_rows(roctest_mlp) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))

# 合并训练集和测试集上性能指标
eval_mlp <- eval_train_mlp %>%
  bind_rows(eval_test_mlp) %>%
  mutate(model = "mlp")
eval_mlp

setwd("E:/R work/EV/results/eval")
write.table (eval_mlp, file ="eval_mlp.csv", sep =",", row.names =FALSE)
#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_mlp <- eval_tune_mlp %>%
  inner_join(hpbest_mlp[, 1:3])
eval_best_cv_mlp

# 最优超参数的交叉验证指标具体结果
eval_best_cv10_mlp <- tune_mlp %>%
  collect_predictions() %>%
  inner_join(hpbest_mlp[, 1:3]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes,event_level="second") %>%
  ungroup() %>%
  mutate(model = "mlp") %>%
  inner_join(eval_best_cv_mlp[c(4,6,8)])
eval_best_cv10_mlp

# 保存评估结果
setwd("E:/R work/EV/cls2")
save(final_mlp,
     predtrain_mlp,
     predtest_mlp,
     eval_mlp,
     eval_best_cv10_mlp, 
     file = "evalresult_mlp.RData")

# 最优超参数的交叉验证指标图示
eval_best_cv10_mlp %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 最优超参数的交叉验证图示
tune_mlp %>%
  collect_predictions() %>%
  inner_join(hpbest_mlp[, 1:3]) %>%
  group_by(id) %>%
  roc_curve(outcome, .pred_Yes,event_level="second") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(size = 1) +
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.3), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))

############################模型比较
# 加载各个模型的评估结果
library(tidymodels)
setwd("E:/R work/EV")
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

#############################################################
# 各个模型在测试集上的误差指标
eval <- bind_rows(
  eval_logistic, 
  eval_rf,
  eval_dt,
  eval_xgboost,
  eval_rsvm,
  eval_mlp)
View(eval)

# 平行线图，修改参数可以看后面的合并代码。
eval %>%
  filter(dataset == "test") %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) + #,shape=model,linetype=model
  geom_point(size=2.5) + #origin：geom_point() +
  geom_line(size=1.2,aes(group = model)) + #origin:geom_line(aes(group = model)) +
  scale_y_continuous(limits = c(0.20, 1))+ 
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(angle = 30,size=12,hjust = 1), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c("#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d")) 
#上述的颜色对应从上向下的模型，DT/LR/MLP/RF/SVM/XGB，但是下面画DCA曲线时，红色是#e8490f，黄色是#ffd401。注意跟DCA曲线保持一致！虽然颜色代码不一样，但是结果是一样的！
#除了红色，其他颜色与DCA曲线是完全一样的，颜色在改的时候上面改了下面对所对应的也要改（最后把红色同时也改了），保持一致。
# 各个模型在测试集上的误差指标表格
eval2 <- eval %>%
  select(-.estimator) %>%
  filter(dataset == "test") %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval2
# 各个模型在测试集上的roc指标条形图
eval2 %>%
  ggplot(aes(x = model, y = roc_auc,fill=model)) + #roc_auc也可以改成其他参数
  geom_col(width = 0.3, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)),  #roc_auc也可以改成其他参数
            nudge_y = -0.03) +
 theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12,hjust = 1), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))+
  scale_fill_manual(values=c("#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))

#############################################################

# 各个模型在测试集上的预测概率
predtest <- bind_rows(
  predtest_logistic,
  predtest_rf,
  predtest_dt,
  predtest_xgboost,
  predtest_rsvm,
  predtest_mlp
)
predtest

# 各个模型在测试集上的ROC
predtest %>%
  group_by(model) %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = model)) +
  geom_path(size = 1) +
  theme_bw()+ #classic
  theme(legend.title = element_blank(),
        legend.position = c(0.8,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))

#各个模型测试集rocdelong检验
library(pROC)
roc1 <- roc(predtest_logistic$outcome, predtest_logistic$.pred_Yes,levels=c("No", "Yes"))
roc2 <- roc(predtest_rf$outcome, predtest_rf$.pred_Yes,levels=c("No", "Yes"))
roc3 <- roc(predtest_xgboost$outcome, predtest_xgboost$.pred_Yes,levels=c("No", "Yes"))
roc4<- roc(predtest_dt$outcome, predtest_dt$.pred_Yes,levels=c("No", "Yes"))
roc5<- roc(predtest_rsvm$outcome, predtest_rsvm$.pred_Yes,levels=c("No", "Yes"))
roc6<- roc(predtest_mlp$outcome, predtest_mlp$.pred_Yes,levels=c("No", "Yes"))

#delong检验（与上述相互验证）
roc.test(roc3,roc1,method = 'delong')     
roc.test(roc3,roc2,method = 'delong') 
roc.test(roc3,roc6,method = 'delong')
roc.test(roc3,roc5,method = 'delong') 
roc.test(roc3,roc4,method = 'delong') 

#xgboost置信区间
proc<-roc(predtest_xgboost$outcome,predtest_xgboost$.pred_Yes,levels=c("No", "Yes"),ci=T);A=proc$ci;A 
# 各个模型交叉验证的各折指标点线图
evalcv <- bind_rows(
  eval_cv10_logistic, eval_best_cv10_rf,eval_best_cv10_dt,eval_best_cv10_xgboost,eval_best_cv10_rsvm,eval_best_cv10_mlp)
evalcv
View(evalcv)

#setwd("E:/R work/EV/results/eval")
#write.table (evalcv, file ="evalcv.csv", sep =",", row.names =FALSE)

evalcv %>%
  ggplot(aes(x = id, y = .estimate, 
             group = model, color = model)) +  #,shape=model,linetype=model
  #geom_point() +
  #geom_line() +
  geom_point(size=2.5)+
  geom_line(size=1.2)+
  scale_y_continuous(limits = c(0.2, 1))+ 
  labs(x = "Round of cross", y = "AUC") +
  theme_bw()+ #classic
  theme(legend.title = element_blank(),
        legend.position = c(0.7,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))

# 各个模型交叉验证的指标平均值图(带上下限)
evalcv %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean,color=model)) +  #shape=model
  geom_point(size = 2) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-std_err, 
                    ymax = mean+std_err),
                width = 0.1, size = 1.2) +
  scale_y_continuous(limits = c(0.5, 1)) +
  labs(y = "cv roc_auc") +
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))

#各个模型验证集rocdelong检验
library(pROC)
roc1 <- roc(predtrain_logistic$outcome, predtrain_logistic$.pred_Yes,levels=c("No", "Yes"))
roc2 <- roc(predtrain_rf$outcome, predtrain_rf$.pred_Yes,levels=c("No", "Yes"))
roc3 <- roc(predtrain_xgboost$outcome, predtrain_xgboost$.pred_Yes,levels=c("No", "Yes"))
roc4<- roc(predtrain_dt$outcome, predtrain_dt$.pred_Yes,levels=c("No", "Yes"))
roc5<- roc(predtrain_rsvm$outcome, predtrain_rsvm$.pred_Yes,levels=c("No", "Yes"))
roc6<- roc(predtrain_mlp$outcome, predtrain_mlp$.pred_Yes,levels=c("No", "Yes"))
#delong检验（与上述相互验证）
roc.test(roc2,roc1,method = 'delong')     
roc.test(roc2,roc3,method = 'delong') 
roc.test(roc2,roc4,method = 'delong')
roc.test(roc2,roc5,method = 'delong') 
roc.test(roc2,roc6,method = 'delong') 

# 各个模型在验证集上的预测概率
predtrain <- bind_rows(
  predtrain_logistic,
  predtrain_rf,
  predtrain_dt,
  predtrain_xgboost,
  predtrain_rsvm,
  predtrain_mlp
)
predtrain

# 各个模型在验证集上的ROC
predtrain %>%
  dplyr::group_by(model) %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = model)) +
  geom_path(size = 1) +
  theme_bw()+ #classic
  theme(legend.title = element_blank(),
        legend.position = c(0.8,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))



#######################################画几个模型的calibration图
# 各个模型在测试集上的预测概率2，只需要预测为1或0的概率，因此需要去除一列。
predtest2 <- predtest %>%
  dplyr::select(-.pred_No) %>% #去除一个概率
  mutate(id = rep(1:nrow(predtest_logistic), 6)) %>% #有几个模型就写几
  pivot_wider(names_from = model, values_from = .pred_Yes) #保留一个概率
View(predtest2)
##下面的是前一段的原始的
# 各个模型在测试集上的预测概率2
#predtest2 <- predtest %>%
#  select(-.pred_No) %>%
 ## mutate(id = rep(1:nrow(predtest_logistic), 7)) %>%
 # pivot_wider(names_from = model, values_from = .pred_Yes)
#predtest2
# 各个模型在测试集上的校准曲线
# http://topepo.github.io/caret/measuring-performance.html#calibration-curves
cal_obj <- caret::calibration(
  as.formula(paste0("outcome ~ ", 
                    paste(colnames(predtest2)[4:9], collapse = " + "))), #4:9是指模型
  data = predtest2,
  class = "Yes",
  cuts = 11
)

cal_obj$data

plot(cal_obj, type = "b", pch = 20,
     auto.key = list(columns = 3,
                     lines = T,
                     points = T))

#calibration分成两个图
cal_obj$data %>%
  filter(Percent != 0) %>%
  ggplot(aes(x = midpoint, y = Percent,
             color=model)) +
  geom_point(color = "brown1") +
  geom_line(color = "brown1") +
  geom_abline(slope = 1, intercept = 0) +
  facet_wrap(~calibModelVar) +
  theme_bw()

########calibration另一种方式
library(PredictABEL)
predtest4 <- predtest2
predtest4$outcome <- ifelse(predtest4$outcome == "Yes", 1, 0)
calall <- data.frame()
for (i in 4:9) {
  cal <- plotCalibration(data = as.data.frame(predtest4),
                         cOutcome = 1,
                         predRisk = predtest4[[i]])
  caldf <- cal$Table_HLtest %>%
    as.data.frame() %>%
    rownames_to_column("pi") %>%
    mutate(model = colnames(predtest4)[i])
  calall <- rbind(calall, caldf)
}

#画的图丑
calall %>%
  ggplot(aes(x = meanpred, y = meanobs, color=model)) +
  geom_point(color = "brown1") +
  geom_line(color = "brown1") +
  geom_abline(slope = 1, intercept = 0) +
  facet_wrap(~model) +
  theme_bw()
#画的图美
ggplot(calall,aes(x = meanpred,y = meanobs,color=model))+
  geom_point(size=2.5)+
  geom_line(linetype = 1,size=1.2)+
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "black")+
  labs(x="predicted_possibility",y = "actual_possibility",title = "calibration_curve")+
  theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))


#######################################################################

# 各个模型在测试集上的DCA
# https://cran.r-project.org/web/packages/dcurves/vignettes/dca.html
dca_obj <- dcurves::dca(
  as.formula(paste0("outcome ~ ", 
                    paste(colnames(predtest2)[4:9], collapse = " + "))),
  data = predtest2,
  thresholds = seq(0, 1, by = 0.01)
)

plot(dca_obj, smooth = T)+
  theme_classic()+ #theme_bw
  theme(legend.title = element_blank(),
        legend.position = c(0.85,0.80), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c("#BDBDBD","#000000","#9ec417","#44c1f0","#e8490f","#9b3a74","#3f60aa","#ffd401"))

#对应"logistic" "rf" "dt" "xgboost" "rsvm""mlp"，c("#9ec417","#44c1f0","#e8490f","#9b3a74","#3f60aa","#ffd401"))
#绿蓝红紫深蓝黄
####################################################3
# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---模型解释
# 训练好的模型对象，建议先运行上面的训练模型，再运行下面的。
# load(file.choose())
# logistic回归
object_model <- fit_logistic
# LASSO-岭回归-弹性网络
object_model <- final_enet
# 决策树
object_model <- final_dt
# 随机森林
object_model <- final_rf
# xgboost
object_model <- final_xgboost
# 支持向量机
object_model <- final_rsvm
# 单隐藏层神经网络
object_model <- final_mlp

# 自变量数据集
colnames(traindata)
traindatax <- traindata[,-1] #去除因变量在的那一列
colnames(traindatax)

# iml包
library(iml)
predictor_model <- Predictor$new(
  object_model, 
  data = traindatax,
  y = traindata$outcome,
  predict.function = function(model, newdata){
    predict(model, newdata, type = "prob") %>%
      pull(2) #因为outcome有0,1两列，提取1这一列，也就是第二列pull(2)
  }
)

# 变量重要性-基于置换(随意置换几个变量后再评估模型的AUC，最后得到的重要性)，和之前自带randomforest包变量重要性不同。
imp_model <- FeatureImp$new(
  predictor_model, 
  loss = function(actual, predicted){
    return(1-Metrics::auc(as.numeric(actual=="Yes"), predicted)) #实际为1和predict为1的
  }
)

# 数值
imp_model$results
# 图示
imp_model$plot() +
  theme_bw()


# 单变量效应
 #predictor_model <- Predictor$new( #运行这一串会有两个0,1偏依赖图。
  # object_model, 
   #data = traindatax,
   #y = traindata$outcome,
   #type="prob"
 #)
pdp_model <- FeatureEffect$new( #在iml包可以看解释
  predictor_model, 
  feature = "SD",
  method = "pdp"
)
# 数值
pdp_model$results
# 图示
pdp_model$plot() +
  theme_bw()

# 所有变量的效应全部输出
effs_model <- FeatureEffects$new(predictor_model, method = "pdp")
# 数值
effs_model$results
# 图示
effs_model$plot()

# 单样本shap分析
shap_model <- Shapley$new(
  predictor_model, 
  x.interest = traindatax[1,] #使用traidatax的第一个样本
)
# 数值
shap_model$results
# 图示
shap_model$plot() +
  theme_bw()

# 基于所有样本的shap分析
# fastshap包
library(fastshap)#运行shap可以看到
shap <- explain(
  object_model, 
  X = as.data.frame(traindatax),
  nsim = 10,
  adjust = T,
  pred_wrapper = function(model, newdata) {
    predict(model, newdata, type = "prob") %>% pull(2)
  }
)

# 单样本图示,找testset里的变量，需要把自变量数据集以及上面的shap改成测试集的。
force_plot(object = shap[512L, ], #第几个样本
           feature_values = as.data.frame(traindatax)[512L, ], #第几个样本
            #baseline = mean(predtrain_model$.pred_Yes), # 换成具体模型预测结果
           display = "viewer") 
View(traindata)
# 变量重要性
autoplot(shap, fill = "skyblue") +
  theme_bw()

data1 <- shap %>%
  as.data.frame() %>%
  dplyr::mutate(id = 1:n()) %>%
  pivot_longer(cols = -(ncol(traindatax)+1), values_to = "shap")

data2 <- traindatax  %>%
  dplyr::mutate(id = 1:n()) %>%
  pivot_longer(cols = -(ncol(traindatax)+1))

shapimp <- data1 %>%
  dplyr::group_by(name) %>%
  dplyr:: summarise(shap.abs.mean = mean(abs(shap))) %>%
  dplyr::arrange(shap.abs.mean) %>%
  dplyr::mutate(name = forcats::as_factor(name))

# 所有变量shap图示
library(ggbeeswarm)
library(ggridges)
OO<-data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  dplyr::group_by(feature) %>%
  dplyr::mutate(value = (value - min(value)) / (max(value) - min(value)), #尺度缩放，都放到0，1之间
         feature = factor(feature, levels = levels(shapimp$name)))
#看value是否缩放到0，1之间
OO%>% 
    ggplot(aes(x = value, y = feature)) + #x=shap也可以
  geom_boxplot()
#看value是否缩放到0，1之间
OO%>% 
  ggplot(aes(x = value, y = feature,fill=feature)) +#x=shap也可以
  ggridges::geom_density_ridges()+
  ggridges::theme_ridges()

OO%>% 
  ggplot(aes(x = shap, y = feature,color=value))+
  # geom_jitter(size = 2, height = 0.2, width = 0) +，使散点图纵向更长
  geom_quasirandom(width = 0.2) +
  scale_color_gradient(
    low = "#FFCC33", #FFCC33
    high = "#6600CC", #6600CC
    breaks = c(0, 1), 
    labels = c("Low", "High "), 
    guide = guide_colorbar(barwidth = 1, barheight = 10)
  ) +
  labs(x = "SHAP value", color = "Feature value") +
  theme_bw()+
  theme(axis.text.x =element_text(size=12,hjust = 1), #坐标轴标签字体大小
                   axis.text.y=element_text(size=12))
  

# 单变量shap图示,分类变量不适用？
data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  filter(feature == "hdlc") %>%
  ggplot(aes(x = value, y = shap)) +
  geom_point(color = "#f46f20") + #改点的颜色
  geom_smooth(se = T, span = 0.5,colour='#13a983') + # '中'可以改颜色
  labs(x = "hdlc")+
  theme_classic()+
  scale_fill_manual(values = c("#00AFBB"))+
  theme(axis.text.x =element_text(size=12,hjust = 1), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))

###############################其他变量与xgboost模型相比
# 数据预处理
# 先对照训练集写配方
datarecipetra <- recipe(outcome ~ NLR+PSDR+FIB4+APRI+AAR+tc+hdlc, dev) %>%
  #step_rm(Id) %>% 剔除模型中的无关变量
  step_naomit(all_predictors(), skip = F) %>% #剔除缺失值
  step_dummy(all_nominal_predictors()) %>% #对所有分类变量进行独热编码
  prep()
datarecipetra  #因为这个公式已经代表了上述6个变量，所以下面可以直接用.
#按方处理训练集和测试集
traindatatra  <- bake(datarecipetra , new_data = NULL) %>%
  select(outcome, everything())
testdatatra <- bake(datarecipetra , new_data =vad) %>%
  select(outcome, everything())
# 数据预处理后数据概况，这一步转化以后只有outcome是factor，其他的是numeric变量
skimr::skim(traindatatra)
skimr::skim(testdatatra)
#################xgboost
# 训练模型
# 设定模型
model_xgboosttra <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25 #提前终止迭代次数
) %>%
  set_args(validation = 0.2)
model_xgboosttra

# workflow
wk_xgboosttra <- 
  workflow() %>%
  add_model(model_xgboosttra) %>%
  add_formula(outcome ~ .)
wk_xgboosttra
# 重抽样设定-n折交叉验证
set.seed(2023)
foldstra <- vfold_cv(traindatatra, v =10)
foldstra

# 超参数寻优范围
hpset_xgboosttra <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
)

#hpgrid_xgboost <- 
#grid_regular(hpset_xgboost, levels = c(3, 2, 2, 3, 2, 2)) #每个超参数的备选值个数，这一步可以省略，因为下一步的存在。
set.seed(2023)
hpgrid_xgboosttra <- grid_random(hpset_xgboosttra, size = 5) #生成五组超参数
hpgrid_xgboosttra

# 交叉验证随机搜索过程
set.seed(2023)
tune_xgboosttra <- wk_xgboosttra %>%
  tune_grid(resamples = foldstra,
            grid = hpgrid_xgboosttra,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_xgboosttra)
eval_tune_xgboosttra <- tune_xgboosttra %>%
  collect_metrics()
eval_tune_xgboosttra

# 经过交叉验证得到的最优超参数
hpbest_xgboosttra <- tune_xgboosttra %>%
  select_best(metric = "roc_auc")
hpbest_xgboosttra

# 采用最优超参数组合训练最终模型
set.seed(2023)
final_xgboosttra <- wk_xgboosttra %>%
  finalize_workflow(hpbest_xgboosttra) %>%
  fit(traindatatra)
final_xgboosttra

###############################################################
# 应用模型-预测训练集
predtrain_xgboosttra <- final_xgboosttra %>%
  predict(new_data = traindatatra, type = "prob") %>%
  bind_cols(traindatatra %>% select(outcome)) %>%
  mutate(dataset = "traintra")
predtrain_xgboosttra 
# 评估模型ROC曲线-训练集上
levels(traindatatra$outcome)
roctrain_xgboosttra <- predtrain_xgboosttra%>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "traintra")
roctrain_xgboosttra
autoplot(roctrain_xgboosttra)

#跟我们构建的模型比较ROC曲线之训练集
roctrain_xgboost %>%
  bind_rows(roctrain_xgboosttra) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.8,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))

#各个模型验证集rocdelong检验,可以看AUC
library(pROC)
roc1 <- roc(predtrain_xgboost$outcome, predtrain_xgboost$.pred_Yes,levels=c("No", "Yes"))
roc2 <- roc(predtrain_xgboosttra$outcome, predtrain_xgboosttra$.pred_Yes,levels=c("No", "Yes"))
#delong检验（与上述相互验证）
roc.test(roc2,roc1,method = 'delong')   

###############################################################
# 应用模型-预测测试集
predtest_xgboosttra <- final_xgboosttra %>%
  predict(new_data = testdatatra, type = "prob") %>%
  bind_cols(testdatatra %>% select(outcome)) %>%
  mutate(dataset = "testtra") %>%
  mutate(model = "xgboost")
predtest_xgboosttra
# 评估模型ROC曲线-测试集上
roctest_xgboosttra <- predtest_xgboosttra %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "testtra")
roctest_xgboosttra
autoplot(roctest_xgboosttra)

#跟我们构建的模型比较ROC曲线之测试集
roctest_xgboost %>%
  bind_rows(roctest_xgboosttra) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.8,0.2), #图例位置
        legend.text= element_text(size=14),#图例大小
        axis.text.x =element_text(size=12), #坐标轴标签字体大小
        axis.text.y=element_text(size=12))
#各个模型验证集rocdelong检验,可以看AUC
library(pROC)
roc1 <- roc(predtest_xgboost$outcome, predtest_xgboost$.pred_Yes,levels=c("No", "Yes"))
roc2 <- roc(predtest_xgboosttra$outcome, predtest_xgboosttra$.pred_Yes,levels=c("No", "Yes"))
#delong检验（与上述相互验证）
roc.test(roc2,roc1,method = 'delong')   
