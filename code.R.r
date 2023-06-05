#ԭʼ����ȱʧ����
setwd("E:/R work/EV")
rawdata<-read.table("3.CSV",header=T, sep=",")
rawdata<-as.matrix(rawdata)
library(mice)
summary(rawdata)
#��������ȱʧ������ԭ���ϣ�����ȱʧ��������5%�����޷����в岹��
pMiss <- function(x){
  sum(is.na(x))/length(x)*100
}

#apply()������1��ʾ�У�2��ʾ��
apply(rawdata,2,pMiss)
apply(rawdata,1,pMiss)

#�б���ʾȱʧ���
md.pattern(rawdata)

#��ͼ��ʾȱʧ���
library(VIM)
aggr_plot <- aggr(rawdata,col=c('navyblue','red'),
                  numbers=TRUE, 
                  sortVars=TRUE,
                  labels=names(rawdata),
                  cex.axis=.7,gap=3,
                  ylab=c('Missing data','Pattern'))
#���ݲ岹
library(impute)
knn.rawdata<-impute.knn(rawdata,k=10,rowmax = 0.5,colmax=0.8,maxp =3000, rng.seed=362436069)
View(knn.rawdata)
write.table (knn.rawdata$data, file ="4.csv", sep =",", row.names =FALSE)

#����ѵ������֤��
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
#����lasso�ع�
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
#����ģ�ͣ�logistic/xgboost/randomforest/��������
rm(list = ls())
library(tidymodels)
setwd("E:/R work/EV")
EV<-read.table("4set.CSV",header=T, sep=",")
colnames(EV)

# ������������
# ���������ת��Ϊfactor,���Ҳ��
for(i in c(2,3,17,19)){ #,3,14,15,22,23,24,25
  EV[[i]] <- factor(EV[[i]])
}

EV$outcome<- factor(EV$outcome,levels=c(0,1),labels=c("No","Yes")) #���һ��Ҫת����
EV$Ascites <- factor(EV$Ascites,levels=c(1,2,3),labels=c("None","Mild","Moderate to severe"))
EV$CTP <- factor(EV$CTP,levels=c(1,2,3),labels=c("A","B","C"))
#EV$tc <- factor(EV$tc,levels=c(1,2,3,4),labels=c("��2.92","2.92-3.42","3.42-4.04",">4.04"))
#EV$hdlc <- factor(EV$hdlc,levels=c(1,2,3,4),labels=c("��0.56","0.56-0.80","0.80-1.09",">1.09"))
#EV$sd <- factor(EV$sd,levels=c(1,2,3,4),labels=c("��114.28","114.28-131.00","131.00-152.60",">152.60"))
#EV$pc <- factor(EV$pc,levels=c(1,2,3,4),labels=c("��53","54-78","79-113","��114"))

# �����������������ݸſ�
skimr::skim(EV)
#����ѵ��������֤��
dev = EV[EV$category==1,] #dev��vad֮ǰ���Ѿ�������
vad = EV[EV$category==2,]
# ����Ԥ����
# �ȶ���ѵ����д�䷽
datarecipe <- recipe(outcome ~ tc+hdlc+PC+Ascites+SD+CTP, dev) %>%
  #step_rm(Id) %>% �޳�ģ���е��޹ر���
  step_naomit(all_predictors(), skip = F) %>% #�޳�ȱʧֵ
  step_dummy(all_nominal_predictors()) %>% #�����з���������ж��ȱ���
  prep()
datarecipe #��Ϊ�����ʽ�Ѿ�����������6�������������������ֱ����.
#��������ѵ�����Ͳ��Լ�
traindata <- bake(datarecipe, new_data = NULL) %>%
  select(outcome, everything())
testdata <- bake(datarecipe, new_data =vad) %>%
  select(outcome, everything())
# ����Ԥ��������ݸſ�����һ��ת���Ժ�ֻ��outcome��factor����������numeric����
skimr::skim(traindata)
skimr::skim(testdata)

#############################ѵ��logisticģ��
# �趨ģ��
model_logistic <- logistic_reg(
  mode = "classification", #Ĭ���Ƿ���
  engine = "glm"
)
model_logistic

# ���ģ��
fit_logistic <- model_logistic %>%
  fit(outcome~., traindata) #ʹ��.����Ϊ�����Ѿ�������ģ��
fit_logistic
fit_logistic$fit
summary(fit_logistic$fit)

# ϵ�����
fit_logistic %>%
  tidy()

# Ӧ��ģ��-Ԥ��ѵ����
predtrain_logistic <- fit_logistic %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "logistic")
predtrain_logistic #pred_Yes��No�Ǹ����Լ�����
# ����ģ��ROC����-ѵ������
contrasts(traindata$outcome) #���ȸ�������������ο�ˮƽ���ĸ���0���Ե��ǲο�ˮƽĬ����first��1���Ե���second����������0����дfirst��
levels(traindata$outcome)
roctrain_logistic <- predtrain_logistic %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_logistic
#ѵ����roc����
autoplot(roctrain_logistic) 
# Լ�Ƿ����Ӧ��pֵ��������ֵ��
yueden_logistic <- roctrain_logistic %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_logistic
# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtrain_logistic2 <- predtrain_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_logistic, "Yes", "No")))
predtrain_logistic2
# ��������
cmtrain_logistic <- predtrain_logistic2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_logistic

autoplot(cmtrain_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# �ϲ�ָ��
eval_train_logistic <- cmtrain_logistic %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_logistic %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_logistic

###################################
# Ӧ��ģ��-Ԥ����Լ�
predtest_logistic <- fit_logistic %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "logistic")
predtest_logistic

# ����ģ��ROC����-���Լ���
roctest_logistic <- predtest_logistic %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
autoplot(roctest_logistic)

# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtest_logistic2 <- predtest_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_logistic, "Yes", "No")))
predtest_logistic2

# ��������
cmtest_logistic <- predtest_logistic2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_logistic

autoplot(cmtest_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# �ϲ�ָ��
eval_test_logistic <- cmtest_logistic %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_logistic %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_logistic

##############################################################
# �ϲ�ѵ�����Ͳ��Լ���ROC����
roctrain_logistic %>%
  bind_rows(roctest_logistic) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))
  
# �ϲ�ѵ�����Ͳ��Լ�������ָ��
eval_logistic <- eval_train_logistic %>%
  bind_rows(eval_test_logistic) %>%
  mutate(model = "logistic")
eval_logistic

setwd("E:/R work/EV/results/eval")
write.table (eval_logistic, file ="eval_logistic.csv", sep =",", row.names =FALSE)

#################################################################
###������֤������10��ͼ
# ����������С��ʱ������趨n�۽�����֤,n�۽�����֤�Ϳ��Բ���ѵ������֤�������ǰ����ж�������ģ��
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# workflow
wf_logistic <- 
  workflow() %>%
  add_model(model_logistic) %>%
  add_formula(outcome ~ .)
wf_logistic

# ������֤
set.seed(2023)
cv_logistic <- 
  wf_logistic %>%
  fit_resamples(folds,
                metrics = metric_set(accuracy, roc_auc, pr_auc),
                control = control_resamples(save_pred = T))
cv_logistic

# ������ָ֤��ƽ�����
eval_cv_logistic <- collect_metrics(cv_logistic)
eval_cv_logistic

# ������ָ֤���������ÿһ��
eval_cv10_logistic <- collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  mutate(model = "logistic") %>%
  left_join(eval_cv_logistic[c(1,3,5)]) #135������ʲô
eval_cv10_logistic

# �����������
setwd("E:/R work/EV/cls2")
save(fit_logistic,
     predtrain_logistic,
     predtest_logistic,
     eval_logistic,
     eval_cv10_logistic, 
     file = "evalresult_logistic.RData")

# ������ָ֤������ͼͼʾ
eval_cv10_logistic %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# ������֤ROC����ͼʾ
collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(size = 1) +
  theme_classic()

#####################################ѵ��randomforestģ��
# �趨ģ��
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
  add_formula(outcome ~ .)#����Ҫд��������ָ��
wk_rf

# �س����趨-n�۽�����֤��Ѱ����õĳ�������Ҳ�����������ROC��������
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# ������Ѱ�ŷ�Χ
hpset_rf <- parameters(
  mtry(range = c(2, 10)), 
  trees(range = c(200, 500)),
  min_n(range = c(20, 50))
)
hpgrid_rf <- grid_regular(hpset_rf, levels = c(3, 2, 2)) #322��ָmtry,tree,min_n�ֱ���3,2,2����ѡֵ
hpgrid_rf

# ������֤������������
set.seed(2023)
tune_rf <- wk_rf %>%
  tune_grid(resamples = folds,
            grid = hpgrid_rf,
            metrics = metric_set(accuracy, roc_auc, pr_auc), #ǰ�����վ�Ͽɲ�
            control = control_grid(save_pred = T, verbose = T)) #��ӡ�м���Ϣ

# ͼʾ������֤���
autoplot(tune_rf)#ͼ������
eval_tune_rf <- tune_rf %>%
  collect_metrics()
eval_tune_rf #3*2*2=12�������������ԣ�ÿ������5�۽�����֤�����õ�ƽ��ֵ��12*3,3��������

# ����������֤�õ������ų�����
hpbest_rf <- tune_rf %>%
  select_best(metric = "roc_auc")
hpbest_rf

# �������ų��������ѵ������ģ��
set.seed(2023)
final_rf <- wk_rf %>%
  finalize_workflow(hpbest_rf) %>%
  fit(traindata)
final_rf

# ��ȡ���յ��㷨ģ��
final_rf2 <- final_rf %>%
  extract_fit_engine()
plot(final_rf2, main = "���ɭ�����Ŀ���������ݱ�")
legend("top", 
       legend = colnames(final_rf2$err.rate),
       lty = 1:3,
       col = 1:3,
       horiz = T)

# ������Ҫ��
importance(final_rf2)
varImpPlot(final_rf2, main = "������Ҫ��")

# ƫ����ͼ
partialPlot(final_rf2, 
            pred.data = as.data.frame(traindata), 
            x.var = tc, #ָ��һ���Ա���
            which.class = "Yes")

##############################################################

# Ӧ��ģ��-Ԥ��ѵ����
predtrain_rf <- final_rf %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "rf")
predtrain_rf

# ����ģ��ROC����-ѵ������
contrasts(traindata$outcome)
roctrain_rf <- predtrain_rf %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
autoplot(roctrain_rf)

# Լ�Ƿ����Ӧ��pֵ
yueden_rf <- roctrain_rf %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_rf
# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtrain_rf2 <- predtrain_rf %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_rf, "Yes", "No")))
predtrain_rf2
# ��������
cmtrain_rf <- predtrain_rf2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_rf

autoplot(cmtrain_rf, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_train_rf <- cmtrain_rf %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_rf %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_rf

##############################################################

# Ӧ��ģ��-Ԥ����Լ�
predtest_rf <- final_rf %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "rf")
predtest_rf
View(predtest_rf)
# ����ģ��ROC����-���Լ���
roctest_rf <- predtest_rf %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
autoplot(roctest_rf)

# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtest_rf2 <- predtest_rf %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_rf, "Yes", "No")))
predtest_rf2
# ��������
cmtest_rf <- predtest_rf2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_rf

autoplot(cmtest_rf, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_test_rf <- cmtest_rf %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_rf %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_rf

##############################################################

# �ϲ�ѵ�����Ͳ��Լ���ROC����
roctrain_rf %>%
  bind_rows(roctest_rf) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))


# �ϲ�ѵ�����Ͳ��Լ�������ָ��
eval_rf <- eval_train_rf %>%
  bind_rows(eval_test_rf) %>%
  mutate(model = "rf")
eval_rf
setwd("E:/R work/EV/results/eval")
write.table (eval_rf, file ="eval_rf.csv", sep =",", row.names =FALSE)
#############################################################
# ���ų������Ľ�����ָ֤��ƽ���������ȡһЩ�м�����
eval_best_cv_rf <- eval_tune_rf %>%
  inner_join(hpbest_rf[, 1:3])
eval_best_cv_rf

# ���ų������Ľ�����ָ֤�������
eval_best_cv10_rf <- tune_rf %>%
  collect_predictions() %>%
  inner_join(hpbest_rf[, 1:3]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes, event_level = "second") %>% #֮ǰ���д�����event_level
  ungroup() %>%
  mutate(model = "rf") %>%
  inner_join(eval_best_cv_rf[c(4,6,8)])
eval_best_cv10_rf

# �����������
setwd("E:/R work/EV/cls2")
save(final_rf,
     predtrain_rf,
     predtest_rf,
     eval_rf,
     eval_best_cv10_rf, 
     file = "evalresult_rf.RData")

# ���ų������Ľ�����֤����ͼͼʾ
eval_best_cv10_rf %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# ���ų������Ľ�����֤ROCͼʾ

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
# ѵ��ģ��
# �趨ģ��
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
  stop_iter = 25 #��ǰ��ֹ��������
) %>%
  set_args(validation = 0.2)
model_xgboost

# workflow
wk_xgboost <- 
  workflow() %>%
  add_model(model_xgboost) %>%
  add_formula(outcome ~ .)
wk_xgboost
# �س����趨-n�۽�����֤
set.seed(2023)
folds <- vfold_cv(traindata, v =10)
folds

# ������Ѱ�ŷ�Χ
hpset_xgboost <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
)
#hpgrid_xgboost <- 
#grid_regular(hpset_xgboost, levels = c(3, 2, 2, 3, 2, 2)) #ÿ���������ı�ѡֵ��������һ������ʡ�ԣ���Ϊ��һ���Ĵ��ڡ�
set.seed(2023)
hpgrid_xgboost <- grid_random(hpset_xgboost, size = 5) #�������鳬����
hpgrid_xgboost

# ������֤�����������
set.seed(2023)
tune_xgboost <- wk_xgboost %>%
  tune_grid(resamples = folds,
            grid = hpgrid_xgboost,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# ͼʾ������֤���
autoplot(tune_xgboost)
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost
# ����������֤�õ������ų�����
hpbest_xgboost <- tune_xgboost %>%
  select_best(metric = "roc_auc")
hpbest_xgboost

# �������ų��������ѵ������ģ��
set.seed(2023)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(traindata)
final_xgboost

# ��ȡ���յ��㷨ģ��
final_xgboost2 <- final_xgboost %>%
  extract_fit_engine()

importance_matrix <- xgb.importance(model = final_xgboost2)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Cover", #Ҳ����ͨ��gain,frequency
                    col = "skyblue")
# SHAP
colnames(traindata)
xgb.plot.shap(data = as.matrix(traindata[,-1]), #ȥ��traindata�������
              model = final_xgboost2,
              top_n = 20)
###############################################################

# Ӧ��ģ��-Ԥ��ѵ����
predtrain_xgboost <- final_xgboost %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "xgboost")
predtrain_xgboost 
# ����ģ��ROC����-ѵ������
levels(traindata$outcome)
roctrain_xgboost <- predtrain_xgboost %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_xgboost
autoplot(roctrain_xgboost)

# Լ�Ƿ����Ӧ��pֵ
yueden_xgboost <- roctrain_xgboost %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_xgboost
# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtrain_xgboost2 <- predtrain_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_xgboost, "Yes", "No")))
predtrain_xgboost2
# ��������
cmtrain_xgboost <- predtrain_xgboost2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_xgboost

autoplot(cmtrain_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_train_xgboost <- cmtrain_xgboost %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_xgboost %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_xgboost

###############################################################

# Ӧ��ģ��-Ԥ����Լ�
predtest_xgboost <- final_xgboost %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "xgboost")
predtest_xgboost

# ����ģ��ROC����-���Լ���
roctest_xgboost <- predtest_xgboost %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
roctest_xgboost
autoplot(roctest_xgboost)

setwd("E:/R work/EV/results/eval")
write.table (roctest_xgboost, file ="roctest_xgboost.csv", sep =",", row.names =FALSE)

# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtest_xgboost2 <- predtest_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_xgboost, "Yes", "No")))
predtest_xgboost2
# ��������
cmtest_xgboost <- predtest_xgboost2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_xgboost

autoplot(cmtest_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_test_xgboost <- cmtest_xgboost %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_xgboost %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_xgboost

##############################################################
# �ϲ�ѵ�����Ͳ��Լ���ROC����
roctrain_xgboost %>%
  bind_rows(roctest_xgboost) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()

# �ϲ�ѵ�����Ͳ��Լ�������ָ��
eval_xgboost <- eval_train_xgboost %>%
  bind_rows(eval_test_xgboost) %>%
  mutate(model = "xgboost")
eval_xgboost

setwd("E:/R work/EV/results/eval")
write.table (eval_xgboost, file ="eval_xgboost.csv", sep =",", row.names =FALSE)
#############################################################

# ���ų������Ľ�����ָ֤��ƽ�����
eval_best_cv_xgboost <- eval_tune_xgboost %>%
  inner_join(hpbest_xgboost[, 1:6])
eval_best_cv_xgboost

# ���ų������Ľ�����ָ֤�������
eval_best_cv10_xgboost <- tune_xgboost %>%
  collect_predictions() %>%
  inner_join(hpbest_xgboost[, 1:6]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes, event_level="second") %>%
  ungroup() %>%
  mutate(model = "xgboost") %>%
  inner_join(eval_best_cv_xgboost[c(7,9,11)])
eval_best_cv10_xgboost

# �����������
setwd("E:/R work/EV/cls2")
save(final_xgboost,
     predtrain_xgboost,
     predtest_xgboost,
     eval_xgboost,
     eval_best_cv10_xgboost, 
     file = "evalresult_xgboost.RData")

# ���ų������Ľ�����ָ֤��ͼʾ
eval_best_cv10_xgboost %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# ���ų������Ľ�����֤ͼʾ
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
####################������ģ��
# ѵ��ģ��
# �趨ģ��
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
  add_formula(outcome~ .) #��Ϊǰ���Ѿ����ݽ�����
wk_dt

# �س����趨-n�۽�����֤
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# ������Ѱ�ŷ�Χ
hpset_dt <- parameters(tree_depth(range = c(3, 7)),
                       min_n(range = c(5, 10)),
                       cost_complexity(range = c(-6, -1))) 
# hpgrid_dt <- grid_regular(hpset_dt, levels = c(3, 2, 4)) #method1��������ÿ����������ȡֵ�������������3*2*4��ģ��
set.seed(2023)
hpgrid_dt <- grid_random(hpset_dt, size = 10)#method2��������10�鳬����
hpgrid_dt #�����ڣ�-6��-1����Χ��
log10(hpgrid_dt$cost_complexity) #����ת�����ڸ÷�Χ��


# ������֤������������
set.seed(2023)
tune_dt <- wk_dt %>%
  tune_grid(resamples = folds,
            grid = hpgrid_dt,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# ͼʾ������֤���
autoplot(tune_dt)
eval_tune_dt <- tune_dt %>%
  collect_metrics()
eval_tune_dt

# ����������֤�õ������ų�����
hpbest_dt <- tune_dt %>%
  select_by_one_std_err(metric = "roc_auc", desc(cost_complexity))#��ָ��������һ����ע������ʹ��ģ����򵥵ĵ��Ǹ������������Ұ��ո��ӶȽ������У����Ӷ�Խ��ģ��Խ�򵥡�
hpbest_dt

# �������ų��������ѵ������ģ��
final_dt <- wk_dt %>%
  finalize_workflow(hpbest_dt) %>%
  fit(traindata)
final_dt

# ��ȡ���յ��㷨ģ��
final_dt2 <- final_dt %>%
  extract_fit_engine()
library(rpart.plot)
rpart.plot(final_dt2) #����������ͼ

final_dt2$variable.importance
par(mar = c(10, 3, 1, 1))
barplot(final_dt2$variable.importance, las = 1) #las=2ʹ������ı�ǩ��ֱ

##############################################################

# Ӧ��ģ��-Ԥ��ѵ����
predtrain_dt <- final_dt %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "dt")
predtrain_dt #pred.no��pred.yesҪ�����Լ������ݸġ�

# ����ģ��ROC����-ѵ������
contrasts(traindata$outcome)#������Ϊ0��Ӧ���ǻ�׼ˮƽ��Ϊfirst��
roctrain_dt <- predtrain_dt %>%
  # roc_curve(AHD, .pred_Yes, event_level = "second") %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_dt
autoplot(roctrain_dt)

# Լ�Ƿ����Ӧ��pֵ
yueden_dt <- roctrain_dt %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_dt
# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtrain_dt2 <- predtrain_dt %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_dt, "Yes", "No")))
predtrain_dt2

# ��������
cmtrain_dt <- predtrain_dt2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_dt

autoplot(cmtrain_dt, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# �ϲ�ָ��
eval_train_dt <- cmtrain_dt %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_dt %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_dt

##############################################################

# Ӧ��ģ��-Ԥ����Լ�
predtest_dt <- final_dt %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "dt")
predtest_dt
# ����ģ��ROC����-���Լ���
roctest_dt <- predtest_dt %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
roctest_dt
autoplot(roctest_dt)

# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtest_dt2 <- predtest_dt %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_dt, "Yes", "No")))
predtest_dt2

# ��������
cmtest_dt <- predtest_dt2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_dt

autoplot(cmtest_dt, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_test_dt <- cmtest_dt %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_dt %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_dt

##############################################################
# �ϲ�ѵ�����Ͳ��Լ���ROC����
roctrain_dt %>%
  bind_rows(roctest_dt) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()

# �ϲ�ѵ�����Ͳ��Լ�������ָ��
eval_dt <- eval_train_dt %>%
  bind_rows(eval_test_dt) %>%
  mutate(model = "dt")
eval_dt
View(eval_dt)

setwd("E:/R work/EV/results/eval")
write.table (eval_dt, file ="eval_dt.csv", sep =",", row.names =FALSE)
#############################################################

# ���ų������Ľ�����ָ֤��ƽ�����
eval_best_cv_dt <- eval_tune_dt %>%
  inner_join(hpbest_dt[, 1:3])
eval_best_cv_dt

# ���ų������Ľ�����ָ֤�������
eval_best_cv10_dt <- tune_dt %>%
  collect_predictions() %>%
  inner_join(hpbest_dt[, 1:3]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes, event_level="second") %>%
  ungroup() %>%
  mutate(model = "dt") %>%
  inner_join(eval_best_cv_dt[c(4,6,8)]) #ĳһ��
eval_best_cv10_dt

# �����������
setwd("E:/R work/EV/cls2")
save(final_dt,
     predtrain_dt,
     predtest_dt,
     eval_dt,
     eval_best_cv10_dt, 
     file = "evalresult_dt.RData")

# ���ų������Ľ�����ָ֤��ͼʾ
eval_best_cv10_dt %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# ���ų������Ľ�����֤ͼʾ
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
# ѵ��ģ��
# �趨ģ��
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

# �س����趨-10�۽�����֤
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# ������Ѱ�ŷ�Χ
hpset_rsvm <- parameters(cost(range = c(-5, 5)), 
                         rbf_sigma(range = c(-4, -1)))
hpgrid_rsvm <- grid_regular(hpset_rsvm, levels = c(2,3))
hpgrid_rsvm
# ������֤������������
library(kernlab)
set.seed(2023)
tune_rsvm <- wk_rsvm %>%
  tune_grid(resamples = folds,
            grid = hpgrid_rsvm,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# ͼʾ������֤���
autoplot(tune_rsvm)
eval_tune_rsvm <- tune_rsvm %>%
  collect_metrics()
eval_tune_rsvm

# ����������֤�õ������ų�����
hpbest_rsvm <- tune_rsvm %>%
  select_best(metric = "roc_auc")
hpbest_rsvm

# �������ų��������ѵ������ģ��
final_rsvm <- wk_rsvm %>%
  finalize_workflow(hpbest_rsvm) %>%
  fit(traindata)
final_rsvm

# ��ȡ���յ��㷨ģ��
final_rsvm %>%
  extract_fit_engine()

###############################################################

# Ӧ��ģ��-Ԥ��ѵ����
predtrain_rsvm <- final_rsvm %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "rsvm")
predtrain_rsvm
# ����ģ��ROC����-ѵ������
levels(traindata$outcome)
roctrain_rsvm <- predtrain_rsvm %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_rsvm
autoplot(roctrain_rsvm)

# Լ�Ƿ����Ӧ��pֵ
yueden_rsvm <- roctrain_rsvm %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_rsvm

# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtrain_rsvm2 <- predtrain_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_rsvm, "Yes", "No")))
predtrain_rsvm2
# ��������
cmtrain_rsvm <- predtrain_rsvm2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)

cmtrain_rsvm

autoplot(cmtrain_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_train_rsvm <- cmtrain_rsvm %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_rsvm %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_rsvm

##############################################################
# Ӧ��ģ��-Ԥ����Լ�
predtest_rsvm <- final_rsvm %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "rsvm")
predtest_rsvm
# ����ģ��ROC����-���Լ���
roctest_rsvm <- predtest_rsvm %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
roctest_rsvm
autoplot(roctest_rsvm)


# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtest_rsvm2 <- predtest_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_rsvm, "Yes", "No")))
predtest_rsvm2
# ��������
cmtest_rsvm <- predtest_rsvm2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_rsvm

autoplot(cmtest_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_test_rsvm <- cmtest_rsvm %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_rsvm %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_rsvm
##############################################################

# �ϲ�ѵ�����Ͳ��Լ���ROC����
roctrain_rsvm %>%
  bind_rows(roctest_rsvm) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))

# �ϲ�ѵ�����Ͳ��Լ�������ָ��
eval_rsvm <- eval_train_rsvm %>%
  bind_rows(eval_test_rsvm) %>%
  mutate(model = "rsvm")
eval_rsvm

setwd("E:/R work/EV/results/eval")
write.table (eval_rsvm, file ="eval_rsvm.csv", sep =",", row.names =FALSE)

View(eval_rsvm)
#############################################################

# ���ų������Ľ�����ָ֤��ƽ�����
eval_best_cv_rsvm <- eval_tune_rsvm %>%
  inner_join(hpbest_rsvm[, 1:2])
eval_best_cv_rsvm

# ���ų������Ľ�����ָ֤�������
eval_best_cv10_rsvm <- tune_rsvm %>%
  collect_predictions() %>%
  inner_join(hpbest_rsvm[, 1:2]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes,event_level="second") %>%
  ungroup() %>%
  mutate(model = "rsvm") %>%
  inner_join(eval_best_cv_rsvm[c(3,5,7)])
eval_best_cv10_rsvm

# �����������
setwd("E:/R work/EV/cls2")
save(final_rsvm,
     predtrain_rsvm,
     predtest_rsvm,
     eval_rsvm,
     eval_best_cv10_rsvm, 
     file = "evalresult_rsvm.RData")
# ���ų������Ľ�����ָ֤��ͼʾ
eval_best_cv10_rsvm %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# ���ų������Ľ�����֤ͼʾ
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
# ѵ��ģ��
# �趨ģ��
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

# �س����趨-10�۽�����֤
set.seed(2023)
folds <- vfold_cv(traindata, v = 10)
folds

# ������Ѱ�ŷ�Χ
hpset_mlp <- parameters(hidden_units(range = c(15, 24)),
                        penalty(range = c(-3, 0)),
                        epochs(range = c(50, 150)))
hpgrid_mlp <- grid_regular(hpset_mlp, levels = 2)
hpgrid_mlp

# ������֤������������
set.seed(2023)
tune_mlp <- wk_mlp %>%
  tune_grid(resamples = folds,
            grid = hpgrid_mlp,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# ͼʾ������֤���
autoplot(tune_mlp)
eval_tune_mlp <- tune_mlp %>%
  collect_metrics()
eval_tune_mlp

# ����������֤�õ������ų�����
hpbest_mlp <- tune_mlp %>%
  select_best(metric = "roc_auc")
hpbest_mlp

# �������ų��������ѵ������ģ��
set.seed(2023)
final_mlp <- wk_mlp %>%
  finalize_workflow(hpbest_mlp) %>%
  fit(traindata)
final_mlp

# ��ȡ���յ��㷨ģ��
final_mlp2 <- final_mlp %>%
  extract_fit_engine()

library(NeuralNetTools)
plotnet(final_mlp2)
garson(final_mlp2) +
  coord_flip()
olden(final_mlp2) +
  coord_flip()

#############################################################

# Ӧ��ģ��-Ԥ��ѵ����
predtrain_mlp <- final_mlp %>%
  predict(new_data = traindata, type = "prob") %>%
  bind_cols(traindata %>% select(outcome)) %>%
  mutate(dataset = "train")%>%
  mutate(model = "mlp")
predtrain_mlp
# ����ģ��ROC����-ѵ������
levels(traindata$outcome)
roctrain_mlp <- predtrain_mlp %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "train")
roctrain_mlp
autoplot(roctrain_mlp)

# Լ�Ƿ����Ӧ��pֵ
yueden_mlp <- roctrain_mlp %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_mlp
# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtrain_mlp2 <- predtrain_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_mlp, "Yes", "No")))
predtrain_mlp2
# ��������
cmtrain_mlp <- predtrain_mlp2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtrain_mlp

autoplot(cmtrain_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_train_mlp <- cmtrain_mlp %>%
  summary(event_level = "second") %>%
  bind_rows(predtrain_mlp %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "train")
eval_train_mlp

#############################################################

# Ӧ��ģ��-Ԥ����Լ�
predtest_mlp <- final_mlp %>%
  predict(new_data = testdata, type = "prob") %>%
  bind_cols(testdata %>% select(outcome)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "mlp")
predtest_mlp
# ����ģ��ROC����-���Լ���
roctest_mlp <- predtest_mlp %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "test")
roctest_mlp
autoplot(roctest_mlp)


# Ԥ�����+Լ�Ƿ���=Ԥ�����
predtest_mlp2 <- predtest_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_Yes >= yueden_mlp, "Yes", "No")))
predtest_mlp2
# ��������
cmtest_mlp <- predtest_mlp2 %>%
  conf_mat(truth = outcome, estimate = .pred_class)
cmtest_mlp

autoplot(cmtest_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# �ϲ�ָ��
eval_test_mlp <- cmtest_mlp %>%
  summary(event_level = "second") %>%
  bind_rows(predtest_mlp %>%
              roc_auc(outcome, .pred_Yes, event_level = "second")) %>%
  mutate(dataset = "test")
eval_test_mlp

#############################################################

# �ϲ�ѵ�����Ͳ��Լ���ROC����
roctrain_mlp %>%
  bind_rows(roctest_mlp) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))

# �ϲ�ѵ�����Ͳ��Լ�������ָ��
eval_mlp <- eval_train_mlp %>%
  bind_rows(eval_test_mlp) %>%
  mutate(model = "mlp")
eval_mlp

setwd("E:/R work/EV/results/eval")
write.table (eval_mlp, file ="eval_mlp.csv", sep =",", row.names =FALSE)
#############################################################

# ���ų������Ľ�����ָ֤��ƽ�����
eval_best_cv_mlp <- eval_tune_mlp %>%
  inner_join(hpbest_mlp[, 1:3])
eval_best_cv_mlp

# ���ų������Ľ�����ָ֤�������
eval_best_cv10_mlp <- tune_mlp %>%
  collect_predictions() %>%
  inner_join(hpbest_mlp[, 1:3]) %>%
  group_by(id) %>%
  roc_auc(outcome, .pred_Yes,event_level="second") %>%
  ungroup() %>%
  mutate(model = "mlp") %>%
  inner_join(eval_best_cv_mlp[c(4,6,8)])
eval_best_cv10_mlp

# �����������
setwd("E:/R work/EV/cls2")
save(final_mlp,
     predtrain_mlp,
     predtest_mlp,
     eval_mlp,
     eval_best_cv10_mlp, 
     file = "evalresult_mlp.RData")

# ���ų������Ľ�����ָ֤��ͼʾ
eval_best_cv10_mlp %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# ���ų������Ľ�����֤ͼʾ
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
        legend.position = c(0.9,0.3), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))

############################ģ�ͱȽ�
# ���ظ���ģ�͵��������
library(tidymodels)
setwd("E:/R work/EV")
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

#############################################################
# ����ģ���ڲ��Լ��ϵ����ָ��
eval <- bind_rows(
  eval_logistic, 
  eval_rf,
  eval_dt,
  eval_xgboost,
  eval_rsvm,
  eval_mlp)
View(eval)

# ƽ����ͼ���޸Ĳ������Կ�����ĺϲ����롣
eval %>%
  filter(dataset == "test") %>%
  ggplot(aes(x = .metric, y = .estimate, color = model)) + #,shape=model,linetype=model
  geom_point(size=2.5) + #origin��geom_point() +
  geom_line(size=1.2,aes(group = model)) + #origin:geom_line(aes(group = model)) +
  scale_y_continuous(limits = c(0.20, 1))+ 
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(angle = 30,size=12,hjust = 1), #�������ǩ�����С
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c("#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d")) 
#��������ɫ��Ӧ�������µ�ģ�ͣ�DT/LR/MLP/RF/SVM/XGB���������滭DCA����ʱ����ɫ��#e8490f����ɫ��#ffd401��ע���DCA���߱���һ�£���Ȼ��ɫ���벻һ�������ǽ����һ���ģ�
#���˺�ɫ��������ɫ��DCA��������ȫһ���ģ���ɫ�ڸĵ�ʱ������������������Ӧ��ҲҪ�ģ����Ѻ�ɫͬʱҲ���ˣ�������һ�¡�
# ����ģ���ڲ��Լ��ϵ����ָ����
eval2 <- eval %>%
  select(-.estimator) %>%
  filter(dataset == "test") %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval2
# ����ģ���ڲ��Լ��ϵ�rocָ������ͼ
eval2 %>%
  ggplot(aes(x = model, y = roc_auc,fill=model)) + #roc_aucҲ���Ըĳ���������
  geom_col(width = 0.3, show.legend = F) +
  geom_text(aes(label = round(roc_auc, 2)),  #roc_aucҲ���Ըĳ���������
            nudge_y = -0.03) +
 theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12,hjust = 1), #�������ǩ�����С
        axis.text.y=element_text(size=12))+
  scale_fill_manual(values=c("#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))

#############################################################

# ����ģ���ڲ��Լ��ϵ�Ԥ�����
predtest <- bind_rows(
  predtest_logistic,
  predtest_rf,
  predtest_dt,
  predtest_xgboost,
  predtest_rsvm,
  predtest_mlp
)
predtest

# ����ģ���ڲ��Լ��ϵ�ROC
predtest %>%
  group_by(model) %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = model)) +
  geom_path(size = 1) +
  theme_bw()+ #classic
  theme(legend.title = element_blank(),
        legend.position = c(0.8,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))

#����ģ�Ͳ��Լ�rocdelong����
library(pROC)
roc1 <- roc(predtest_logistic$outcome, predtest_logistic$.pred_Yes,levels=c("No", "Yes"))
roc2 <- roc(predtest_rf$outcome, predtest_rf$.pred_Yes,levels=c("No", "Yes"))
roc3 <- roc(predtest_xgboost$outcome, predtest_xgboost$.pred_Yes,levels=c("No", "Yes"))
roc4<- roc(predtest_dt$outcome, predtest_dt$.pred_Yes,levels=c("No", "Yes"))
roc5<- roc(predtest_rsvm$outcome, predtest_rsvm$.pred_Yes,levels=c("No", "Yes"))
roc6<- roc(predtest_mlp$outcome, predtest_mlp$.pred_Yes,levels=c("No", "Yes"))

#delong���飨�������໥��֤��
roc.test(roc3,roc1,method = 'delong')     
roc.test(roc3,roc2,method = 'delong') 
roc.test(roc3,roc6,method = 'delong')
roc.test(roc3,roc5,method = 'delong') 
roc.test(roc3,roc4,method = 'delong') 

#xgboost��������
proc<-roc(predtest_xgboost$outcome,predtest_xgboost$.pred_Yes,levels=c("No", "Yes"),ci=T);A=proc$ci;A 
# ����ģ�ͽ�����֤�ĸ���ָ�����ͼ
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
        legend.position = c(0.7,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))

# ����ģ�ͽ�����֤��ָ��ƽ��ֵͼ(��������)
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
        legend.position = c(0.9,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))

#����ģ����֤��rocdelong����
library(pROC)
roc1 <- roc(predtrain_logistic$outcome, predtrain_logistic$.pred_Yes,levels=c("No", "Yes"))
roc2 <- roc(predtrain_rf$outcome, predtrain_rf$.pred_Yes,levels=c("No", "Yes"))
roc3 <- roc(predtrain_xgboost$outcome, predtrain_xgboost$.pred_Yes,levels=c("No", "Yes"))
roc4<- roc(predtrain_dt$outcome, predtrain_dt$.pred_Yes,levels=c("No", "Yes"))
roc5<- roc(predtrain_rsvm$outcome, predtrain_rsvm$.pred_Yes,levels=c("No", "Yes"))
roc6<- roc(predtrain_mlp$outcome, predtrain_mlp$.pred_Yes,levels=c("No", "Yes"))
#delong���飨�������໥��֤��
roc.test(roc2,roc1,method = 'delong')     
roc.test(roc2,roc3,method = 'delong') 
roc.test(roc2,roc4,method = 'delong')
roc.test(roc2,roc5,method = 'delong') 
roc.test(roc2,roc6,method = 'delong') 

# ����ģ������֤���ϵ�Ԥ�����
predtrain <- bind_rows(
  predtrain_logistic,
  predtrain_rf,
  predtrain_dt,
  predtrain_xgboost,
  predtrain_rsvm,
  predtrain_mlp
)
predtrain

# ����ģ������֤���ϵ�ROC
predtrain %>%
  dplyr::group_by(model) %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = model)) +
  geom_path(size = 1) +
  theme_bw()+ #classic
  theme(legend.title = element_blank(),
        legend.position = c(0.8,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))



#######################################������ģ�͵�calibrationͼ
# ����ģ���ڲ��Լ��ϵ�Ԥ�����2��ֻ��ҪԤ��Ϊ1��0�ĸ��ʣ������Ҫȥ��һ�С�
predtest2 <- predtest %>%
  dplyr::select(-.pred_No) %>% #ȥ��һ������
  mutate(id = rep(1:nrow(predtest_logistic), 6)) %>% #�м���ģ�;�д��
  pivot_wider(names_from = model, values_from = .pred_Yes) #����һ������
View(predtest2)
##�������ǰһ�ε�ԭʼ��
# ����ģ���ڲ��Լ��ϵ�Ԥ�����2
#predtest2 <- predtest %>%
#  select(-.pred_No) %>%
 ## mutate(id = rep(1:nrow(predtest_logistic), 7)) %>%
 # pivot_wider(names_from = model, values_from = .pred_Yes)
#predtest2
# ����ģ���ڲ��Լ��ϵ�У׼����
# http://topepo.github.io/caret/measuring-performance.html#calibration-curves
cal_obj <- caret::calibration(
  as.formula(paste0("outcome ~ ", 
                    paste(colnames(predtest2)[4:9], collapse = " + "))), #4:9��ָģ��
  data = predtest2,
  class = "Yes",
  cuts = 11
)

cal_obj$data

plot(cal_obj, type = "b", pch = 20,
     auto.key = list(columns = 3,
                     lines = T,
                     points = T))

#calibration�ֳ�����ͼ
cal_obj$data %>%
  filter(Percent != 0) %>%
  ggplot(aes(x = midpoint, y = Percent,
             color=model)) +
  geom_point(color = "brown1") +
  geom_line(color = "brown1") +
  geom_abline(slope = 1, intercept = 0) +
  facet_wrap(~calibModelVar) +
  theme_bw()

########calibration��һ�ַ�ʽ
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

#����ͼ��
calall %>%
  ggplot(aes(x = meanpred, y = meanobs, color=model)) +
  geom_point(color = "brown1") +
  geom_line(color = "brown1") +
  geom_abline(slope = 1, intercept = 0) +
  facet_wrap(~model) +
  theme_bw()
#����ͼ��
ggplot(calall,aes(x = meanpred,y = meanobs,color=model))+
  geom_point(size=2.5)+
  geom_line(linetype = 1,size=1.2)+
  geom_abline(slope = 1,intercept = 0,lty="solid",color = "black")+
  labs(x="predicted_possibility",y = "actual_possibility",title = "calibration_curve")+
  theme_classic()+
  theme(legend.title = element_blank(),
        legend.position = c(0.9,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c( "#ef1828","#9ec417","#ffbc14","#00bdcd","#006b7b","#942d8d"))


#######################################################################

# ����ģ���ڲ��Լ��ϵ�DCA
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
        legend.position = c(0.85,0.80), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))+
  scale_color_manual(values = c("#BDBDBD","#000000","#9ec417","#44c1f0","#e8490f","#9b3a74","#3f60aa","#ffd401"))

#��Ӧ"logistic" "rf" "dt" "xgboost" "rsvm""mlp"��c("#9ec417","#44c1f0","#e8490f","#9b3a74","#3f60aa","#ffd401"))
#��������������
####################################################3
# ģ�ͻ���---R����tidymodels������ѧϰ������ع�ģ��---������---ģ�ͽ���
# ѵ���õ�ģ�Ͷ��󣬽��������������ѵ��ģ�ͣ�����������ġ�
# load(file.choose())
# logistic�ع�
object_model <- fit_logistic
# LASSO-��ع�-��������
object_model <- final_enet
# ������
object_model <- final_dt
# ���ɭ��
object_model <- final_rf
# xgboost
object_model <- final_xgboost
# ֧��������
object_model <- final_rsvm
# �����ز�������
object_model <- final_mlp

# �Ա������ݼ�
colnames(traindata)
traindatax <- traindata[,-1] #ȥ��������ڵ���һ��
colnames(traindatax)

# iml��
library(iml)
predictor_model <- Predictor$new(
  object_model, 
  data = traindatax,
  y = traindata$outcome,
  predict.function = function(model, newdata){
    predict(model, newdata, type = "prob") %>%
      pull(2) #��Ϊoutcome��0,1���У���ȡ1��һ�У�Ҳ���ǵڶ���pull(2)
  }
)

# ������Ҫ��-�����û�(�����û�����������������ģ�͵�AUC�����õ�����Ҫ��)����֮ǰ�Դ�randomforest��������Ҫ�Բ�ͬ��
imp_model <- FeatureImp$new(
  predictor_model, 
  loss = function(actual, predicted){
    return(1-Metrics::auc(as.numeric(actual=="Yes"), predicted)) #ʵ��Ϊ1��predictΪ1��
  }
)

# ��ֵ
imp_model$results
# ͼʾ
imp_model$plot() +
  theme_bw()


# ������ЧӦ
 #predictor_model <- Predictor$new( #������һ����������0,1ƫ����ͼ��
  # object_model, 
   #data = traindatax,
   #y = traindata$outcome,
   #type="prob"
 #)
pdp_model <- FeatureEffect$new( #��iml�����Կ�����
  predictor_model, 
  feature = "SD",
  method = "pdp"
)
# ��ֵ
pdp_model$results
# ͼʾ
pdp_model$plot() +
  theme_bw()

# ���б�����ЧӦȫ�����
effs_model <- FeatureEffects$new(predictor_model, method = "pdp")
# ��ֵ
effs_model$results
# ͼʾ
effs_model$plot()

# ������shap����
shap_model <- Shapley$new(
  predictor_model, 
  x.interest = traindatax[1,] #ʹ��traidatax�ĵ�һ������
)
# ��ֵ
shap_model$results
# ͼʾ
shap_model$plot() +
  theme_bw()

# ��������������shap����
# fastshap��
library(fastshap)#����shap���Կ���
shap <- explain(
  object_model, 
  X = as.data.frame(traindatax),
  nsim = 10,
  adjust = T,
  pred_wrapper = function(model, newdata) {
    predict(model, newdata, type = "prob") %>% pull(2)
  }
)

# ������ͼʾ,��testset��ı�������Ҫ���Ա������ݼ��Լ������shap�ĳɲ��Լ��ġ�
force_plot(object = shap[512L, ], #�ڼ�������
           feature_values = as.data.frame(traindatax)[512L, ], #�ڼ�������
            #baseline = mean(predtrain_model$.pred_Yes), # ���ɾ���ģ��Ԥ����
           display = "viewer") 
View(traindata)
# ������Ҫ��
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

# ���б���shapͼʾ
library(ggbeeswarm)
library(ggridges)
OO<-data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  dplyr::group_by(feature) %>%
  dplyr::mutate(value = (value - min(value)) / (max(value) - min(value)), #�߶����ţ����ŵ�0��1֮��
         feature = factor(feature, levels = levels(shapimp$name)))
#��value�Ƿ����ŵ�0��1֮��
OO%>% 
    ggplot(aes(x = value, y = feature)) + #x=shapҲ����
  geom_boxplot()
#��value�Ƿ����ŵ�0��1֮��
OO%>% 
  ggplot(aes(x = value, y = feature,fill=feature)) +#x=shapҲ����
  ggridges::geom_density_ridges()+
  ggridges::theme_ridges()

OO%>% 
  ggplot(aes(x = shap, y = feature,color=value))+
  # geom_jitter(size = 2, height = 0.2, width = 0) +��ʹɢ��ͼ�������
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
  theme(axis.text.x =element_text(size=12,hjust = 1), #�������ǩ�����С
                   axis.text.y=element_text(size=12))
  

# ������shapͼʾ,������������ã�
data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  filter(feature == "hdlc") %>%
  ggplot(aes(x = value, y = shap)) +
  geom_point(color = "#f46f20") + #�ĵ����ɫ
  geom_smooth(se = T, span = 0.5,colour='#13a983') + # '��'���Ը���ɫ
  labs(x = "hdlc")+
  theme_classic()+
  scale_fill_manual(values = c("#00AFBB"))+
  theme(axis.text.x =element_text(size=12,hjust = 1), #�������ǩ�����С
        axis.text.y=element_text(size=12))

###############################����������xgboostģ�����
# ����Ԥ����
# �ȶ���ѵ����д�䷽
datarecipetra <- recipe(outcome ~ NLR+PSDR+FIB4+APRI+AAR+tc+hdlc, dev) %>%
  #step_rm(Id) %>% �޳�ģ���е��޹ر���
  step_naomit(all_predictors(), skip = F) %>% #�޳�ȱʧֵ
  step_dummy(all_nominal_predictors()) %>% #�����з���������ж��ȱ���
  prep()
datarecipetra  #��Ϊ�����ʽ�Ѿ�����������6�������������������ֱ����.
#��������ѵ�����Ͳ��Լ�
traindatatra  <- bake(datarecipetra , new_data = NULL) %>%
  select(outcome, everything())
testdatatra <- bake(datarecipetra , new_data =vad) %>%
  select(outcome, everything())
# ����Ԥ��������ݸſ�����һ��ת���Ժ�ֻ��outcome��factor����������numeric����
skimr::skim(traindatatra)
skimr::skim(testdatatra)
#################xgboost
# ѵ��ģ��
# �趨ģ��
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
  stop_iter = 25 #��ǰ��ֹ��������
) %>%
  set_args(validation = 0.2)
model_xgboosttra

# workflow
wk_xgboosttra <- 
  workflow() %>%
  add_model(model_xgboosttra) %>%
  add_formula(outcome ~ .)
wk_xgboosttra
# �س����趨-n�۽�����֤
set.seed(2023)
foldstra <- vfold_cv(traindatatra, v =10)
foldstra

# ������Ѱ�ŷ�Χ
hpset_xgboosttra <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
)

#hpgrid_xgboost <- 
#grid_regular(hpset_xgboost, levels = c(3, 2, 2, 3, 2, 2)) #ÿ���������ı�ѡֵ��������һ������ʡ�ԣ���Ϊ��һ���Ĵ��ڡ�
set.seed(2023)
hpgrid_xgboosttra <- grid_random(hpset_xgboosttra, size = 5) #�������鳬����
hpgrid_xgboosttra

# ������֤�����������
set.seed(2023)
tune_xgboosttra <- wk_xgboosttra %>%
  tune_grid(resamples = foldstra,
            grid = hpgrid_xgboosttra,
            metrics = metric_set(accuracy, roc_auc, pr_auc),
            control = control_grid(save_pred = T, verbose = T))

# ͼʾ������֤���
autoplot(tune_xgboosttra)
eval_tune_xgboosttra <- tune_xgboosttra %>%
  collect_metrics()
eval_tune_xgboosttra

# ����������֤�õ������ų�����
hpbest_xgboosttra <- tune_xgboosttra %>%
  select_best(metric = "roc_auc")
hpbest_xgboosttra

# �������ų��������ѵ������ģ��
set.seed(2023)
final_xgboosttra <- wk_xgboosttra %>%
  finalize_workflow(hpbest_xgboosttra) %>%
  fit(traindatatra)
final_xgboosttra

###############################################################
# Ӧ��ģ��-Ԥ��ѵ����
predtrain_xgboosttra <- final_xgboosttra %>%
  predict(new_data = traindatatra, type = "prob") %>%
  bind_cols(traindatatra %>% select(outcome)) %>%
  mutate(dataset = "traintra")
predtrain_xgboosttra 
# ����ģ��ROC����-ѵ������
levels(traindatatra$outcome)
roctrain_xgboosttra <- predtrain_xgboosttra%>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "traintra")
roctrain_xgboosttra
autoplot(roctrain_xgboosttra)

#�����ǹ�����ģ�ͱȽ�ROC����֮ѵ����
roctrain_xgboost %>%
  bind_rows(roctrain_xgboosttra) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.8,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))

#����ģ����֤��rocdelong����,���Կ�AUC
library(pROC)
roc1 <- roc(predtrain_xgboost$outcome, predtrain_xgboost$.pred_Yes,levels=c("No", "Yes"))
roc2 <- roc(predtrain_xgboosttra$outcome, predtrain_xgboosttra$.pred_Yes,levels=c("No", "Yes"))
#delong���飨�������໥��֤��
roc.test(roc2,roc1,method = 'delong')   

###############################################################
# Ӧ��ģ��-Ԥ����Լ�
predtest_xgboosttra <- final_xgboosttra %>%
  predict(new_data = testdatatra, type = "prob") %>%
  bind_cols(testdatatra %>% select(outcome)) %>%
  mutate(dataset = "testtra") %>%
  mutate(model = "xgboost")
predtest_xgboosttra
# ����ģ��ROC����-���Լ���
roctest_xgboosttra <- predtest_xgboosttra %>%
  roc_curve(outcome, .pred_Yes, event_level = "second") %>%
  mutate(dataset = "testtra")
roctest_xgboosttra
autoplot(roctest_xgboosttra)

#�����ǹ�����ģ�ͱȽ�ROC����֮���Լ�
roctest_xgboost %>%
  bind_rows(roctest_xgboosttra) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()+
  theme(legend.title = element_blank(),
        legend.position = c(0.8,0.2), #ͼ��λ��
        legend.text= element_text(size=14),#ͼ����С
        axis.text.x =element_text(size=12), #�������ǩ�����С
        axis.text.y=element_text(size=12))
#����ģ����֤��rocdelong����,���Կ�AUC
library(pROC)
roc1 <- roc(predtest_xgboost$outcome, predtest_xgboost$.pred_Yes,levels=c("No", "Yes"))
roc2 <- roc(predtest_xgboosttra$outcome, predtest_xgboosttra$.pred_Yes,levels=c("No", "Yes"))
#delong���飨�������໥��֤��
roc.test(roc2,roc1,method = 'delong')   
