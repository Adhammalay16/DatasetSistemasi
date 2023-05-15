## Package
library(tidyverse)
library(psych)
library(ggplot2)

## Input Data
library(readxl)
data.uas <- read_excel("~/Dataset Klasifikasi.xlsx", sheet = "Dataset Imbalance")

# melihat dataset di baris atas
head(data.uas)

# Melihat karakteristik setiap variabel
str(data.uas)
summary(data.uas)

## Preprocessing Data
#------------------------- Mengecek Missing Value
library(visdat)
vis_miss(data.uas)

#------------------------- Kategorisasi IPM
data.uas$Class <- cut(data.uas$IPM, 
                      breaks = c(0,60,69.9,79.9,Inf),
                      labels = c("Rendah","Menengah","Tinggi","Sangat.Tinggi"),
                      right = FALSE)
data.uas2 <- data.uas[,-c(1,8)]
head(data.uas2)

nclass <- summary(data.uas$Class)
barplot(nclass, xlab = "IPM Class", ylab = "jumlah data", col = rainbow(10))

#------------------------- Box-plot
library(caret)
featurePlot(x = data.uas2[,-7], 
            y = data.uas2$Class, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(3,2), ## ukuran grafik 2x3
            auto.key = list(columns = 2))

#------------------------- Plot Imbalance Data
library(ggplot2)
table(data.uas2$Class)
plotImbalance <- function(loadedData){
  ggplot(loadedData, aes(x = Class, fill = Class)) +
    geom_bar(aes(fill = 'blue')) +
    scale_fill_manual(values = c('steelblue', 'tomato','red','blue')) +
    theme(legend.position="none")
}
plotImbalance(data.uas2)

#-------------------------- SMOTE
library(tidymodels)
library(themis)

rec <- recipe(Class~., data = data.uas2) %>% 
  step_smote(Class,over_ratio = 1,neighbors = 5, seed = 1) %>%
  prep()

new_data <- rec %>% bake(new_data = NULL)
table(new_data$Class)
table(data.uas2$Class)

plotImbalance(new_data)

### Simpan Data Balance
writexl::write_xlsx(new_data, "~/Dataset Klasifikasi.xlsx", sheet = "Dataset Balance (Hasil SMOTE)")

### Input data balanced
df_balance <- read_excel("~/Dataset Klasifikasi.xlsx", sheet = "Dataset Balance (Hasil SMOTE)")
df_balance

#------------------------- Membagi Data Testing dan Training
library(caret)

## Imbalance data
set.seed(123)
idx_im <- createDataPartition(data.uas2$Class, p=0.8, list = FALSE, times = 1)
train_im <- data.uas2[idx_im, ]
test_im <- data.uas2[-idx_im, ]

## Balance data
set.seed(123)
idx_bl <- createDataPartition(df_balance$Class, p=0.8, list = FALSE, times = 1)
train_bl <- df_balance[idx_bl, ]
test_bl <- df_balance[-idx_bl, ]

print("Jumlah observasi data training")
table(train_im$Class)
prop.table(table(train_im$Class))
table(train_bl$Class)
prop.table(table(train_bl$Class))

print("Jumlah observasi data testing")
table(test_im$Class)
prop.table(table(test_im$Class))
table(test_bl$Class)
prop.table(table(test_bl$Class))

#---------------------------- K-Fold Validation
library(caret)
train_control <- trainControl(method = "repeatedcv",
                              number = 10, 
                              repeats = 10,
                              summaryFunction = multiClassSummary)

#---------------------------- SVM
#model train imbalance
set.seed(123)
mod.svm_im <- train(Class~., data = train_im, 
                    trControl = train_control, 
                    method = "svmRadial",
                    preProc = c("center", "scale"),
                    tuneLength = 10)
print(mod.svm_im)

#model train balance
set.seed(123)
mod.svm_bl <- train(Class~., data = train_bl, 
                    trControl = train_control, 
                    method = "svmRadial",
                    preProc = c("center", "scale"),
                    tuneLength = 10)
print(mod.svm_bl)

## Grafik perbandingan akurasi
akurasi_svm <- data.frame(
  C = mod.svm_bl$results$C,
  akurasi_im = mod.svm_im$results$Accuracy,
  akurasi_bl = mod.svm_bl$results$Accuracy
)

my_color <- c('steelblue', 'tomato')
akurasi_svm %>% 
  pivot_longer(-C) %>% 
  ggplot(aes(x = C, y = value, group = name, col = name)) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    x = 'Cost',
    y = 'Accuracy (Repeated Cross-Validation)',
    colour = "Data"
  ) +
  scale_color_manual(
    values = my_color,
    labels = c('Balance','Imbalance')
  ) +
  theme(
    legend.position="top",
    plot.title = element_text(face = 2, vjust = 0),
    plot.subtitle = element_text(colour = 'gray30', vjust = 0)
    
  )

#----------------------- KNN

#model train Imbalance
set.seed(123)
mod.knn_im <- train(Class~., data = train_im, 
                    trControl = train_control, 
                    method = "knn",
                    preProc = c("center", "scale"),
                    tuneLength = 10)
print(mod.knn_im)

#model train balance
set.seed(123)
mod.knn_bl <- train(Class~., data = train_bl, 
                    trControl = train_control, 
                    method = "knn",
                    preProc = c("center", "scale"),
                    tuneLength = 10)
print(mod.knn_bl)

## Grafik perbandingan akurasi
akurasi_knn <- data.frame(
  C = mod.knn_bl$results$k,
  akurasi_im = mod.knn_im$results$Accuracy,
  akurasi_bl = mod.knn_bl$results$Accuracy
)

my_color <- c('steelblue', 'tomato')
akurasi_knn %>% 
  pivot_longer(-C) %>% 
  ggplot(aes(x = C, y = value, group = name, col = name)) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    x = 'Neighbors',
    y = 'Accuracy (Repeated Cross-Validation)',
    colour = "Data"
  ) +
  scale_color_manual(
    values = my_color,
    labels = c('Balance','Imbalance')
  ) +
  theme(
    legend.position="top",
    plot.title = element_text(face = 2, vjust = 0),
    plot.subtitle = element_text(colour = 'gray30', vjust = 0)
  )

#---------------------- RF

#model train Imbalance
set.seed(123)
mod.rf_im <- train(Class~., data = train_im, 
                   trControl = train_control, 
                   method = "rf",
                   preProc = c("center", "scale"),
                   tuneLength = 10)
print(mod.rf_im)

#model train balance
set.seed(123)
mod.rf_bl <- train(Class~., data = train_bl, 
                   trControl = train_control, 
                   method = "rf",
                   preProc = c("center", "scale"),
                   tuneLength = 10)
print(mod.rf_bl)

## Grafik perbandingan akurasi
akurasi_rf <- data.frame(
  C = mod.rf_bl$results$mtry,
  akurasi_im = mod.rf_im$results$Accuracy,
  akurasi_bl = mod.rf_bl$results$Accuracy
)

my_color <- c('steelblue', 'tomato')
akurasi_rf %>% 
  pivot_longer(-C) %>% 
  ggplot(aes(x = C, y = value, group = name, col = name)) +
  geom_line() +
  geom_point(size = 2) +
  labs(
    x = 'Randomly Selected Predictors',
    y = 'Accuracy (Repeated Cross-Validation)',
    colour = "Data"
  ) +
  scale_color_manual(
    values = my_color,
    labels = c('Balance','Imbalance')
  ) +
  theme(
    legend.position="top",
    plot.title = element_text(face = 2, vjust = 0),
    plot.subtitle = element_text(colour = 'gray30', vjust = 0)
  )

#---------------------  Perbandingan

## Data Imbalance
set.seed(123)
resamps_im <- resamples(list(SVM = mod.svm_im,
                             KNN = mod.knn_im,
                             RF = mod.rf_im))
resamps_im
summary(resamps_im)

## Data balance
set.seed(123)
resamps_bl <- resamples(list(SVM = mod.svm_bl,
                             KNN = mod.knn_bl,
                             RF = mod.rf_bl))
resamps_bl
summary(resamps_bl)

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)

bwplot(resamps_im, layout = c(2,1))
bwplot(resamps_bl, layout = c(2,1))

trellis.par.set(caretTheme())
dotplot(resamps_im, metric = "Accuracy")
dotplot(resamps_im, metric = "Kappa")

dotplot(resamps_bl, metric = "Accuracy")
dotplot(resamps_bl, metric = "Kappa")

#---------------------- Prediksi
library(pROC)
library(ROCR)
library(caret)
library(tidyverse)

#---------------------- SVM
pred.svm_im <- predict(mod.svm_im, newdata = test_im)
pred.svm_bl <- predict(mod.svm_bl, newdata = test_bl)

## Confusion Matrix
print("============= Confusion Matrix Imbalance =================")
confusionMatrix(pred.svm_im, as.factor(test_im$Class))

print("============= Confusion Matrix balance =================")
confusionMatrix(pred.svm_bl, as.factor(test_bl$Class))

# ROC
auc.svm_im <- multiclass.roc(test_im$Class, as.numeric(pred.svm_im))
auc.svm_im

auc.svm_bl <- multiclass.roc(test_bl$Class, as.numeric(pred.svm_bl))
auc.svm_bl

#------------------------- KNN
pred.knn_im <- predict(mod.knn_im,test_im)
pred.knn_bl <- predict(mod.knn_bl,test_bl)

# Confusion Matrix
print("============= Confusion Matrix Imbalance =================")
confusionMatrix(pred.knn_im, as.factor(test_im$Class))

print("============= Confusion Matrix balance =================")
confusionMatrix(pred.knn_bl, as.factor(test_bl$Class))

# ROC
auc.knn_im <- multiclass.roc(test_im$Class, as.numeric(pred.knn_im))
auc.knn_im

auc.knn_bl <- multiclass.roc(test_bl$Class, as.numeric(pred.knn_bl))
auc.knn_bl

#------------------------- RF
pred.rf_im <- predict(mod.rf_im,test_im)
pred.rf_bl <- predict(mod.rf_bl,test_bl)

# Confusion Matrix
print("============= Confusion Matrix Imbalance =================")
confusionMatrix(pred.rf_im, as.factor(test_im$Class))

print("============= Confusion Matrix balance =================")
confusionMatrix(pred.rf_bl, as.factor(test_bl$Class))

# ROC
auc.rf_im <- multiclass.roc(test_im$Class, as.numeric(pred.rf_im))
auc.rf_im

auc.rf_bl <- multiclass.roc(test_bl$Class, as.numeric(pred.rf_bl))
auc.rf_bl