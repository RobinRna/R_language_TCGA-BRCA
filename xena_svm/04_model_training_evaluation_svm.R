# 04_model_training_evaluation.R
# 功能:
# 1. 加载由03脚本划分好的亚型数据集。
# 2. 对标签进行规范化处理，以适应caret内部需求，并保留原始标签用于报告。
# 3. 使用 caret::train() 对SVM（RBF核）模型进行超参数调优和交叉验证训练。
#    - 特征会在caret内部进行标准化。
#    - 处理CV中可能因类别不平衡导致的NA指标警告。
# 4. 记录并可视化超参数调优过程。
# 5. 在独立的测试集上全面评估最终选定的最优模型的性能。
#    - 输出清晰、关键的性能指标，并将类别标签映射回原始可读名称。
# 6. (如果模型支持) 提取、可视化并保存模型的特征重要性。
# 7. 保存训练好的最优caret模型对象及所有评估结果。

# --- 加载必要的R包 ---
suppressPackageStartupMessages(library(caret))       # 核心包，用于训练和评估
suppressPackageStartupMessages(library(dplyr))       # 数据操作
suppressPackageStartupMessages(library(ggplot2))     # 用于绘图 (caret的plot也可能用到)
suppressPackageStartupMessages(library(MLmetrics))   # 用于手动计算某些多分类指标 (如果需要)
suppressPackageStartupMessages(library(kernlab))   # caret的svmRadial等方法可能依赖此包
suppressPackageStartupMessages(library(e1071))     # caret的svmLinear等方法可能依赖此包

# --- 参数定义与文件路径 ---
data_splits_dir <- "data_splits"
# 【确保这些文件名与03脚本的输出完全一致】
train_file <- file.path(data_splits_dir, "train_set_subtypes.rds")
validation_file <- file.path(data_splits_dir, "validation_set_subtypes.rds") # 验证集可用于最终检查，或不使用
test_file <- file.path(data_splits_dir, "test_set_subtypes.rds")

model_output_dir <- "model"
caret_model_output_file <- file.path(model_output_dir, "caret_svm_model_tuned_subtypes.rds") # 更新文件名

results_output_dir <- "results"
tuning_results_plot_file_svm <- file.path(results_output_dir, "svm_tuning_plot.png") # 统一图名
importance_plot_file_svm <- file.path(results_output_dir, "svm_feat_imp_tuned.png")
performance_metrics_file_svm <- file.path(results_output_dir, "svm_performance_summary.txt")
# log_file_svm <- file.path(results_output_dir, "svm_training_log.txt") # 可选日志文件

cat("--- 开始模型训练与评估 (SVM with caret, 全面更新) ---\n")

# --- 1. 创建输出目录 ---
if (!dir.exists(model_output_dir)) dir.create(model_output_dir, recursive = TRUE)
if (!dir.exists(results_output_dir)) dir.create(results_output_dir, recursive = TRUE)

# --- 2. 加载数据集与标签处理 ---
cat("步骤1: 加载数据集并处理标签...\n")
if (!all(file.exists(train_file, validation_file, test_file))) {
  stop(paste0("错误: 一个或多个数据集文件未找到。请检查路径和文件名:\n",
              "Train: ", train_file, "\nValidation: ", validation_file, "\nTest: ", test_file))
}
train_set <- readRDS(train_file)
validation_set <- readRDS(validation_file) # 可用于最终评估或早停（如果train函数支持）
test_set <- readRDS(test_file)

cat("亚型数据集加载完成:\n")
cat("  训练集维度: ", paste(dim(train_set), collapse = " x "), "\n")
cat("  验证集维度: ", paste(dim(validation_set), collapse = " x "), "\n")
cat("  测试集维度: ", paste(dim(test_set), collapse = " x "), "\n")

# 规范化类别标签 (因子水平)，使其成为合法的R变量名
# 这是因为caret（特别是当classProbs=TRUE时）内部可能会将因子水平用作列名
original_levels <- levels(train_set$label)
if (is.null(original_levels) || length(original_levels) < 2) {
  stop("错误: 训练集标签不是有效的因子或类别少于2。")
}
cleaned_levels <- make.names(original_levels, unique = TRUE) # 创建R兼容的名称
# 创建映射表，用于之后将结果中的标签转换回原始可读名称
level_map_cleaned_to_original <- setNames(original_levels, cleaned_levels)

# 将所有数据集的label列的因子水平更新为清理后的名称
levels(train_set$label) <- cleaned_levels
levels(validation_set$label) <- cleaned_levels
levels(test_set$label) <- cleaned_levels

common_levels_for_model <- cleaned_levels # caret模型将使用这些清理后的level
num_classes <- length(common_levels_for_model)
cat("所有数据集的 'label' 列的因子水平已规范化为: ", paste(common_levels_for_model, collapse=", "), "\n")
cat("原始可读级别映射: \n"); print(level_map_cleaned_to_original); cat("\n")


# --- 3. 特征数量检查 ---
num_features_for_svm <- ncol(train_set) - 1 # 减去'label'列
cat("最终用于SVM训练的特征数量 (来自02/03脚本的预过滤): ", num_features_for_svm, "\n")
if (num_features_for_svm <= 0) stop("错误: 没有特征可用于训练。请检查02和03脚本的输出。")
if (num_features_for_svm > 1000) { # 示例阈值，SVM对高维数据计算成本高
  cat("警告: 特征数量 (", num_features_for_svm, ") 对于SVM可能仍然较多，训练可能非常慢。\n",
      "如果训练时间过长或内存不足，强烈建议在02脚本中设置更小的 TARGET_NUM_FEATURES_PREFILTER，\n",
      "或在此处加入额外的特征选择步骤 (例如使用 caret::rfe)。\n")
}

# --- 4. 设置caret训练控制参数 (SVM) ---
cat("\n步骤2: 设置caret训练控制参数 (SVM)...\n")
# 检查训练集中各亚型的样本数量，以辅助决定CV折叠数
min_class_samples_in_train <- min(table(train_set$label))
cat("训练集中最小类别的样本数: ", min_class_samples_in_train, "\n")

# 动态调整CV折叠数，但不小于2，且通常不超过10
cv_folds <- 5 # 默认5折
if (min_class_samples_in_train < cv_folds && min_class_samples_in_train >= 2) {
  cv_folds <- min_class_samples_in_train
  cat("警告: 由于最小类别样本数较少，CV折叠数调整为: ", cv_folds, "\n")
} else if (min_class_samples_in_train < 2) {
  stop(paste0("错误: 训练集中存在样本数小于2的亚型 ('", 
              names(which.min(table(train_set$label))), 
              "' 有 ", min_class_samples_in_train, " 个样本). ",
              "无法进行有效的交叉验证。请在02脚本中处理或合并此稀有类别。"))
}

train_control_svm <- trainControl(
  method = "cv", 
  number = cv_folds,                   # 使用动态确定的K值
  summaryFunction = multiClassSummary, # 获取全面的多分类指标
  classProbs = TRUE,                   # 允许计算类别概率 (对某些SVM方法和指标如AUC有用)
  savePredictions = "final",           # 保存最优模型在CV折叠上的预测
  verboseIter = TRUE,                  # 显示训练过程中的迭代信息
  allowParallel = TRUE                 # 如果已设置并行后端 (如doParallel)
)

# 定义SVM (RBF核) 超参数调优网格
# 【可调参数网格】这些范围需要根据经验和数据初步探索来设定
# sigma (或 gamma for e1071) 控制RBF核的宽度； C (cost) 控制对错误的惩罚。
tune_grid_svm_radial <- expand.grid(
  sigma = 10^seq(-5, -2, length.out = 4), # 例如: 0.001, 0.01, 0.1
  C = 2^seq(-10, -6, length.out = 5)        # 例如: 1, 4, 16, 64
)
cat("将对 ", nrow(tune_grid_svm_radial), " 组SVM (RBF核) 超参数组合进行评估:\n")
print(tune_grid_svm_radial)


# --- 5. 使用caret训练SVM模型 ---
cat("\n步骤3: 使用caret训练SVM模型 (含超参数调优)...\n")
set.seed(123) # 保证可复现性

caret_svm_model <- NULL
tryCatch({
  caret_svm_model <- train(
    label ~ .,                            # 公式：用label作为目标，其他所有列为特征
    data = train_set,                     # 训练数据集
    method = "svmRadial",                 # 指定使用带RBF核的SVM (caret会自动加载kernlab或e1071)
    trControl = train_control_svm,        # 传入训练控制参数 (含CV设置)
    tuneGrid = tune_grid_svm_radial,      # 传入要调优的超参数网格
    metric = "Mean_F1",                   # 选择用于评估和挑选最优模型的指标
    # 其他常用: "Accuracy", "Kappa", "Mean_Balanced_Accuracy"
    preProcess = c("center", "scale"),    # 【重要】对特征进行中心化和标准化，SVM对此敏感
    na.action = na.omit                   # 如果数据中仍有NA，则移除包含NA的行
    # prob.model = TRUE                   # 对于kernlab的ksvm，若要获取概率，需设置此参数。
    # caret的classProbs=TRUE通常会处理好这个。
  )
}, error = function(e) {
  cat("错误: caret SVM模型训练失败: ", conditionMessage(e), "\n")
  cat("详细错误对象:\n"); print(e)
  caret_svm_model <<- NULL # 确保在父环境中也设为NULL
})

if (is.null(caret_svm_model)) {
  stop("caret SVM模型未能成功训练。请检查之前的错误信息和数据 (特别是类别平衡和CV折叠数)。")
}
cat("caret SVM模型训练和调优完成。\n")

# --- 6. 查看、记录和绘制调优结果 ---
cat("\n步骤4: 查看、记录和绘制SVM调优结果...\n")
cat("最优超参数组合 (Best Tune):\n"); print(caret_svm_model$bestTune)
cat("\n调优过程中的性能指标概览 (基于最优 metric='", caret_svm_model$metric, "'):\n")
# 打印与最优模型相关的结果行 (即bestTune对应的行)
best_tune_cv_performance <- caret_svm_model$results[rownames(caret_svm_model$bestTune), ]
print(best_tune_cv_performance)
cat("\n查看所有调优参数组合的结果 (部分，前6行):\n")
print(head(caret_svm_model$results))

# 使用caret内置的plot函数绘制调优结果图
# 它会自动选择合适的绘图方式，通常是性能指标 vs. 超参数
png(tuning_results_plot_file_svm, width = 900, height = 650)
plot_obj_svm_tune <- plot(caret_svm_model, 
                          main = paste("SVM (RBF Kernel) Tuning Results\nOptimal C =", sprintf("%.2f", caret_svm_model$bestTune$C), 
                                       ", Optimal sigma =", sprintf("%.4f", caret_svm_model$bestTune$sigma)),
                          xlab = "Hyperparameter Combinations / Values", # caret的plot会根据参数数量调整
                          ylab = caret_svm_model$metric # Y轴是用于优化的指标
)
print(plot_obj_svm_tune) # 确保图被打印到设备
dev.off()
cat("SVM调优性能图已保存到: '", tuning_results_plot_file_svm, "'\n")

# 将完整的调优结果和最优参数写入文件
# 【注意】这里的sink会覆盖之前的内容，因为append=FALSE
sink(performance_metrics_file_svm, append = FALSE) 
cat("--- Caret SVM Model Tuning Results ---\n\n")
cat("Method used: svmRadial (RBF Kernel)\n")
cat("Resampling Method: ", caret_svm_model$control$method, " (", caret_svm_model$control$number, "-fold CV)\n")
cat("Metric for optimization: ", caret_svm_model$metric, "\n\n")
cat("Optimal Hyperparameters (bestTune):\n"); print(caret_svm_model$bestTune)
cat("\nPerformance for Optimal Hyperparameters (from CV on training data):\n")
print(best_tune_cv_performance) # 使用之前获取的最优参数对应的CV结果
cat("\nFull Tuning Grid Results (all hyperparameter combinations tried):\n"); print(caret_svm_model$results)
# sink() # 【重要】暂时不关闭sink，测试集结果要追加到同一个文件
cat("\nSVM调优结果已写入: '", performance_metrics_file_svm, "' (将追加测试集结果)\n")


# --- 7. 在测试集上评估【最优】SVM模型 (输出更全面的指标) ---
cat("\n步骤5: 在测试集上评估最优SVM模型 (更全面)...\n")
if (nrow(test_set) > 0 && (ncol(test_set) -1) > 0) { # 确保测试集有样本和特征
  predictions_test_class_tuned_svm <- NULL
  tryCatch({
    predictions_test_class_tuned_svm <- predict(caret_svm_model, newdata = test_set)
  }, error = function(e){
    cat("错误: 使用最优SVM模型在测试集上预测失败: ", conditionMessage(e), "\n")
    predictions_test_class_tuned_svm <<- NULL
  })
  
  if(!is.null(predictions_test_class_tuned_svm)){
    test_labels_for_eval_cleaned <- test_set$label # 已经是清理后的level
    
    if(length(predictions_test_class_tuned_svm) == length(test_labels_for_eval_cleaned) && 
       length(predictions_test_class_tuned_svm) > 0) {
      
      predictions_test_class_tuned_svm <- factor(predictions_test_class_tuned_svm, levels = common_levels_for_model)
      test_labels_for_eval_cleaned <- factor(test_labels_for_eval_cleaned, levels = common_levels_for_model)
      
      conf_matrix_test_tuned_svm <- NULL
      tryCatch({
        conf_matrix_test_tuned_svm <- confusionMatrix(
          data = predictions_test_class_tuned_svm,
          reference = test_labels_for_eval_cleaned,
          mode = "everything" 
        )
      }, error = function(e){
        cat("错误: 计算测试集混淆矩阵失败: ", conditionMessage(e), "\n")
        conf_matrix_test_tuned_svm <<- NULL
      })
      
      if(!is.null(conf_matrix_test_tuned_svm)){
        stats_by_class_svm_cleaned <- conf_matrix_test_tuned_svm$byClass
        stats_by_class_svm_original_names <- stats_by_class_svm_cleaned
        # 安全地映射行名
        valid_cleaned_names_stats <- rownames(stats_by_class_svm_cleaned)[rownames(stats_by_class_svm_cleaned) %in% names(level_map_cleaned_to_original)]
        original_names_for_stats <- level_map_cleaned_to_original[valid_cleaned_names_stats]
        rownames(stats_by_class_svm_original_names)[match(valid_cleaned_names_stats, rownames(stats_by_class_svm_original_names))] <- original_names_for_stats
        
        cm_table_original_names <- conf_matrix_test_tuned_svm$table
        # 安全地映射维度名称
        cleaned_row_names_cm <- rownames(cm_table_original_names)
        cleaned_col_names_cm <- colnames(cm_table_original_names)
        original_row_names_cm <- level_map_cleaned_to_original[cleaned_row_names_cm]
        original_col_names_cm <- level_map_cleaned_to_original[cleaned_col_names_cm]
        original_row_names_cm[is.na(original_row_names_cm)] <- cleaned_row_names_cm[is.na(original_row_names_cm)] # 回退到清理名如果映射失败
        original_col_names_cm[is.na(original_col_names_cm)] <- cleaned_col_names_cm[is.na(original_col_names_cm)]
        dimnames(cm_table_original_names) <- list(Prediction = original_row_names_cm, Reference = original_col_names_cm)
        
        cat("\n\n--- Final SVM Model Performance on Test Set (Tuned SVM) ---\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Optimal Hyperparameters used:\n", file=performance_metrics_file_svm, append=TRUE); utils::capture.output(print(caret_svm_model$bestTune), file=performance_metrics_file_svm, append=TRUE)
        cat("Number of features used in model: ", num_features_for_svm, "\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Subtype categories (original readable names): ", paste(original_levels, collapse=", "), "\n\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Confusion Matrix (Test Set - Original Readable Names):\n", file=performance_metrics_file_svm, append=TRUE); utils::capture.output(print(cm_table_original_names), file=performance_metrics_file_svm, append=TRUE)
        cat("\nOverall Statistics (Test Set):\n", file=performance_metrics_file_svm, append=TRUE); utils::capture.output(print(conf_matrix_test_tuned_svm$overall), file=performance_metrics_file_svm, append=TRUE)
        cat("\nStatistics by Class (Test Set - Original Readable Names):\n", file=performance_metrics_file_svm, append=TRUE); utils::capture.output(print(stats_by_class_svm_original_names, digits=4), file=performance_metrics_file_svm, append=TRUE)
        
        precision_by_class_svm <- stats_by_class_svm_cleaned[, "Pos Pred Value"]; recall_by_class_svm <- stats_by_class_svm_cleaned[, "Sensitivity"]; f1_by_class_svm <- stats_by_class_svm_cleaned[, "F1"]
        precision_by_class_svm[is.na(precision_by_class_svm)] <- 0; recall_by_class_svm[is.na(recall_by_class_svm)] <- 0; f1_by_class_svm[is.na(f1_by_class_svm)] <- 0
        macro_precision_svm <- mean(precision_by_class_svm); macro_recall_svm <- mean(recall_by_class_svm); macro_f1_svm <- mean(f1_by_class_svm)
        support_by_class_svm <- colSums(conf_matrix_test_tuned_svm$table)
        weighted_precision_svm <- weighted.mean(precision_by_class_svm, w=support_by_class_svm, na.rm=TRUE); weighted_recall_svm <- weighted.mean(recall_by_class_svm, w=support_by_class_svm, na.rm=TRUE); weighted_f1_svm <- weighted.mean(f1_by_class_svm, w=support_by_class_svm, na.rm=TRUE)
        
        cat("\n--- Averaged Metrics (Test Set - SVM) ---\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Macro-Averaged Precision: ", sprintf("%.4f", macro_precision_svm), "\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Macro-Averaged Recall:    ", sprintf("%.4f", macro_recall_svm), "\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Macro-Averaged F1-Score:  ", sprintf("%.4f", macro_f1_svm), "\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Weighted-Averaged Precision: ", sprintf("%.4f", weighted_precision_svm), "\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Weighted-Averaged Recall:    ", sprintf("%.4f", weighted_recall_svm), "\n", file=performance_metrics_file_svm, append=TRUE)
        cat("Weighted-Averaged F1-Score:  ", sprintf("%.4f", weighted_f1_svm), "\n", file=performance_metrics_file_svm, append=TRUE)
        cat("\nSVM调优和测试集评估结果已完整保存到: '", performance_metrics_file_svm, "'\n")
      } else { cat("警告: SVM测试集混淆矩阵未能计算。\n")}
    } else { cat("警告: SVM测试集预测或标签数据不足、不一致或为空。\n") }
  } else { cat("警告: SVM在测试集上预测失败。\n")}
} else { cat("测试集为空或无特征，跳过SVM最终模型评估。\n")}
# 关闭sink应该在所有写入完成后进行
if(exists("performance_metrics_file_svm") && file.info(performance_metrics_file_svm)$size > 0) sink() # 只有在sink被打开且写入内容后才关闭

if(exists("predictions_test_class_tuned_svm")) rm(predictions_test_class_tuned_svm)
if(exists("conf_matrix_test_tuned_svm")) rm(conf_matrix_test_tuned_svm); gc()


# --- 8. SVM特征重要性 (如果可用) ---
cat("\n步骤6: 提取最终SVM模型的特征重要性 (如果可用)...\n")
importance_obj_svm <- NULL; tryCatch({ importance_obj_svm <- varImp(caret_svm_model, scale = FALSE) }, error = function(e) { cat("提取SVM varImp失败:",conditionMessage(e),"\n")})
if (!is.null(importance_obj_svm) && !is.null(importance_obj_svm$importance) && nrow(importance_obj_svm$importance) > 0) {
  imp_df_svm <- importance_obj_svm$importance
  imp_scores_svm <- if("Overall" %in% colnames(imp_df_svm)) imp_df_svm[,"Overall", drop=FALSE] else imp_df_svm[,1, drop=FALSE]
  importance_df_svm_sorted <- data.frame(Feature = rownames(imp_scores_svm), Importance = imp_scores_svm[[1]]) %>% arrange(desc(Importance))
  cat("SVM特征重要性 (Top 20):\n"); print(head(importance_df_svm_sorted, 20))
  png(importance_plot_file_svm, width = 1000, height = 700)
  p_imp_svm <- ggplot(head(importance_df_svm_sorted, 20), aes(x = reorder(Feature, Importance), y = Importance)) + 
    geom_bar(stat="identity", fill="steelblue") + coord_flip() + 
    labs(title="Top 20 Important Features (Tuned SVM)", x="Feature", y="Importance Score") + 
    theme_minimal(base_size = 10) + theme(plot.title = element_text(hjust = 0.5))
  print(p_imp_svm); dev.off()
  cat("SVM特征重要性图已保存到: '", importance_plot_file_svm, "'\n")
} else { cat("警告: 未能从调优的SVM模型中提取到有效的特征重要性数据。\n") }
if(exists("importance_obj_svm")) rm(importance_obj_svm); if(exists("importance_df_svm_sorted")) rm(importance_df_svm_sorted); gc()


# --- 9. 保存训练好的【最优】caret SVM模型 ---
cat("\n步骤7: 保存训练好的最优caret SVM模型...\n")
saveRDS(caret_svm_model, file = caret_model_output_file)
cat("最优caret SVM模型已保存到: '", caret_model_output_file, "'\n")
if(exists("caret_svm_model")) rm(caret_svm_model); gc()

cat("\n--- 模型训练与评估 (使用SVM通过caret进行调优) 完成！ ---\n")
# if(exists("log_file_svm") && file.exists(log_file_svm)) sink() # 如果之前开启了全局日志