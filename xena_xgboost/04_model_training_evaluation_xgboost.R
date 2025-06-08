# 04_model_training_evaluation.R (直接xgboost, 手动网格搜索, 增强评估和绘图, 加入内存管理提示)

# --- 加载必要的R包 ---
suppressPackageStartupMessages(library(xgboost))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(MLmetrics))
suppressPackageStartupMessages(library(caret)) # 仅用于confusionMatrix
suppressPackageStartupMessages(library(reshape2))
# install.packages("pryr") # 可选，用于内存监控 pryr::mem_used()

# --- 参数定义与文件路径 ---
# ... (与上一版本相同) ...
data_splits_dir <- "data_splits"
train_file <- file.path(data_splits_dir, "train_set_subtypes.rds")
validation_file <- file.path(data_splits_dir, "validation_set_subtypes.rds")
test_file <- file.path(data_splits_dir, "test_set_subtypes.rds")
model_output_dir <- "model"
xgb_model_output_file <- file.path(model_output_dir, "xgboost_model_tuned_direct_subtypes.rds")
results_output_dir <- "results"
param_search_results_file <- file.path(results_output_dir, "xgb_param_search_results.csv")
cv_metrics_plot_file <- file.path(results_output_dir, "xgb_best_cv_metrics_plot.png")
train_metrics_plot_file <- file.path(results_output_dir, "xgb_final_train_metrics_plot.png")
importance_plot_file <- file.path(results_output_dir, "xgb_tuned_direct_feat_imp.png")
performance_metrics_file <- file.path(results_output_dir, "xgb_tuned_direct_perf_metrics.txt")

cat("--- 开始模型训练与评估 (XGBoost手动调优, 增强评估, 内存管理) ---\n")

# --- 1. 创建输出目录 ---
if (!dir.exists(model_output_dir)) dir.create(model_output_dir, recursive = TRUE)
if (!dir.exists(results_output_dir)) dir.create(results_output_dir, recursive = TRUE)

# --- 2. 加载数据集与标签处理 ---
cat("步骤1: 加载数据集并处理标签...\n")
if (!all(file.exists(train_file, validation_file, test_file))) stop("错误: 一个或多个数据集文件未找到。")
train_set_df <- readRDS(train_file)
validation_set_df <- readRDS(validation_file)
test_set_df <- readRDS(test_file)
# ... (标签数值化逻辑同前) ...
original_levels <- levels(train_set_df$label)
if (is.null(original_levels) || length(original_levels) < 2) stop("错误: 标签问题。")
level_map_to_numeric <- setNames(0:(length(original_levels)-1), original_levels)
numeric_map_to_original_factor <- setNames(factor(original_levels, levels=original_levels), 0:(length(original_levels)-1))
train_labels_numeric <- level_map_to_numeric[as.character(train_set_df$label)]
validation_labels_numeric <- level_map_to_numeric[as.character(validation_set_df$label)]
test_labels_numeric <- level_map_to_numeric[as.character(test_set_df$label)]
num_classes <- length(original_levels)
cat("类别数量: ", num_classes, "\n")
final_num_features <- ncol(train_set_df) - 1
cat("用于XGBoost训练的特征数量: ", final_num_features, "\n")


# --- 3. 准备XGBoost数据格式 (xgb.DMatrix) ---
cat("\n步骤2: 准备XGBoost数据格式...\n")
train_features_matrix <- as.matrix(train_set_df[, -which(names(train_set_df) == "label")])
validation_features_matrix <- as.matrix(validation_set_df[, -which(names(validation_set_df) == "label")])
test_features_matrix <- as.matrix(test_set_df[, -which(names(test_set_df) == "label")])
# 移除原始数据框以节省内存
rm(train_set_df, validation_set_df, test_set_df)
gc() # 尝试回收内存
cat("原始数据框已移除。\n")

if(!is.numeric(train_features_matrix)) storage.mode(train_features_matrix) <- "numeric"
if(!is.numeric(validation_features_matrix)) storage.mode(validation_features_matrix) <- "numeric"
if(!is.numeric(test_features_matrix)) storage.mode(test_features_matrix) <- "numeric"

dtrain <- xgb.DMatrix(data = train_features_matrix, label = train_labels_numeric)
dvalidation <- xgb.DMatrix(data = validation_features_matrix, label = validation_labels_numeric)
dtest <- xgb.DMatrix(data = test_features_matrix, label = test_labels_numeric)
# 特征矩阵在创建DMatrix后理论上可以移除，但xgb.importance可能需要colnames
# 如果内存非常紧张，可以在xgb.importance之前重新从dtrain中提取特征名
# rm(train_features_matrix, validation_features_matrix, test_features_matrix)
# gc()
cat("xgb.DMatrix对象创建完成。\n")


# --- 4. 定义超参数网格进行手动搜索 ---
cat("\n步骤3: 定义XGBoost超参数网格...\n")
# 【可调参数网格】
param_grid <- expand.grid(
  eta = c(0.05, 0.075, 0.1),
  max_depth = c(3, 4, 5),
  subsample = c(0.5, 0.6, 0.7),
  colsample_bytree = c(0.5, 0.6, 0.7),
  min_child_weight = c(1),
  gamma = c(0)
)
cat("将对 ", nrow(param_grid), " 组超参数组合进行评估。\n")

# --- 5. 手动网格搜索与交叉验证 ---
cat("\n步骤4: 开始手动网格搜索与交叉验证...\n")
set.seed(123)
NROUNDS_CV_MAX <- 200
EARLY_STOPPING_ROUNDS_CV <- 20
cv_fold_n <- 5
all_cv_results <- list()
best_cv_metric_value <- Inf
best_params_from_cv <- NULL
best_nrounds_from_cv <- NULL

for (i in 1:nrow(param_grid)) {
  current_params <- param_grid[i, ]
  cat("\n评估参数组合 ", i, "/", nrow(param_grid), ":\n")
  print(current_params)
  xgb_cv_params_list <- list(
    objective = "multi:softprob", eval_metric = "mlogloss", eval_metric = "merror",
    num_class = num_classes, eta = current_params$eta, max_depth = current_params$max_depth,
    subsample = current_params$subsample, colsample_bytree = current_params$colsample_bytree,
    min_child_weight = current_params$min_child_weight, gamma = current_params$gamma
  )
  xgb_cv_run <- NULL
  tryCatch({
    xgb_cv_run <- xgb.cv(
      params = xgb_cv_params_list, data = dtrain, nrounds = NROUNDS_CV_MAX,
      nfold = cv_fold_n, showsd = TRUE, stratified = TRUE, print_every_n = 50,
      early_stopping_rounds = EARLY_STOPPING_ROUNDS_CV, maximize = FALSE, verbose = 0
    )
  }, error = function(e) { cat("错误: xgb.cv对参数组合 ", i, " 失败: ", conditionMessage(e), "\n"); xgb_cv_run <<- NULL })
  
  if (!is.null(xgb_cv_run) && !is.null(xgb_cv_run$best_iteration)) {
    current_best_iteration <- xgb_cv_run$best_iteration
    current_best_metric <- xgb_cv_run$evaluation_log[current_best_iteration, test_mlogloss_mean]
    cat("  参数组合 ", i, ": CV完成。最佳轮数 (mlogloss): ", current_best_iteration,
        ", 对应 Test mlogloss: ", sprintf("%.5f", current_best_metric), "\n")
    all_cv_results[[i]] <- list(
      params = current_params, best_iteration = current_best_iteration,
      best_mlogloss = current_best_metric, full_log = xgb_cv_run$evaluation_log
    )
    if (current_best_metric < best_cv_metric_value) {
      best_cv_metric_value <- current_best_metric
      best_params_from_cv <- current_params
      best_nrounds_from_cv <- current_best_iteration
      cat("  >> 新的最优参数组合 found!\n")
    }
  } else {
    cat("  参数组合 ", i, ": CV失败或未找到最佳轮数。\n")
    all_cv_results[[i]] <- list(params=current_params, error_msg="CV Failed")
  }
  # 在每次CV迭代后尝试回收内存
  rm(xgb_cv_run) # 移除当次CV的结果对象
  gc()
  cat("  内存已尝试回收。\n")
}

 # ... (保存CV摘要和绘制最优CV曲线逻辑同前) ...
cv_summary_df <- do.call(rbind, lapply(all_cv_results, function(res) {
  if("error_msg" %in% names(res)) { data.frame(as.list(res$params), best_iteration=NA, best_mlogloss=NA, error=res$error_msg)
  } else { data.frame(as.list(res$params), best_iteration=res$best_iteration, best_mlogloss=res$best_mlogloss, error=NA) }
}))
write.csv(cv_summary_df, param_search_results_file, row.names = FALSE)
cat("\n所有CV参数搜索结果已保存到: '", param_search_results_file, "'\n")
if (is.null(best_params_from_cv) || is.null(best_nrounds_from_cv)) stop("未能通过网格搜索找到有效的最优参数组合。")
cat("\n--- 网格搜索完成 --- \n"); print(best_params_from_cv); cat("最佳轮数: ", best_nrounds_from_cv, "\n")
best_cv_run_log <- NULL; for(res in all_cv_results){ if(!is.null(res$params) && all(res$params == best_params_from_cv)){ best_cv_run_log <- res$full_log; break } }
if(!is.null(best_cv_run_log)){
  best_cv_log_df_melt <- melt(as.data.frame(best_cv_run_log), id.vars = "iter", measure.vars = c("train_mlogloss_mean", "test_mlogloss_mean", "train_merror_mean", "test_merror_mean"), variable.name = "metric_type", value.name = "value")
  png(cv_metrics_plot_file, width = 1200, height = 700); p_best_cv <- ggplot(best_cv_log_df_melt, aes(x = iter, y = value, color = metric_type)) + geom_line(linewidth = 1) + geom_vline(xintercept = best_nrounds_from_cv, linetype = "dashed", color = "red") + labs(title = paste("XGBoost CV Metrics for Best Params (nrounds ~", best_nrounds_from_cv, ")"), x = "Number of Rounds", y = "Metric Value") + theme_minimal() + theme(legend.position = "top"); print(p_best_cv); dev.off()
  cat("最优参数组合的CV指标曲线图已保存到: '", cv_metrics_plot_file, "'\n")
}
rm(all_cv_results, cv_summary_df, best_cv_run_log, best_cv_log_df_melt, xgb_cv_params_list) # 清理CV相关大对象
gc()


# --- 6. 训练最终XGBoost模型 (使用最优参数) ---
cat("\n步骤5: 训练最终XGBoost模型 (使用最优参数和nrounds)...\n")
# ... (final_params_list 定义和最终模型训练逻辑同前) ...
final_params_list <- list( objective = "multi:softprob", eval_metric = "mlogloss", eval_metric = "merror", num_class = num_classes, eta = best_params_from_cv$eta, max_depth = best_params_from_cv$max_depth, subsample = best_params_from_cv$subsample, colsample_bytree = best_params_from_cv$colsample_bytree, min_child_weight = best_params_from_cv$min_child_weight, gamma = best_params_from_cv$gamma)
set.seed(123); watchlist <- list(train = dtrain, eval = dvalidation); final_xgb_model <- NULL; training_history_final <- NULL
tryCatch({
  final_xgb_model <- xgb.train( params = final_params_list, data = dtrain, nrounds = best_nrounds_from_cv, watchlist = watchlist, print_every_n = 10, verbose = 1)
  if (!is.null(final_xgb_model$evaluation_log)) training_history_final <- as.data.frame(final_xgb_model$evaluation_log)
}, error = function(e) { cat("错误: 训练最终XGBoost模型时失败: ", conditionMessage(e), "\n"); final_xgb_model <<- NULL })
if(is.null(final_xgb_model)) stop("最终XGBoost模型训练失败。")
cat("最终XGBoost模型训练完成。\n")

# 绘制最终模型训练过程中的指标曲线
# ... (绘制训练历史曲线的代码同前) ...
if (!is.null(training_history_final) && nrow(training_history_final) > 0) {
  train_hist_final_melt <- melt(training_history_final, id.vars = "iter", measure.vars = grep("^(train|eval)_(mlogloss|merror)$", names(training_history_final), value = TRUE), variable.name = "metric_type", value.name = "value")
  png(train_metrics_plot_file, width = 1200, height = 700); p_train_hist_final <- ggplot(train_hist_final_melt, aes(x = iter, y = value, color = metric_type)) + geom_line(linewidth = 1) + labs(title = "Final XGBoost Model Training History", x = "Number of Rounds", y = "Metric Value") + theme_minimal() + theme(legend.position = "top"); print(p_train_hist_final); dev.off()
  cat("最终模型训练历史指标曲线图已保存到: '", train_metrics_plot_file, "'\n")
}
rm(training_history_final, train_hist_final_melt) # 清理
gc()

# --- 7. 在测试集上评估模型性能 (更全面) ---
cat("\n步骤6: 在测试集上评估最终XGBoost模型 (更全面)...\n")
# 使用 final_xgb_model, dtest, test_labels_numeric, numeric_map_to_original_factor, original_levels

# 【修改这里的条件】
# 检查 test_features_matrix 是否有行和列，并且 test_labels_numeric 也有对应的长度
if (exists("test_features_matrix") && nrow(test_features_matrix) > 0 && ncol(test_features_matrix) > 0 &&
    exists("test_labels_numeric") && length(test_labels_numeric) == nrow(test_features_matrix)) {
  
  pred_probs_xgb_test <- predict(final_xgb_model, dtest, reshape = TRUE)
  pred_labels_numeric_xgb_test <- max.col(pred_probs_xgb_test) - 1
  pred_labels_factor_xgb_test <- numeric_map_to_original_factor[as.character(pred_labels_numeric_xgb_test)]
  test_labels_factor_original <- numeric_map_to_original_factor[as.character(test_labels_numeric)]
  pred_labels_factor_xgb_test <- factor(pred_labels_factor_xgb_test, levels = original_levels)
  test_labels_factor_original <- factor(test_labels_factor_original, levels = original_levels)
  
  if(length(pred_labels_factor_xgb_test) == length(test_labels_factor_original) && length(pred_labels_factor_xgb_test) > 0) {
    conf_matrix_test_xgb <- confusionMatrix(pred_labels_factor_xgb_test, test_labels_factor_original)
    # ... (后续的手动计算Precision/Recall/F1和写入文件的逻辑不变) ...
    precision_by_class <- diag(conf_matrix_test_xgb$table) / colSums(conf_matrix_test_xgb$table); recall_by_class <- diag(conf_matrix_test_xgb$table) / rowSums(conf_matrix_test_xgb$table); f1_by_class <- 2 * (precision_by_class * recall_by_class) / (precision_by_class + recall_by_class); f1_by_class[is.na(f1_by_class)] <- 0; macro_precision <- mean(precision_by_class, na.rm = TRUE); macro_recall <- mean(recall_by_class, na.rm = TRUE); macro_f1 <- mean(f1_by_class, na.rm = TRUE); support_by_class <- rowSums(conf_matrix_test_xgb$table); weighted_precision <- weighted.mean(precision_by_class, w = support_by_class, na.rm = TRUE); weighted_recall <- weighted.mean(recall_by_class, w = support_by_class, na.rm = TRUE); weighted_f1 <- weighted.mean(f1_by_class, w = support_by_class, na.rm = TRUE);
    sink(performance_metrics_file, append = TRUE); # 追加到之前的CV结果之后
    cat("\n\n--- Tuned XGBoost Model Performance on Test Set (Direct Usage, Full Eval) ---\n\n"); # 加个换行
    if(exists("best_params_from_cv")) { cat("Best Hyperparameters from CV:\n"); print(best_params_from_cv) }
    if(exists("best_nrounds_from_cv")) { cat("\nNrounds used (from CV): ", best_nrounds_from_cv, "\n") }
    if(exists("final_num_features")) { cat("Features used in model: ", final_num_features, "\n") }
    cat("亚型类别 (原始): ", paste(original_levels, collapse=", "), "\n\n");
    print(conf_matrix_test_xgb);
    cat("\n--- Manually Calculated Metrics ---\nPrecision by Class:\n"); print(precision_by_class);
    cat("Recall by Class:\n"); print(recall_by_class);
    cat("F1-Score by Class:\n"); print(f1_by_class);
    cat("\nMacro-Averaged Precision: ", sprintf("%.4f", macro_precision), "\nMacro-Averaged Recall:    ", sprintf("%.4f", macro_recall), "\nMacro-Averaged F1-Score:  ", sprintf("%.4f", macro_f1), "\n\nWeighted-Averaged Precision: ", sprintf("%.4f", weighted_precision), "\nWeighted-Averaged Recall:    ", sprintf("%.4f", weighted_recall), "\nWeighted-Averaged F1-Score:  ", sprintf("%.4f", weighted_f1), "\n\n--- Caret Overall Statistics ---\nAccuracy: ", sprintf("%.4f", conf_matrix_test_xgb$overall['Accuracy']), "\nKappa:    ", sprintf("%.4f", conf_matrix_test_xgb$overall['Kappa']), "\n");
    sink()
    cat("Tuned XGBoost测试集评估结果已保存到: '", performance_metrics_file, "'\n")
  } else { cat("警告: Tuned XGBoost测试集预测或标签数据不一致或为空。\n") }
} else {
  cat("测试集特征矩阵为空、无特征或标签不匹配，跳过Tuned XGBoost最终模型评估。\n")
}
rm(pred_probs_xgb_test, pred_labels_numeric_xgb_test, pred_labels_factor_xgb_test, test_labels_factor_original, conf_matrix_test_xgb) # 清理
gc()

# --- 8. XGBoost特征重要性 ---
cat("\n步骤7: 提取、排序并可视化最终XGBoost模型的特征重要性...\n")
# xgb.importance 需要原始特征名，我们是从train_features_matrix的colnames获取
original_feature_names_for_imp <- colnames(train_features_matrix) # 获取用于训练的特征名
rm(train_features_matrix, validation_features_matrix, test_features_matrix, dtrain, dvalidation, dtest) # 在获取特征名后，现在可以安全移除这些大对象
gc()
cat("大型矩阵和DMatrix对象已移除。\n")

tryCatch({
  importance_matrix_xgb <- xgb.importance(feature_names = original_feature_names_for_imp, model = final_xgb_model)
  if (!is.null(importance_matrix_xgb) && nrow(importance_matrix_xgb) > 0) {
    cat("Tuned XGBoost特征重要性 (Top 20):\n"); print(head(importance_matrix_xgb, 20))
    png(importance_plot_file, width = 1000, height = 700)
    xgb.plot.importance(importance_matrix_xgb, top_n = min(20, nrow(importance_matrix_xgb)), main = "Top Important Features (Tuned XGBoost Direct)")
    dev.off()
    cat("Tuned XGBoost特征重要性图已保存到: '", importance_plot_file, "'\n")
  } else { cat("警告: 未能从Tuned XGBoost模型中提取特征重要性。\n") }
}, error = function(e) { cat("错误: 提取Tuned XGBoost特征重要性时失败: ", conditionMessage(e), "\n") })
rm(importance_matrix_xgb, original_feature_names_for_imp) # 清理
gc()

# --- 9. 保存训练好的XGBoost模型 ---
cat("\n步骤8: 保存训练好的最终XGBoost模型...\n")
xgb.save(final_xgb_model, fname = xgb_model_output_file)
cat("最终XGBoost模型已保存到: '", xgb_model_output_file, "'\n")
rm(final_xgb_model) # 清理
gc()

cat("\n--- 模型训练与评估 (XGBoost手动调优, 增强评估, 内存管理) 完成！ ---\n")
# sink() # 关闭日志