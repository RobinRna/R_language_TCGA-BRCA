# 05_mlp_classification_subtypes.R
# 功能:
# 1. 加载划分好的亚型数据集。
# 2. 准备数据 (特征缩放, 标签one-hot编码)。
# 3. 定义MLP参数网格。
# 4. 循环遍历所有参数网格组合，训练和评估MLP模型。
# 5. 选择最佳模型，并在测试集上评估。
# 6. 绘制最佳模型的训练历史、混淆矩阵。
# 7. 保存最佳模型、参数和评估结果。

# --- 加载必要的R包 ---
suppressPackageStartupMessages(library(keras))
suppressPackageStartupMessages(library(tensorflow))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(MLmetrics))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(stringr))

# --- 激活 Virtualenv 环境 ---
cat("正在激活 virtualenv 环境: tf210_gpu_venv...\n")
tryCatch({
  reticulate::use_virtualenv("tf210_gpu_venv", required = TRUE)
  cat("virtualenv 环境 tf210_gpu_venv 已成功激活。\n")
}, error = function(e) {
  cat("错误: 无法激活 virtualenv 环境 tf210_gpu_venv. 请确保该环境已存在且配置正确。\n")
  cat("您可能需要重新运行：\n")
  cat("1. install_tensorflow(version = \"2.10.0\", method = \"virtualenv\", gpu = TRUE, envname = \"tf210_gpu_venv\")\n")
  cat("2. (在新会话中) reticulate::use_virtualenv(\"tf210_gpu_venv\", required = TRUE)\n")
  cat("3. (在新会话中) reticulate::pip_install(\"numpy==1.23.4\", envname = \"tf210_gpu_venv\", force = TRUE)\n")
  stop(e$message)
})

# --- TensorFlow GPU 配置检查 ---
cat("\n--- TensorFlow 和 GPU 配置检查 ---\n")
cat("TensorFlow R 包版本:", as.character(packageVersion("tensorflow")), "\n")
cat("Keras R 包版本:", as.character(packageVersion("keras")), "\n")
cat("Reticulate Python 配置:\n")
print(reticulate::py_config())

cat("\nPython TensorFlow 版本:", as.character(tf$`__version__`), "\n")
cat("Python Keras 版本 (通过 TensorFlow):", as.character(tf$keras$`__version__`), "\n")
cat("Python NumPy 版本:\n")
reticulate::py_run_string("import numpy; print(numpy.__version__)")


gpu_devices <- NULL
tryCatch({
  gpu_devices <- tf$config$list_physical_devices("GPU")
}, error = function(e) {
  message("警告：查询 GPU 设备失败。")
  message("错误详情：", e$message)
  gpu_devices <- list()
})

if (length(gpu_devices) > 0) {
  cat("\n检测到以下 GPU 设备:\n"); print(gpu_devices)
  cat("TensorFlow GPU 已配置。\n")
  tryCatch({
    tf$config$experimental$set_memory_growth(gpu_devices[[1L]], TRUE) # 使用 L 后缀确保是整数
    cat("已为 GPU 0 启用内存增长。\n")
  }, error = function(e) {
    cat("警告：为 GPU 设置内存增长失败: ", conditionMessage(e), "\n")
  })
} else {
  cat("\n未检测到 GPU。TensorFlow 将使用 CPU 运行。\n")
  cat("如果您希望使用 GPU，请确保 CUDA (11.2) 和 cuDNN (8.1.0) 已正确安装并配置环境变量，\n")
  cat("并且 TensorFlow 2.10.0 GPU 版本已在 'tf210_gpu_venv'环境中通过 NumPy 1.23.4 (或兼容版本) 安装。\n")
}

# --- 设置全局随机种子 ---
set.seed(123) # R 全局种子

# --- 参数定义与文件路径 ---
data_splits_dir <- "data_splits"
train_file <- file.path(data_splits_dir, "train_set_subtypes.rds")
validation_file <- file.path(data_splits_dir, "validation_set_subtypes.rds")
test_file <- file.path(data_splits_dir, "test_set_subtypes.rds")

model_output_dir <- "model_mlp"
results_output_dir <- "results_mlp"

if (!dir.exists(model_output_dir)) dir.create(model_output_dir, recursive = TRUE)
if (!dir.exists(results_output_dir)) dir.create(results_output_dir, recursive = TRUE)

best_model_output_file <- file.path(model_output_dir, "best_mlp_model_subtypes.keras")
history_plot_file <- file.path(results_output_dir, "mlp_training_history_plot_subtypes.png")
confusion_matrix_plot_file <- file.path(results_output_dir, "mlp_confusion_matrix_plot_subtypes.png")
performance_metrics_file <- file.path(results_output_dir, "mlp_performance_metrics_subtypes.txt")
parameter_tuning_results_file <- file.path(results_output_dir, "mlp_parameter_tuning_results.csv")

cat("\n--- 开始MLP模型参数搜索、训练与评估 (用于癌症亚型多分类) ---\n")

# --- 1. 加载数据集 ---
cat("步骤1: 加载亚型训练集、验证集和测试集...\n")
train_set <- readr::read_rds(train_file)
validation_set <- readr::read_rds(validation_file)
test_set <- readr::read_rds(test_file)
cat("数据集加载完成。\n")

if (!is.factor(train_set$label)) train_set$label <- as.factor(train_set$label)
common_levels <- levels(train_set$label)
num_classes <- length(common_levels)
if (num_classes < 2) stop("错误: 亚型类别少于2，无法进行多分类。")
cat("检测到 ", num_classes, " 个亚型类别: ", paste(common_levels, collapse=", "), "\n")

train_set$label <- factor(train_set$label, levels = common_levels)
validation_set$label <- factor(validation_set$label, levels = common_levels)
test_set$label <- factor(test_set$label, levels = common_levels)

# --- 2. 数据预处理 (Keras 专属) ---
cat("\n步骤2: 为Keras模型准备数据...\n")
x_train <- train_set %>% select(-label) %>% as.matrix()
y_train_labels <- train_set$label
x_val <- validation_set %>% select(-label) %>% as.matrix()
y_val_labels <- validation_set$label
x_test <- test_set %>% select(-label) %>% as.matrix()
y_test_labels <- test_set$label

colnames(x_train) <- make.names(colnames(x_train), unique = TRUE)
colnames(x_val) <- make.names(colnames(x_val), unique = TRUE)
colnames(x_test) <- make.names(colnames(x_test), unique = TRUE)

train_mean <- apply(x_train, 2, mean, na.rm = TRUE)
train_sd <- apply(x_train, 2, sd, na.rm = TRUE)
train_sd[train_sd == 0 | is.na(train_sd)] <- 1e-6

x_train_scaled <- scale(x_train, center = train_mean, scale = train_sd)
x_val_scaled <- scale(x_val, center = train_mean, scale = train_sd)
x_test_scaled <- scale(x_test, center = train_mean, scale = train_sd)
cat("特征数据已完成标准化处理。\n")

y_train_numeric <- as.numeric(y_train_labels) - 1L # 确保是整数
y_val_numeric <- as.numeric(y_val_labels) - 1L
y_test_numeric <- as.numeric(y_test_labels) - 1L

y_train_one_hot <- to_categorical(y_train_numeric, num_classes = num_classes)
y_val_one_hot <- to_categorical(y_val_numeric, num_classes = num_classes)
y_test_one_hot <- to_categorical(y_test_numeric, num_classes = num_classes)
cat("标签数据已完成 One-Hot 编码。\n")

input_shape <- ncol(x_train_scaled)
cat("模型输入形状 (特征数量): ", input_shape, "\n")

# --- 3. 定义MLP参数网格 (完整版) ---
cat("\n步骤3: 定义MLP参数网格 (调整后，用于快速演示)...\n")
param_grid_mlp <- expand.grid(
  units1 = c(64L, 128L, 256L),      # 使用 L 后缀指定整数
  units2 = c(32L, 64L, 128L),
  dropout_rate1 = c(0.2, 0.3, 0.4), # Dropout 率是浮点数
  dropout_rate2 = c(0.2, 0.3, 0.4),
  learning_rate = c(0.01, 0.001, 0.0001, 0.00001),
  batch_size = c(16L, 32L, 64L),    # 批大小是整数
  epochs = c(30L, 50L, 100L)             # 周期数是整数
)
# 筛选，确保 units1 >= units2
param_grid_mlp <- subset(param_grid_mlp, units1 >= units2)
cat("将测试以下 ", nrow(param_grid_mlp), " 组MLP参数组合。\n")
cat("参数网格示例:\n")
print(head(param_grid_mlp, 5))
# 即使这样调整，组合数可能仍然不少，可以根据需要进一步减少每个参数的选项。
# 例如，只选一组dropout，一组batch_size等。

# --- 4. 参数搜索与模型训练 ---
cat("\n步骤4: 开始MLP参数搜索和模型训练...\n")
tuning_results_list <- list()
best_val_accuracy <- -Inf
best_params_mlp <- NULL
best_mlp_model_history <- NULL
best_epoch_model <- NULL

for (i in 1:nrow(param_grid_mlp)) {
  current_params <- param_grid_mlp[i, ]
  cat(paste0("\n--- MLP参数组合 ", i, "/", nrow(param_grid_mlp), " ---\n"))
  print(current_params)

  k_clear_session() # 清除之前的Keras会话
  tensorflow::set_random_seed(123L + i) # 确保每次循环种子略有不同但仍可控

  model_mlp <- keras_model_sequential(name = paste0("mlp_model_config_", i)) %>%
    layer_dense(units = current_params$units1, activation = "relu", input_shape = input_shape, # units1 已经是整数
                kernel_regularizer = regularizer_l2(0.001), name = "dense_1") %>%
    layer_batch_normalization(name = "bn_1") %>%
    layer_dropout(rate = current_params$dropout_rate1, name = "dropout_1") %>% # dropout_rate1 是浮点数
    layer_dense(units = current_params$units2, activation = "relu", # units2 已经是整数
                kernel_regularizer = regularizer_l2(0.001), name = "dense_2") %>%
    layer_batch_normalization(name = "bn_2") %>%
    layer_dropout(rate = current_params$dropout_rate2, name = "dropout_2") %>% # dropout_rate2 是浮点数
    layer_dense(units = as.integer(num_classes), activation = "softmax", name = "output_dense") # 确保 num_classes 是整数

  model_mlp %>% compile(
    optimizer = optimizer_adam(learning_rate = current_params$learning_rate), # learning_rate 是浮点数
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )

  callbacks_list <- list(
    callback_early_stopping(monitor = "val_accuracy", patience = 15L, mode = "max", restore_best_weights = TRUE), # patience 是整数
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 7L, min_lr = 0.00001) # patience 是整数
  )
  
  cat("训练MLP模型 (配置 ", i, ")...\n")
  history_mlp <- NULL
  tryCatch({
    history_mlp <- model_mlp %>% fit(
      x_train_scaled, y_train_one_hot,
      epochs = current_params$epochs, # epochs 已经是整数
      batch_size = current_params$batch_size, # batch_size 已经是整数
      validation_data = list(x_val_scaled, y_val_one_hot),
      callbacks = callbacks_list,
      verbose = 1
    )
  }, error = function(e) {
    cat("错误: 训练MLP模型 (配置 ", i, ") 失败: ", conditionMessage(e), "\n")
    history_mlp <<- NULL
  })

  if (!is.null(history_mlp) && !is.null(history_mlp$metrics$val_accuracy) && length(history_mlp$metrics$val_accuracy) > 0) {
    current_val_accuracy <- max(history_mlp$metrics$val_accuracy, na.rm = TRUE)
    cat("当前配置验证集最高准确率: ", sprintf("%.4f", current_val_accuracy), "\n")
    
    result_row <- cbind(current_params, val_accuracy = current_val_accuracy, num_actual_epochs = length(history_mlp$metrics$loss))
    tuning_results_list[[i]] <- result_row

    if (current_val_accuracy > best_val_accuracy) {
      best_val_accuracy <- current_val_accuracy
      best_params_mlp <- current_params
      best_mlp_model_history <- history_mlp
      best_epoch_model <- model_mlp
      cat("找到新的最佳MLP模型！验证集准确率: ", sprintf("%.4f", best_val_accuracy), "\n")
      save_model_tf(best_epoch_model, best_model_output_file) # 使用 Keras 3 推荐的 save_model_tf 或 save_model_keras
      cat("最佳MLP模型已保存到: '", best_model_output_file, "'\n")
    }
  } else {
     cat("警告: MLP配置 ", i, " 训练未产生有效的验证准确率或训练失败。\n")
     result_row <- cbind(current_params, val_accuracy = NA_real_, num_actual_epochs = NA_integer_) # 使用类型化的NA
     tuning_results_list[[i]] <- result_row
  }
  gc()
}

if (length(tuning_results_list) > 0) {
  tuning_results_df <- do.call(rbind, tuning_results_list)
  # 确保所有列的数据类型正确，特别是数值型参数应为数值
  for(col_name in names(best_params_mlp)) { # 使用最佳参数的列名作为参考
      if(is.numeric(best_params_mlp[[col_name]])) {
          tuning_results_df[[col_name]] <- as.numeric(tuning_results_df[[col_name]])
      } else if(is.integer(best_params_mlp[[col_name]])) {
           tuning_results_df[[col_name]] <- as.integer(tuning_results_df[[col_name]])
      }
  }
  tuning_results_df$val_accuracy <- as.numeric(tuning_results_df$val_accuracy)
  tuning_results_df$num_actual_epochs <- as.integer(tuning_results_df$num_actual_epochs)

  write.csv(tuning_results_df, parameter_tuning_results_file, row.names = FALSE)
  cat("\nMLP参数调优结果已保存到: '", parameter_tuning_results_file, "'\n")
} else {
  cat("\nMLP参数调优未产生任何结果。\n")
}

if (is.null(best_params_mlp) || is.null(best_epoch_model)) {
  stop("错误: MLP参数搜索未能找到最佳模型。请检查训练过程或参数网格。")
}

cat("\n--- MLP参数搜索完成 ---")
cat("\n最佳MLP验证集准确率: ", sprintf("%.4f", best_val_accuracy), "\n")
cat("最佳MLP参数:\n")
print(best_params_mlp)

cat("\n加载最佳MLP模型进行最终评估...\n")
best_model_mlp <- load_model_tf(best_model_output_file)


# --- 5. 绘制最佳模型的训练历史 ---
cat("\n步骤5: 绘制最佳MLP模型的训练历史曲线图...\n")
if (!is.null(best_mlp_model_history) && !is.null(best_mlp_model_history$metrics$loss)) {
  df_history <- as.data.frame(best_mlp_model_history$metrics)
  # Keras history$metrics 已经是包含 epoch 的数据框
  # 如果没有 epoch 列，可以手动添加
  if (!"epoch" %in% names(df_history)) {
    df_history$epoch <- 1:nrow(df_history)
  }

  mlp_training_plot <- ggplot(df_history, aes(x = epoch)) +
    geom_line(aes(y = loss, colour = "训练损失")) +
    geom_line(aes(y = val_loss, colour = "验证损失")) +
    geom_line(aes(y = accuracy, colour = "训练准确率")) +
    geom_line(aes(y = val_accuracy, colour = "验证准确率")) +
    scale_colour_manual("",
      breaks = c("训练损失", "验证损失", "训练准确率", "验证准确率"),
      values = c("blue", "red", "green", "orange")) +
    labs(x = "周期 (Epoch)", y = "值", title = "最佳MLP模型训练历史 (亚型分类)") +
    theme_minimal() + theme(legend.position = "top")

  tryCatch({
    ggsave(history_plot_file, plot = mlp_training_plot, width = 10, height = 6, dpi = 300)
    cat("最佳MLP训练历史图已保存到: '", history_plot_file, "'\n")
    print(mlp_training_plot)
  }, error = function(e){cat("错误: 保存MLP训练历史图失败: ", conditionMessage(e), "\n")})
} else {
  cat("警告: 最佳MLP模型的训练历史数据不完整或不存在，无法绘图。\n")
}

# --- 6. 在测试集上评估最佳MLP模型 ---
cat("\n步骤6: 在测试集上评估最佳MLP模型性能...\n")
evaluation_mlp_list <- best_model_mlp %>% evaluate(x_test_scaled, y_test_one_hot, verbose = 0)
# evaluate 返回的是一个列表，第一个是loss，第二个是accuracy
test_loss_mlp <- evaluation_mlp_list[[1]]
test_accuracy_mlp <- evaluation_mlp_list[[2]]

cat("最佳MLP模型在测试集上的评估结果:\n")
cat("  损失 (Loss): ", sprintf("%.4f", test_loss_mlp), "\n")
cat("  准确率 (Accuracy): ", sprintf("%.4f", test_accuracy_mlp), "\n")

predictions_test_prob_mlp <- best_model_mlp %>% predict(x_test_scaled)
predictions_test_classes_numeric_mlp <- apply(predictions_test_prob_mlp, 1, which.max) - 1L
predictions_test_labels_mlp <- factor(common_levels[predictions_test_classes_numeric_mlp + 1L], levels = common_levels)

if (!is.factor(y_test_labels)) y_test_labels <- factor(y_test_labels, levels = common_levels)

conf_matrix_test_mlp <- NULL
if (length(y_test_labels) == length(predictions_test_labels_mlp) && length(y_test_labels) > 0) {
  tryCatch({
    conf_matrix_test_mlp <- confusionMatrix(data = predictions_test_labels_mlp, reference = y_test_labels)
    cat("\n最佳MLP模型混淆矩阵 (测试集):\n"); print(conf_matrix_test_mlp)

    png(confusion_matrix_plot_file, width = 800, height = 700, res = 100)
    MLmetrics::plot_ConfusionMatrix(conf_matrix_test_mlp$table,
                                    add_sums = TRUE,
                                    main = "测试集混淆矩阵 (最佳MLP - 亚型分类)")
    dev.off()
    cat("MLP混淆矩阵图已保存到: '", confusion_matrix_plot_file, "'\n")
  }, error = function(e){
    cat("错误: 生成或绘制MLP测试集混淆矩阵失败: ", conditionMessage(e), "\n")
  })
} else {
   cat("警告: MLP测试集真实标签或预测标签长度不匹配或为空。\n")
}

# --- 7. 保存性能指标和最佳MLP模型 ---
cat("\n步骤7: 保存最佳MLP模型和性能指标...\n")
sink(performance_metrics_file)
cat("--- 最佳MLP模型性能评估 (测试集 - 癌症亚型多分类) ---\n\n")
cat("最佳参数组合:\n"); print(best_params_mlp)
cat("\n模型结构摘要:\n")
capture.output(summary(best_model_mlp), file = performance_metrics_file, append = TRUE)
cat("\n训练参数详情 (最佳模型):\n")
if(!is.null(best_params_mlp)){
  cat("  Units Layer 1: ", best_params_mlp$units1, "\n")
  cat("  Units Layer 2: ", best_params_mlp$units2, "\n")
  cat("  Dropout Rate 1: ", best_params_mlp$dropout_rate1, "\n")
  cat("  Dropout Rate 2: ", best_params_mlp$dropout_rate2, "\n")
  cat("  Learning Rate: ", best_params_mlp$learning_rate, "\n")
  cat("  Batch Size: ", best_params_mlp$batch_size, "\n")
  cat("  Target Epochs: ", best_params_mlp$epochs, "\n")
}
if (!is.null(best_mlp_model_history)) {
  cat("  Actual Epochs (best model): ", length(best_mlp_model_history$metrics$loss), "\n\n")
} else {
  cat("  Actual Epochs (best model): 未知 (历史记录缺失)\n\n")
}
cat("使用的特征数量: ", input_shape, "\n")
cat("亚型类别: ", paste(common_levels, collapse=", "), "\n\n")
cat("测试集损失: ", sprintf("%.4f", test_loss_mlp), "\n")
cat("测试集准确率: ", sprintf("%.4f", test_accuracy_mlp), "\n\n")
if (!is.null(conf_matrix_test_mlp)) {
  cat("测试集混淆矩阵及详细统计:\n"); print(conf_matrix_test_mlp)
} else {
  cat("测试集混淆矩阵未能生成。\n")
}
sink()
cat(paste("\n最佳MLP性能评估指标已保存到: '", performance_metrics_file, "'\n"))

# 最终保存最佳模型 (如果之前是临时保存，这里会覆盖)
save_model_tf(best_epoch_model, best_model_output_file)
cat("最终最佳MLP模型已保存到: '", best_model_output_file, "'\n")

cat("\n--- MLP模型参数搜索、训练与评估完成！ ---\n")