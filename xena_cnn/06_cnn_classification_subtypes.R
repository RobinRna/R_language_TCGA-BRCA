# 06_cnn_classification_subtypes.R
# 功能:
# 1. 加载划分好的亚型数据集。
# 2. 准备数据 (特征缩放, CNN输入塑形, 标签one-hot编码)。
# 3. 定义CNN参数网格。
# 4. 循环遍历所有参数网格组合，训练和评估1D CNN模型。
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
  # ... (添加与MLP脚本中相同的错误提示) ...
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
    tf$config$experimental$set_memory_growth(gpu_devices[[1L]], TRUE)
    cat("已为 GPU 0 启用内存增长。\n")
  }, error = function(e) {
    cat("警告：为 GPU 设置内存增长失败: ", conditionMessage(e), "\n")
  })
} else {
  cat("\n未检测到 GPU。TensorFlow 将使用 CPU 运行。\n")
  # ... (添加与MLP脚本中相同的GPU未检测到提示) ...
}

# --- 设置全局随机种子 ---
set.seed(456) # R 全局种子 (与MLP不同以示区分)

# --- 参数定义与文件路径 ---
data_splits_dir <- "data_splits"
train_file <- file.path(data_splits_dir, "train_set_subtypes.rds")
validation_file <- file.path(data_splits_dir, "validation_set_subtypes.rds")
test_file <- file.path(data_splits_dir, "test_set_subtypes.rds")

model_output_dir <- "model_cnn"
results_output_dir <- "results_cnn"

if (!dir.exists(model_output_dir)) dir.create(model_output_dir, recursive = TRUE)
if (!dir.exists(results_output_dir)) dir.create(results_output_dir, recursive = TRUE)

best_model_output_file_cnn <- file.path(model_output_dir, "best_cnn_model_subtypes.keras")
history_plot_file_cnn <- file.path(results_output_dir, "cnn_training_history_plot_subtypes.png")
confusion_matrix_plot_file_cnn <- file.path(results_output_dir, "cnn_confusion_matrix_plot_subtypes.png")
performance_metrics_file_cnn <- file.path(results_output_dir, "cnn_performance_metrics_subtypes.txt")
parameter_tuning_results_file_cnn <- file.path(results_output_dir, "cnn_parameter_tuning_results.csv")

cat("\n--- 开始1D CNN模型参数搜索、训练与评估 (用于癌症亚型多分类) ---\n")

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

# --- 2. 数据预处理 (Keras CNN 专属) ---
cat("\n步骤2: 为Keras CNN模型准备数据...\n")
x_train_df <- train_set %>% select(-label)
y_train_labels <- train_set$label
x_val_df <- validation_set %>% select(-label)
y_val_labels <- validation_set$label
x_test_df <- test_set %>% select(-label)
y_test_labels <- test_set$label

colnames(x_train_df) <- make.names(colnames(x_train_df), unique = TRUE)
colnames(x_val_df) <- make.names(colnames(x_val_df), unique = TRUE)
colnames(x_test_df) <- make.names(colnames(x_test_df), unique = TRUE)

x_train <- as.matrix(x_train_df)
x_val <- as.matrix(x_val_df)
x_test <- as.matrix(x_test_df)

train_mean <- apply(x_train, 2, mean, na.rm = TRUE)
train_sd <- apply(x_train, 2, sd, na.rm = TRUE)
train_sd[train_sd == 0 | is.na(train_sd)] <- 1e-6

x_train_scaled <- scale(x_train, center = train_mean, scale = train_sd)
x_val_scaled <- scale(x_val, center = train_mean, scale = train_sd)
x_test_scaled <- scale(x_test, center = train_mean, scale = train_sd)
cat("特征数据已完成标准化处理。\n")

num_features <- ncol(x_train_scaled)
x_train_reshaped <- array_reshape(x_train_scaled, c(nrow(x_train_scaled), num_features, 1L)) # 通道数为整数
x_val_reshaped <- array_reshape(x_val_scaled, c(nrow(x_val_scaled), num_features, 1L))
x_test_reshaped <- array_reshape(x_test_scaled, c(nrow(x_test_scaled), num_features, 1L))
cat("特征数据已塑形为CNN输入格式 (samples, features, 1)。维度: ", dim(x_train_reshaped)[1L],"x",dim(x_train_reshaped)[2L],"x",dim(x_train_reshaped)[3L],"\n")

input_shape_cnn <- c(num_features, 1L) # 输入形状为 (序列长度, 通道数=1L)

y_train_numeric <- as.numeric(y_train_labels) - 1L
y_val_numeric <- as.numeric(y_val_labels) - 1L
y_test_numeric <- as.numeric(y_test_labels) - 1L

y_train_one_hot <- to_categorical(y_train_numeric, num_classes = as.integer(num_classes))
y_val_one_hot <- to_categorical(y_val_numeric, num_classes = as.integer(num_classes))
y_test_one_hot <- to_categorical(y_test_numeric, num_classes = as.integer(num_classes))
cat("标签数据已完成 One-Hot 编码。\n")

# --- 3. 定义1D CNN参数网格 (完整版) ---
cat("\n步骤3: 定义1D CNN参数网格 (调整后，用于快速演示)...\n")
param_grid_cnn <- expand.grid(
  filters1 = c(32L, 64L, 128L),
  kernel_size1 = c(3L, 5L, 7L),
  pool_size1 = c(2L),
  filters2 = c(0L, 64L, 128L), # 0L 表示不使用第二卷积层
  kernel_size2 = c(3L, 5L),
  dense_units = c(64L, 128L, 256L),
  dropout_cnn = c(0.2, 0.3, 0.4),
  dropout_dense = c(0.3, 0.4, 0.5),
  learning_rate = c(0.01, 0.001, 0.0001, 0.00001),
  batch_size = c(16L, 32L, 64L),
  epochs = c(30L, 50L, 100L)
)
# 筛选逻辑
param_grid_cnn <- param_grid_cnn %>%
  filter(!(filters2 == 0L & kernel_size2 != 3L)) %>% # 如果filters2=0, kernel_size2固定
  filter(ifelse(filters2 > 0L, filters2 >= filters1, TRUE)) # 确保filters2如果存在，其值不小于filters1

cat("将测试以下 ", nrow(param_grid_cnn), " 组1D CNN参数组合。\n")
cat("参数网格示例:\n")
print(head(param_grid_cnn, 5))

# --- 4. CNN参数搜索与模型训练 ---
cat("\n步骤4: 开始1D CNN参数搜索和模型训练...\n")
tuning_results_list_cnn <- list()
best_val_accuracy_cnn <- -Inf
best_params_cnn <- NULL
best_cnn_model_history <- NULL
best_epoch_model_cnn <- NULL

for (i in 1:nrow(param_grid_cnn)) {
  current_params <- param_grid_cnn[i, ]
  cat(paste0("\n--- CNN参数组合 ", i, "/", nrow(param_grid_cnn), " ---\n"))
  print(current_params)

  k_clear_session()
  tensorflow::set_random_seed(456L + i)

  model_cnn <- keras_model_sequential(name = paste0("cnn_model_config_", i)) %>%
    layer_conv_1d(filters = current_params$filters1, # 已经是整数
                  kernel_size = current_params$kernel_size1, # 已经是整数
                  activation = "relu", input_shape = input_shape_cnn, padding = "same", name="conv1d_1") %>%
    layer_batch_normalization(name="bn_conv1") %>%
    layer_max_pooling_1d(pool_size = current_params$pool_size1, name="maxpool1") %>% # 已经是整数
    layer_dropout(rate = current_params$dropout_cnn, name="dropout_conv1") # 浮点数

  if (current_params$filters2 > 0L) { # 确保是整数比较
    model_cnn <- model_cnn %>%
      layer_conv_1d(filters = current_params$filters2, # 已经是整数
                    kernel_size = current_params$kernel_size2, # 已经是整数
                    activation = "relu", padding = "same", name="conv1d_2") %>%
      layer_batch_normalization(name="bn_conv2") %>%
      layer_global_average_pooling_1d(name="globalavgpool_last") %>%
      layer_dropout(rate = current_params$dropout_cnn, name="dropout_conv2") # 浮点数
  } else {
      model_cnn <- model_cnn %>% layer_global_average_pooling_1d(name="globalavgpool_single")
  }

  model_cnn <- model_cnn %>%
    layer_dense(units = current_params$dense_units, activation = "relu", name="dense_1_cnn") %>% # 已经是整数
    layer_batch_normalization(name="bn_dense_cnn") %>%
    layer_dropout(rate = current_params$dropout_dense, name="dropout_dense_cnn") %>% # 浮点数
    layer_dense(units = as.integer(num_classes), activation = "softmax", name="output_dense_cnn")

  model_cnn %>% compile(
    optimizer = optimizer_adam(learning_rate = current_params$learning_rate), # 浮点数
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  cat("训练CNN模型 (配置 ", i, ")...\n")
  history_cnn <- NULL
  tryCatch({
    history_cnn <- model_cnn %>% fit(
      x_train_reshaped, y_train_one_hot,
      epochs = current_params$epochs, # 已经是整数
      batch_size = current_params$batch_size, # 已经是整数
      validation_data = list(x_val_reshaped, y_val_one_hot),
      callbacks = list(
        callback_early_stopping(monitor = "val_accuracy", patience = 15L, mode="max", restore_best_weights = TRUE),
        callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 7L, min_lr = 0.00001)
      ),
      verbose = 1
    )
  }, error = function(e) {
    cat("错误: 训练CNN模型 (配置 ", i, ") 失败: ", conditionMessage(e), "\n")
    try(print(summary(model_cnn)))
    history_cnn <<- NULL
  })

  if (!is.null(history_cnn) && !is.null(history_cnn$metrics$val_accuracy) && length(history_cnn$metrics$val_accuracy) > 0) {
    current_val_accuracy_cnn <- max(history_cnn$metrics$val_accuracy, na.rm = TRUE)
    cat("当前CNN配置验证集最高准确率: ", sprintf("%.4f", current_val_accuracy_cnn), "\n")
    
    result_row_cnn <- cbind(current_params, val_accuracy = current_val_accuracy_cnn, num_actual_epochs = length(history_cnn$metrics$loss))
    tuning_results_list_cnn[[i]] <- result_row_cnn

    if (current_val_accuracy_cnn > best_val_accuracy_cnn) {
      best_val_accuracy_cnn <- current_val_accuracy_cnn
      best_params_cnn <- current_params
      best_cnn_model_history <- history_cnn
      best_epoch_model_cnn <- model_cnn
      cat("找到新的最佳CNN模型！验证集准确率: ", sprintf("%.4f", best_val_accuracy_cnn), "\n")
      save_model_tf(best_epoch_model_cnn, best_model_output_file_cnn)
      cat("最佳CNN模型已保存到: '", best_model_output_file_cnn, "'\n")
    }
  } else {
     cat("警告: CNN配置 ", i, " 训练未产生有效的验证准确率或训练失败。\n")
     result_row_cnn <- cbind(current_params, val_accuracy = NA_real_, num_actual_epochs = NA_integer_)
     tuning_results_list_cnn[[i]] <- result_row_cnn
  }
  gc()
}

# ... (后续的评估、绘图和保存部分与您之前的脚本类似，确保所有参数都正确处理为整数或浮点数) ...
if (length(tuning_results_list_cnn) > 0) {
  tuning_results_df_cnn <- do.call(rbind, tuning_results_list_cnn)
   # 确保所有列的数据类型正确
  for(col_name in names(best_params_cnn)) {
      if(is.numeric(best_params_cnn[[col_name]])) { # 包括整数和浮点数
          tuning_results_df_cnn[[col_name]] <- as.numeric(tuning_results_df_cnn[[col_name]])
      }
  }
  tuning_results_df_cnn$val_accuracy <- as.numeric(tuning_results_df_cnn$val_accuracy)
  tuning_results_df_cnn$num_actual_epochs <- as.integer(tuning_results_df_cnn$num_actual_epochs)

  write.csv(tuning_results_df_cnn, parameter_tuning_results_file_cnn, row.names = FALSE)
  cat("\nCNN参数调优结果已保存到: '", parameter_tuning_results_file_cnn, "'\n")
} else {
  cat("\nCNN参数调优未产生任何结果。\n")
}


if (is.null(best_params_cnn) || is.null(best_epoch_model_cnn)) {
  stop("错误: CNN参数搜索未能找到最佳模型。请检查训练过程或参数网格。")
}

cat("\n--- CNN参数搜索完成 ---")
cat("\n最佳CNN验证集准确率: ", sprintf("%.4f", best_val_accuracy_cnn), "\n")
cat("最佳CNN参数:\n")
print(best_params_cnn)

cat("\n加载最佳CNN模型进行最终评估...\n")
best_model_cnn <- load_model_tf(best_model_output_file_cnn)

# --- 5. 绘制最佳CNN模型的训练历史 ---
cat("\n步骤5: 绘制最佳CNN模型的训练历史曲线图...\n")
if (!is.null(best_cnn_model_history) && !is.null(best_cnn_model_history$metrics$loss)) {
  df_history_cnn <- as.data.frame(best_cnn_model_history$metrics)
  if (!"epoch" %in% names(df_history_cnn)) {
    df_history_cnn$epoch <- 1:nrow(df_history_cnn)
  }
  
  cnn_training_plot <- ggplot(df_history_cnn, aes(x = epoch)) +
    geom_line(aes(y = loss, colour = "训练损失")) +
    geom_line(aes(y = val_loss, colour = "验证损失")) +
    geom_line(aes(y = accuracy, colour = "训练准确率")) +
    geom_line(aes(y = val_accuracy, colour = "验证准确率")) +
    scale_colour_manual("",
      breaks = c("训练损失", "验证损失", "训练准确率", "验证准确率"),
      values = c("blue", "red", "green", "orange")) +
    labs(x = "周期 (Epoch)", y = "值", title = "最佳1D CNN模型训练历史 (亚型分类)") +
    theme_minimal() + theme(legend.position = "top")

  tryCatch({
    ggsave(history_plot_file_cnn, plot = cnn_training_plot, width = 10, height = 6, dpi = 300)
    cat("最佳CNN训练历史图已保存到: '", history_plot_file_cnn, "'\n")
    print(cnn_training_plot)
  }, error = function(e){cat("错误: 保存CNN训练历史图失败: ", conditionMessage(e), "\n")})
} else {
  cat("警告: 最佳CNN模型的训练历史数据不完整或不存在，无法绘图。\n")
}

# --- 6. 在测试集上评估最佳CNN模型 ---
cat("\n步骤6: 在测试集上评估最佳CNN模型性能...\n")
evaluation_cnn_list <- best_model_cnn %>% evaluate(x_test_reshaped, y_test_one_hot, verbose = 0)
test_loss_cnn <- evaluation_cnn_list[[1]]
test_accuracy_cnn <- evaluation_cnn_list[[2]]

cat("最佳CNN模型在测试集上的评估结果:\n")
cat("  损失 (Loss): ", sprintf("%.4f", test_loss_cnn), "\n")
cat("  准确率 (Accuracy): ", sprintf("%.4f", test_accuracy_cnn), "\n")

predictions_test_prob_cnn <- best_model_cnn %>% predict(x_test_reshaped)
predictions_test_classes_numeric_cnn <- apply(predictions_test_prob_cnn, 1, which.max) - 1L
predictions_test_labels_cnn <- factor(common_levels[predictions_test_classes_numeric_cnn + 1L], levels = common_levels)

if (!is.factor(y_test_labels)) y_test_labels <- factor(y_test_labels, levels = common_levels)

conf_matrix_test_cnn <- NULL
if (length(y_test_labels) == length(predictions_test_labels_cnn) && length(y_test_labels) > 0) {
  tryCatch({
    conf_matrix_test_cnn <- confusionMatrix(data = predictions_test_labels_cnn, reference = y_test_labels)
    cat("\n最佳CNN模型混淆矩阵 (测试集):\n"); print(conf_matrix_test_cnn)

    png(confusion_matrix_plot_file_cnn, width = 800, height = 700, res = 100)
    MLmetrics::plot_ConfusionMatrix(conf_matrix_test_cnn$table,
                                    add_sums = TRUE,
                                    main = "测试集混淆矩阵 (最佳CNN - 亚型分类)")
    dev.off()
    cat("CNN混淆矩阵图已保存到: '", confusion_matrix_plot_file_cnn, "'\n")
  }, error = function(e){
    cat("错误: 生成或绘制CNN测试集混淆矩阵失败: ", conditionMessage(e), "\n")
  })
} else {
  cat("警告: CNN测试集真实标签或预测标签长度不匹配或为空。\n")
}

# --- 7. 保存性能指标和最佳CNN模型 ---
cat("\n步骤7: 保存最佳CNN模型和性能指标...\n")
sink(performance_metrics_file_cnn)
cat("--- 最佳1D CNN模型性能评估 (测试集 - 癌症亚型多分类) ---\n\n")
cat("最佳参数组合:\n"); print(best_params_cnn)
cat("\n模型结构摘要:\n")
capture.output(summary(best_model_cnn), file = performance_metrics_file_cnn, append = TRUE)
cat("\n训练参数详情 (最佳模型):\n")
if(!is.null(best_params_cnn)){
  cat("  Filters L1: ", best_params_cnn$filters1, ", Kernel L1: ", best_params_cnn$kernel_size1, ", Pool L1: ", best_params_cnn$pool_size1, "\n")
  if(best_params_cnn$filters2 > 0L){ # 确保整数比较
    cat("  Filters L2: ", best_params_cnn$filters2, ", Kernel L2: ", best_params_cnn$kernel_size2, "\n")
  }
  cat("  Dense Units: ", best_params_cnn$dense_units, "\n")
  cat("  Dropout CNN: ", best_params_cnn$dropout_cnn, ", Dropout Dense: ", best_params_cnn$dropout_dense, "\n")
  cat("  Learning Rate: ", best_params_cnn$learning_rate, "\n")
  cat("  Batch Size: ", best_params_cnn$batch_size, "\n")
  cat("  Target Epochs: ", best_params_cnn$epochs, "\n")
}
if (!is.null(best_cnn_model_history)) {
  cat("  Actual Epochs (best model): ", length(best_cnn_model_history$metrics$loss), "\n\n")
} else {
  cat("  Actual Epochs (best model): 未知 (历史记录缺失)\n\n")
}
cat("使用的特征数量 (序列长度): ", num_features, "\n")
cat("亚型类别: ", paste(common_levels, collapse=", "), "\n\n")
cat("测试集损失: ", sprintf("%.4f", test_loss_cnn), "\n")
cat("测试集准确率: ", sprintf("%.4f", test_accuracy_cnn), "\n\n")
if (!is.null(conf_matrix_test_cnn)) {
  cat("测试集混淆矩阵及详细统计:\n"); print(conf_matrix_test_cnn)
} else {
  cat("测试集混淆矩阵未能生成。\n")
}
sink()
cat(paste("\n最佳CNN性能评估指标已保存到: '", performance_metrics_file_cnn, "'\n"))

save_model_tf(best_epoch_model_cnn, best_model_output_file_cnn)
cat("最终最佳CNN模型已保存到: '", best_model_output_file_cnn, "'\n")

cat("\n--- 1D CNN模型参数搜索、训练与评估完成！ ---\n")
