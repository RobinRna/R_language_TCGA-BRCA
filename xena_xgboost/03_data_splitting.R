# 03_data_splitting.R (用于癌症亚型多分类)
# 功能:
# 1. 加载由 '02_data_preprocessing.R' (亚型版) 处理好的数据集。
# 2. 将数据集通过分层抽样的方式划分为训练集、验证集和测试集。
#    分层抽样会基于亚型标签进行。
# 3. 保存划分后的三个数据集。

suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(dplyr))

# 修改读取的文件名
processed_data_file <- file.path("data", "processed_TCGA_BRCA_CNV_subtypes.rds") # <--- 修改这里

output_dir_splits <- "data_splits"
# 输出文件名可以保持不变，或者加上_subtypes后缀
train_file_path <- file.path(output_dir_splits, "train_set_subtypes.rds") # <--- 修改这里 (可选)
validation_file_path <- file.path(output_dir_splits, "validation_set_subtypes.rds") # <--- 修改这里 (可选)
test_file_path <- file.path(output_dir_splits, "test_set_subtypes.rds") # <--- 修改这里 (可选)

cat("--- 开始数据划分 (用于癌症亚型多分类) ('03_data_splitting.R') ---\n")

if (!dir.exists(output_dir_splits)) {
  dir.create(output_dir_splits, recursive = TRUE)
  cat("已创建目录: '", output_dir_splits, "' 用于存放划分后的数据集。\n")
}

cat("步骤1: 加载处理好的亚型数据从 '", processed_data_file, "'...\n")
if (!file.exists(processed_data_file)) {
  stop(paste("错误: 处理后的亚型数据文件未找到:", processed_data_file,
             "\n请确保您已成功运行 '02_data_preprocessing.R' (亚型版) 脚本。"))
}
final_data <- readRDS(processed_data_file)
cat("亚型数据加载完成。数据集维度 (样本 x 特征+亚型标签): ", paste(dim(final_data), collapse = " x "), "\n")

if (!is.factor(final_data$label)) {
  final_data$label <- as.factor(final_data$label)
  cat("'label' (亚型) 列已确认为或转换为因子类型。\n")
}
if (length(levels(final_data$label)) < 2) {
    stop("错误: 加载的数据中亚型类别少于2，无法进行数据划分和多分类。")
}


set.seed(42)
test_indices <- createDataPartition(final_data$label, p = 0.20, list = FALSE)
test_set <- final_data[test_indices, ]
train_validation_set <- final_data[-test_indices, ]

cat("初步划分完成:\n")
cat("  测试集维度: ", paste(dim(test_set), collapse = " x "), "\n")
cat("  训练+验证集维度: ", paste(dim(train_validation_set), collapse = " x "), "\n")

validation_indices_in_tv <- createDataPartition(train_validation_set$label, p = 0.25, list = FALSE)
validation_set <- train_validation_set[validation_indices_in_tv, ]
train_set <- train_validation_set[-validation_indices_in_tv, ]

cat("最终划分完成:\n")
cat("  训练集维度: ", paste(dim(train_set), collapse = " x "), "\n")
cat("  验证集维度: ", paste(dim(validation_set), collapse = " x "), "\n")
cat("  测试集维度 (同上): ", paste(dim(test_set), collapse = " x "), "\n")

cat("\n--- 各数据集亚型标签分布比例检查 ---\n")
cat("整体亚型数据集标签分布:\n")
print(prop.table(table(final_data$label)))
cat("训练集亚型标签分布:\n")
print(prop.table(table(train_set$label)))
cat("验证集亚型标签分布:\n")
print(prop.table(table(validation_set$label)))
cat("测试集亚型标签分布:\n")
print(prop.table(table(test_set$label)))
# 这里的比例会根据各亚型的实际数量而变化。

cat("\n步骤2: 保存划分后的亚型数据集到 '", output_dir_splits, "' 文件夹...\n")
saveRDS(train_set, file = train_file_path)
cat("  训练集已保存到: '", train_file_path, "'\n")
saveRDS(validation_set, file = validation_file_path)
cat("  验证集已保存到: '", validation_file_path, "'\n")
saveRDS(test_set, file = test_file_path)
cat("  测试集已保存到: '", test_file_path, "'\n")

cat("--- 数据划分 (用于癌症亚型多分类) ('03_data_splitting.R') 完成！所有数据集已保存。 ---\n")
gc()
