# 02_data_preprocessing.R (修改版，用于癌症亚型多分类)
# 功能:
# 1. 加载原始的CNV数据和临床表型数据。
# 2. 清洗和转换CNV数据。
# 3. 从临床数据中提取【肿瘤样本】及其【癌症亚型】标签。
# 4. 合并CNV数据和亚型标签。
# 5. (移除了1:1癌症vs正常平衡抽样)
# 6. 移除低方差特征。
# 7. 保存处理后的、可用于机器学习的数据集。

# --- 加载必要的R包 ---
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyverse))

# --- 参数定义与文件路径 ---
data_dir <- "data"
cnv_file_name <- "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes" # CNV数据文件名
phenotype_file_name <- "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"       # 临床表型数据文件名

cnv_file_path <- file.path(data_dir, cnv_file_name)
phenotype_file_path <- file.path(data_dir, phenotype_file_name)
output_processed_data_file <- file.path(data_dir, "processed_TCGA_BRCA_CNV_subtypes.rds") # 修改输出文件名

cat("--- 开始数据预处理 (用于癌症亚型多分类) ('02_data_preprocessing.R') ---\n")

# --- 1. 检查输入文件是否存在 --- (同前)
if (!file.exists(cnv_file_path)) {
  stop(paste("错误: CNV数据文件未找到:", cnv_file_path))
}
if (!file.exists(phenotype_file_path)) {
  stop(paste("错误: 临床表型数据文件未找到:", phenotype_file_path))
}
cat("输入文件检查通过。\n")

# --- 2. 加载CNV数据 --- (同前)
cat("步骤1: 加载CNV数据从 '", cnv_file_path, "'...\n")
cnv_data_raw <- fread(cnv_file_path, sep = "\t", header = TRUE, data.table = FALSE)
cat("CNV数据加载完成。原始维度: ", paste(dim(cnv_data_raw), collapse = " x "), "\n")
gene_ids <- cnv_data_raw[[1]]
cnv_matrix_genes_as_rows <- as.matrix(cnv_data_raw[, -1])
rownames(cnv_matrix_genes_as_rows) <- gene_ids
rm(cnv_data_raw)
cnv_data_transposed <- t(cnv_matrix_genes_as_rows)
rm(cnv_matrix_genes_as_rows)
gc()
cat("CNV数据转置完成。转置后维度: ", paste(dim(cnv_data_transposed), collapse = " x "), "\n")

# --- 3. 加载临床表型数据 --- (同前)
cat("步骤2: 加载临床表型数据从 '", phenotype_file_path, "'...\n")
phenotype_data_raw <- fread(phenotype_file_path, sep = "\t", header = TRUE, data.table = FALSE)
cat("临床表型数据加载完成。原始维度: ", paste(dim(phenotype_data_raw), collapse = " x "), "\n")

# --- 4. 从临床数据中筛选【肿瘤样本】并创建【亚型标签】 ---
cat("步骤3: 筛选肿瘤样本并根据指定列创建亚型 'label' 列...\n")
first_col_name_pheno <- names(phenotype_data_raw)[1] # 应为样本ID列
cat("临床数据中的样本ID列名被识别为: '", first_col_name_pheno, "'\n")

# 【重要】选择用于定义亚型的列名!
# 常见的有 "PAM50_mRNA_nature2012", "ER_Status_nature2012", "PR_Status_nature2012", "HER2_Final_Status_nature2012"
# 或者 "Integrated_Clusters_no_exp__nature2012"
# 您需要检查您的临床文件 'TCGA.BRCA.sampleMap_BRCA_clinicalMatrix' 包含哪些亚型相关的列。
# 这里我们假设使用 "PAM50_mRNA_nature2012" 作为示例。如果该列不存在，您需要更改它。
subtype_column_name <- "PAM50_mRNA_nature2012" # <--- 【请根据您的临床文件修改此列名!】

if (!(subtype_column_name %in% names(phenotype_data_raw))) {
  stop(paste("错误: 指定的亚型列 '", subtype_column_name, "' 在临床数据中未找到。",
             "请检查列名或选择一个临床文件中存在的亚型列。",
             "可用的列名有:\n", paste(names(phenotype_data_raw), collapse = "\n")))
}
cat("将使用临床列 '", subtype_column_name, "' 作为亚型标签。\n")

# 首先，筛选出肿瘤样本 (根据样本ID的14-15位)
phenotype_data_raw$sample_code_char <- substr(phenotype_data_raw[[first_col_name_pheno]], 14, 15)
phenotype_data_raw$sample_code_int <- suppressWarnings(as.integer(phenotype_data_raw$sample_code_char))

phenotype_tumor_samples <- phenotype_data_raw %>%
  filter(!is.na(sample_code_int) & sample_code_int >= 1 & sample_code_int <= 9) # 只选择肿瘤样本

# 然后，从这些肿瘤样本中提取亚型标签
# 将亚型列重命名为 'label'
phenotype_selected_subtypes <- phenotype_tumor_samples %>%
  select(sampleID = !!sym(first_col_name_pheno), label = !!sym(subtype_column_name)) %>%
  filter(!is.na(label) & label != "" & label != "null" & label != "NA" & !grepl(" வெளியீடு களை வெளியிட்டது", label, ignore.case = TRUE)) # 移除没有亚型信息或无效信息的样本

# 检查是否有可用的亚型数据
if (nrow(phenotype_selected_subtypes) == 0) {
  stop(paste("错误: 从肿瘤样本中未能提取到有效的亚型标签。",
             "请检查指定的亚型列 '", subtype_column_name, "' 是否包含有效数据，或筛选条件是否过于严格。", sep=""))
}

cat("筛选后具有有效亚型标签的肿瘤样本数量: ", nrow(phenotype_selected_subtypes), "\n")
cat("初步亚型标签分布情况:\n")
print(table(phenotype_selected_subtypes$label))

# 【可选】处理样本量过少的亚型：可以将它们合并到"Other"类别或移除
# 例如，移除样本数少于10的亚型
subtype_counts <- table(phenotype_selected_subtypes$label)
rare_subtypes <- names(subtype_counts[subtype_counts < 10]) # 阈值可以调整
if (length(rare_subtypes) > 0) {
  cat("\n发现样本量较少的亚型: ", paste(rare_subtypes, collapse=", "), ". 将移除这些样本。\n")
  phenotype_selected_subtypes <- phenotype_selected_subtypes %>%
    filter(!(label %in% rare_subtypes))
  cat("移除稀有亚型后，剩余样本数量: ", nrow(phenotype_selected_subtypes), "\n")
  cat("处理后亚型标签分布情况:\n")
  print(table(phenotype_selected_subtypes$label))
}

if (nrow(phenotype_selected_subtypes) < 2 || length(unique(phenotype_selected_subtypes$label)) < 2) { # 至少要有2个样本和2个不同的亚型类别
  stop("错误: 处理后，没有足够的样本或亚型类别进行多分类分析。")
}


# --- 5. 数据合并 --- (与之前类似，但现在合并的是亚型标签)
cat("步骤4: 合并CNV数据和亚型标签数据...\n")
cnv_df_for_merge <- as.data.frame(cnv_data_transposed)
cnv_df_for_merge$sampleID <- rownames(cnv_df_for_merge)
rownames(cnv_df_for_merge) <- NULL

# 注意：此时 phenotype_selected_subtypes 包含的是【肿瘤样本】的亚型信息
# inner_join 会确保只保留那些既有CNV数据又有亚型标签的肿瘤样本
merged_data_subtypes <- inner_join(phenotype_selected_subtypes, cnv_df_for_merge, by = "sampleID")

if (nrow(merged_data_subtypes) == 0) {
  stop("错误: CNV数据与亚型数据合并后无匹配肿瘤样本。")
}
cat("数据合并完成。合并后维度 (肿瘤样本 x (sampleID+label(亚型)+基因数)): ", paste(dim(merged_data_subtypes), collapse = " x "), "\n")
cat("合并后亚型标签分布:\n")
print(table(merged_data_subtypes$label))

# --- 6. 【移除】1:1癌症与正常样本抽样步骤 ---
# 这个步骤对于亚型分类不再需要。我们使用所有具有亚型标签的肿瘤样本。
cat("步骤5: (跳过癌症vs正常1:1平衡抽样步骤，因为现在是亚型分类)\n")
final_data_for_ml <- merged_data_subtypes # 直接使用合并后的数据

# --- 7. 最终数据准备与清洗 --- (与之前类似)
cat("步骤6: 最终数据准备与清洗...\n")
final_data_for_ml$label <- as.factor(final_data_for_ml$label) # 确保亚型标签是因子
final_data_for_ml <- final_data_for_ml %>% select(-sampleID)

feature_cols <- setdiff(colnames(final_data_for_ml), "label")
for(col_name in feature_cols) {
  if(!is.numeric(final_data_for_ml[[col_name]])) {
    final_data_for_ml[[col_name]] <- suppressWarnings(as.numeric(as.character(final_data_for_ml[[col_name]])))
  }
}
na_counts_in_features <- sapply(final_data_for_ml[, feature_cols, drop=FALSE], function(x) sum(is.na(x)))
if (sum(na_counts_in_features) > 0) {
  cat("警告: 特征数据中存在NA值。将尝试用0填充这些NA值。\n")
  for(col_name in feature_cols) {
    if(any(is.na(final_data_for_ml[[col_name]]))) {
      final_data_for_ml[[col_name]][is.na(final_data_for_ml[[col_name]])] <- 0
    }
  }
}

numeric_feature_columns_data <- final_data_for_ml[, feature_cols, drop = FALSE]
if(ncol(numeric_feature_columns_data) > 0){
  feature_vars <- apply(numeric_feature_columns_data, 2, var, na.rm = TRUE)
  constant_features <- names(feature_vars[feature_vars == 0 | is.na(feature_vars)])
  if (length(constant_features) > 0) {
    cat("发现并移除方差为0或NA的特征共 ", length(constant_features), " 个。\n")
    final_data_for_ml <- final_data_for_ml %>%
      select(-all_of(constant_features))
    cat("移除低方差特征后数据维度: ", paste(dim(final_data_for_ml), collapse = " x "), "\n")
  } else {
    cat("未发现方差为0的特征。\n")
  }
} else {
  cat("警告: 预处理后没有数值型特征列用于方差计算。\n")
}

if (ncol(final_data_for_ml) <= 1 || !"label" %in% colnames(final_data_for_ml) || length(levels(final_data_for_ml$label)) < 2) {
  stop("错误: 预处理后没有足够的特征列、标签列或亚型类别少于2。请检查数据和处理步骤。")
}

# --- 7.5. 新增：激进的特征预过滤 (基于高方差选择) ---
cat("\n步骤7.5: 进行激进的特征预过滤 (基于高方差选择)...\n")
# final_data_for_ml 此时的结构是: label, gene1, gene2, ...
# 确保 'label' 列存在且是第一列 (如果不是，调整select的逻辑)
if (!("label" %in% colnames(final_data_for_ml)) || names(final_data_for_ml)[1] != "label") {
  # 如果label不是第一列，需要调整后续的 select(-label) 和 c("label", ...)
  # 但基于之前的脚本，label应该是第一列
  warning("警告：'label'列可能不在预期的第一列位置，请检查后续选择逻辑。")
}

current_num_features <- ncol(final_data_for_ml) - 1 # -1 for label column
# 【可调参数】目标保留的特征数量，例如1000或2000个。
# 这个数量需要足够小以避免后续RF的栈溢出，但又不能太小以至于丢失过多信息。
TARGET_NUM_FEATURES_PREFILTER <- 24776 # 尝试保留1500个特征，您可以调整这个值

if (current_num_features > TARGET_NUM_FEATURES_PREFILTER) {
  cat("当前特征数 (", current_num_features, ") 较多，将基于方差进行预过滤，保留Top", TARGET_NUM_FEATURES_PREFILTER, "个特征。\n")
  
  # 提取所有特征列的数据 (不包括label列)
  # drop=FALSE 确保即使只有一个特征列，结果也是数据框而不是向量
  feature_data_only <- final_data_for_ml[, setdiff(names(final_data_for_ml), "label"), drop = FALSE]
  
  # 计算每个特征的方差。
  # 确保所有特征列都是数值型，并且NA已被处理 (之前的步骤应该已经完成)
  # 如果仍然担心有非数值列混入（不太可能），可以加一层检查：
  # numeric_cols_for_variance <- sapply(feature_data_only, is.numeric)
  # feature_variances <- apply(feature_data_only[, numeric_cols_for_variance, drop=FALSE], 2, var, na.rm = TRUE)
  
  # 假设所有特征列已是数值型且NA已处理为0
  if (ncol(feature_data_only) > 0) { # 确保有特征列
    feature_variances <- apply(feature_data_only, 2, var, na.rm = TRUE)
    
    # 检查是否有NA方差 (如果某列只有一个唯一值或全是NA，var可能返回NA)
    if(any(is.na(feature_variances))) {
      cat("警告: 计算方差时出现NA值，将从高方差特征选择中排除这些特征。\n")
      feature_variances <- feature_variances[!is.na(feature_variances)] # 移除NA方差
    }
    
    if(length(feature_variances) == 0) {
      stop("错误: 所有特征的方差都为NA或没有可计算方差的特征，无法进行基于方差的预过滤。")
    }
    
    # 获取方差最高的特征名
    # 对feature_variances降序排序，并取前TARGET_NUM_FEATURES_PREFILTER个
    # 确保请求的数量不超过实际可用的具有有效方差的特征数量
    num_to_select_actual_prefilter <- min(TARGET_NUM_FEATURES_PREFILTER, length(feature_variances))
    
    top_variance_features <- names(sort(feature_variances, decreasing = TRUE)[1:num_to_select_actual_prefilter])
    
    # 更新 final_data_for_ml，只保留label列和选出的高方差特征
    final_data_for_ml <- final_data_for_ml[, c("label", top_variance_features)]
    
    cat("特征预过滤完成。数据维度更新为: ", paste(dim(final_data_for_ml), collapse = " x "), "\n")
  } else {
    cat("警告：没有特征列可用于基于方差的预过滤。\n")
  }
  
} else {
  cat("当前特征数 (", current_num_features, ") 已在目标范围内 (<= ", TARGET_NUM_FEATURES_PREFILTER, ")，跳过激进预过滤步骤。\n")
}
# --- 特征预过滤结束 ---

# --- 7.6. 新增：规范化特征列名 ---
cat("\n步骤7.6: 规范化特征列名 (替换非法字符)...\n")
# final_data_for_ml 此时的结构是: label, feature1, feature2, ...
# 我们只想规范化特征列的名称，保持 'label' 列名不变。

feature_column_names <- colnames(final_data_for_ml)
# 找到非 'label' 的列（即所有特征列）
features_to_clean <- setdiff(feature_column_names, "label")

if (length(features_to_clean) > 0) {
  # 对这些特征列名应用 make.names()
  # unique=TRUE 确保如果清理后有重名，会自动添加后缀
  cleaned_feature_names <- make.names(features_to_clean, unique = TRUE)
  
  # 更新 final_data_for_ml 中的列名
  # 首先获取 'label' 列的索引
  label_col_index <- which(colnames(final_data_for_ml) == "label")
  # 获取特征列的索引
  feature_cols_indices <- which(colnames(final_data_for_ml) %in% features_to_clean)
  
  # 创建新的列名向量
  new_colnames <- colnames(final_data_for_ml)
  new_colnames[feature_cols_indices] <- cleaned_feature_names
  
  colnames(final_data_for_ml) <- new_colnames
  
  cat("特征列名已规范化。示例新列名 (前5个特征):\n")
  print(head(setdiff(new_colnames, "label")))
} else {
  cat("没有特征列需要规范化名称。\n")
}
# --- 列名规范化结束 --

# --- 8. 保存处理好的数据 ---
# 注意：之前脚本中这里可能是步骤7，如果按顺序编号，现在这个是步骤8
cat("\n步骤8: 保存【预过滤后】的处理后的亚型数据到 '", output_processed_data_file, "'...\n")
saveRDS(final_data_for_ml, file = output_processed_data_file) # 保存的是预过滤后的数据

cat("--- 数据预处理 (用于癌症亚型多分类, 含激进特征预过滤) ('02_data_preprocessing.R') 完成！ ---\n")

cat("\n最终【预过滤后】亚型数据集的前几行 (最多显示标签和前5个基因特征):\n")
print(head(final_data_for_ml[, 1:min(6, ncol(final_data_for_ml))]))
cat("\n最终【预过滤后】亚型数据集的标签分布:\n")
print(table(final_data_for_ml$label))

