# 01_data_download.R
# 功能: 本脚本主要用于记录和说明TCGA-BRCA CNV和临床数据的下载来源。
#       实际的数据下载通常通过浏览器或命令行工具手动完成。
#       请确保在运行后续脚本前，数据已按要求下载并放置在正确位置。

# --- 数据来源说明 ---

# 1. CNV (Copy Number Variation) 数据 - 基因水平拷贝数变异数据
#    描述: 来自TCGA乳腺癌项目(BRCA)，经过GISTIC2算法处理并阈值化的基因水平拷贝数。
#    来源: UCSC Xena Hub (tcga.xenahubs.net)
#    数据集名称: TCGA breast invasive carcinoma (BRCA) copy number gistic2 thresholded estimate
#    原始文件名 (下载时): Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz
#    实际下载链接: https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz
#    处理要求: 下载后解压。将解压得到的CNV数据文件 (其名称通常为 "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes")
#               【直接放入项目下的 "data" 文件夹中，不需要重命名】。
#    数据格式预期: 文本文件(内部为制表符分隔)，第一列为基因ID，后续列为样本ID，值为拷贝数估计值。

# 2. 临床表型数据 (Phenotype / Clinical Data)
#    描述: 包含TCGA-BRCA样本的临床信息，如样本类型、年龄、癌症分期等。
#    来源: UCSC Xena Hub (tcga.xenahubs.net)
#    数据集名称: TCGA Breast Cancer (BRCA) clinicalMatrix
#    原始文件名 (下载时): BRCA_clinicalMatrix (或类似，如 TCGA.BRCA.sampleMap_BRCA_clinicalMatrix，也可能是.gz压缩包)
#    实际下载链接: https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix
#    处理要求: 下载后（如果需要则解压）。将文件 (其名称通常为 "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix")
#               【直接放入项目下的 "data" 文件夹中，不需要重命名】。
#    数据格式预期: 文本文件(内部为制表符分隔)，第一行为临床特征的列名，第一列为样本ID。

# --- 项目文件夹结构建议 ---
# Your_Project_Folder/
# |-- data/
# |   |-- Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes  <- CNV数据 (实际名称)
# |   |-- TCGA.BRCA.sampleMap_BRCA_clinicalMatrix             <- 临床数据 (实际名称)
# |-- data_splits/
# |-- model/
# |-- results/
# |-- 01_data_download.R
# |-- 02_data_preprocessing.R
# |-- 03_data_splitting.R
# |-- 04_model_training_evaluation.R

# --- 脚本执行信息 ---
print("脚本 '01_data_download.R' 执行完毕。")
print("这是一个说明性脚本，请确保您已按照上述说明手动完成数据下载和准备工作。")
print("需要准备的文件及其在 'data' 文件夹中的【实际文件名】应为:")
print("1. CNV数据: 例如 'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes'")
print("2. 临床表型数据: 例如 'TCGA.BRCA.sampleMap_BRCA_clinicalMatrix'")
print("请确认这些文件已正确放置，并且在 '02_data_preprocessing.R' 脚本中正确指定了这些文件名。")