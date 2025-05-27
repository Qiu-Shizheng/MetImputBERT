import os
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from .model import MetaboliteBERTModel

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class Imputer:
    def __init__(self, model_type="168", weight_path=None):
        self.model_type = model_type
        if weight_path is None:
            # Construct weight path from current working directory first.
            local_path = os.path.join(os.getcwd(), "metimputbert", "weights", f"{model_type}_model.pt")
            # Then fall back to the package directory.
            package_path = os.path.join(os.path.dirname(__file__), "weights", f"{model_type}_model.pt")
            if os.path.exists(local_path):
                weight_path = local_path
                logger.info(f"Using local weight file: {weight_path}")
            else:
                weight_path = package_path
                logger.info(f"Using packaged weight file: {weight_path}")
        self.weight_path = weight_path

        if self.model_type == "168":
            # 原本自定义的168个特征列表
            self.columns = [
                "Total Cholesterol",
                "Total Cholesterol Minus HDL-C",
                "Remnant Cholesterol (Non-HDL, Non-LDL -Cholesterol)",
                "VLDL Cholesterol",
                "Clinical LDL Cholesterol",
                "LDL Cholesterol",
                "HDL Cholesterol",
                "Total Triglycerides",
                "Triglycerides in VLDL",
                "Triglycerides in LDL",
                "Triglycerides in HDL",
                "Total Phospholipids in Lipoprotein Particles",
                "Phospholipids in VLDL",
                "Phospholipids in LDL",
                "Phospholipids in HDL",
                "Total Esterified Cholesterol",
                "Cholesteryl Esters in VLDL",
                "Cholesteryl Esters in LDL",
                "Cholesteryl Esters in HDL",
                "Total Free Cholesterol",
                "Free Cholesterol in VLDL",
                "Free Cholesterol in LDL",
                "Free Cholesterol in HDL",
                "Total Lipids in Lipoprotein Particles",
                "Total Lipids in VLDL",
                "Total Lipids in LDL",
                "Total Lipids in HDL",
                "Total Concentration of Lipoprotein Particles",
                "Concentration of VLDL Particles",
                "Concentration of LDL Particles",
                "Concentration of HDL Particles",
                "Average Diameter for VLDL Particles",
                "Average Diameter for LDL Particles",
                "Average Diameter for HDL Particles",
                "Phosphoglycerides",
                "Total Cholines",
                "Phosphatidylcholines",
                "Sphingomyelins",
                "Apolipoprotein B",
                "Apolipoprotein A1",
                "Total Fatty Acids",
                "Degree of Unsaturation",
                "Omega-3 Fatty Acids",
                "Omega-6 Fatty Acids",
                "Polyunsaturated Fatty Acids",
                "Monounsaturated Fatty Acids",
                "Saturated Fatty Acids",
                "Linoleic Acid",
                "Docosahexaenoic Acid",
                "Alanine",
                "Glutamine",
                "Glycine",
                "Histidine",
                "Total Concentration of Branched-Chain Amino Acids (Leucine + Isoleucine + Valine)",
                "Isoleucine",
                "Leucine",
                "Valine",
                "Phenylalanine",
                "Tyrosine",
                "Glucose",
                "Lactate",
                "Pyruvate",
                "Citrate",
                "3-Hydroxybutyrate",
                "Acetate",
                "Acetoacetate",
                "Acetone",
                "Creatinine",
                "Albumin",
                "Glycoprotein Acetyls",
                "Concentration of Chylomicrons and Extremely Large VLDL Particles",
                "Total Lipids in Chylomicrons and Extremely Large VLDL",
                "Phospholipids in Chylomicrons and Extremely Large VLDL",
                "Cholesterol in Chylomicrons and Extremely Large VLDL",
                "Cholesteryl Esters in Chylomicrons and Extremely Large VLDL",
                "Free Cholesterol in Chylomicrons and Extremely Large VLDL",
                "Triglycerides in Chylomicrons and Extremely Large VLDL",
                "Concentration of Very Large VLDL Particles",
                "Total Lipids in Very Large VLDL",
                "Phospholipids in Very Large VLDL",
                "Cholesterol in Very Large VLDL",
                "Cholesteryl Esters in Very Large VLDL",
                "Free Cholesterol in Very Large VLDL",
                "Triglycerides in Very Large VLDL",
                "Concentration of Large VLDL Particles",
                "Total Lipids in Large VLDL",
                "Phospholipids in Large VLDL",
                "Cholesterol in Large VLDL",
                "Cholesteryl Esters in Large VLDL",
                "Free Cholesterol in Large VLDL",
                "Triglycerides in Large VLDL",
                "Concentration of Medium VLDL Particles",
                "Total Lipids in Medium VLDL",
                "Phospholipids in Medium VLDL",
                "Cholesterol in Medium VLDL",
                "Cholesteryl Esters in Medium VLDL",
                "Free Cholesterol in Medium VLDL",
                "Triglycerides in Medium VLDL",
                "Concentration of Small VLDL Particles",
                "Total Lipids in Small VLDL",
                "Phospholipids in Small VLDL",
                "Cholesterol in Small VLDL",
                "Cholesteryl Esters in Small VLDL",
                "Free Cholesterol in Small VLDL",
                "Triglycerides in Small VLDL",
                "Concentration of Very Small VLDL Particles",
                "Total Lipids in Very Small VLDL",
                "Phospholipids in Very Small VLDL",
                "Cholesterol in Very Small VLDL",
                "Cholesteryl Esters in Very Small VLDL",
                "Free Cholesterol in Very Small VLDL",
                "Triglycerides in Very Small VLDL",
                "Concentration of IDL Particles",
                "Total Lipids in IDL",
                "Phospholipids in IDL",
                "Cholesterol in IDL",
                "Cholesteryl Esters in IDL",
                "Free Cholesterol in IDL",
                "Triglycerides in IDL",
                "Concentration of Large LDL Particles",
                "Total Lipids in Large LDL",
                "Phospholipids in Large LDL",
                "Cholesterol in Large LDL",
                "Cholesteryl Esters in Large LDL",
                "Free Cholesterol in Large LDL",
                "Triglycerides in Large LDL",
                "Concentration of Medium LDL Particles",
                "Total Lipids in Medium LDL",
                "Phospholipids in Medium LDL",
                "Cholesterol in Medium LDL",
                "Cholesteryl Esters in Medium LDL",
                "Free Cholesterol in Medium LDL",
                "Triglycerides in Medium LDL",
                "Concentration of Small LDL Particles",
                "Total Lipids in Small LDL",
                "Phospholipids in Small LDL",
                "Cholesterol in Small LDL",
                "Cholesteryl Esters in Small LDL",
                "Free Cholesterol in Small LDL",
                "Triglycerides in Small LDL",
                "Concentration of Very Large HDL Particles",
                "Total Lipids in Very Large HDL",
                "Phospholipids in Very Large HDL",
                "Cholesterol in Very Large HDL",
                "Cholesteryl Esters in Very Large HDL",
                "Free Cholesterol in Very Large HDL",
                "Triglycerides in Very Large HDL",
                "Concentration of Large HDL Particles",
                "Total Lipids in Large HDL",
                "Phospholipids in Large HDL",
                "Cholesterol in Large HDL",
                "Cholesteryl Esters in Large HDL",
                "Free Cholesterol in Large HDL",
                "Triglycerides in Large HDL",
                "Concentration of Medium HDL Particles",
                "Total Lipids in Medium HDL",
                "Phospholipids in Medium HDL",
                "Cholesterol in Medium HDL",
                "Cholesteryl Esters in Medium HDL",
                "Free Cholesterol in Medium HDL",
                "Triglycerides in Medium HDL",
                "Concentration of Small HDL Particles",
                "Total Lipids in Small HDL",
                "Phospholipids in Small HDL",
                "Cholesterol in Small HDL",
                "Cholesteryl Esters in Small HDL",
                "Free Cholesterol in Small HDL",
                "Triglycerides in Small HDL"
            ]
        elif self.model_type == "249":
            self.columns = [
                "Total Cholesterol",
                "Total Cholesterol Minus HDL-C",
                "Remnant Cholesterol (Non-HDL, Non-LDL -Cholesterol)",
                "VLDL Cholesterol",
                "Clinical LDL Cholesterol",
                "LDL Cholesterol",
                "HDL Cholesterol",
                "Total Triglycerides",
                "Triglycerides in VLDL",
                "Triglycerides in LDL",
                "Triglycerides in HDL",
                "Total Phospholipids in Lipoprotein Particles",
                "Phospholipids in VLDL",
                "Phospholipids in LDL",
                "Phospholipids in HDL",
                "Total Esterified Cholesterol",
                "Cholesteryl Esters in VLDL",
                "Cholesteryl Esters in LDL",
                "Cholesteryl Esters in HDL",
                "Total Free Cholesterol",
                "Free Cholesterol in VLDL",
                "Free Cholesterol in LDL",
                "Free Cholesterol in HDL",
                "Total Lipids in Lipoprotein Particles",
                "Total Lipids in VLDL",
                "Total Lipids in LDL",
                "Total Lipids in HDL",
                "Total Concentration of Lipoprotein Particles",
                "Concentration of VLDL Particles",
                "Concentration of LDL Particles",
                "Concentration of HDL Particles",
                "Average Diameter for VLDL Particles",
                "Average Diameter for LDL Particles",
                "Average Diameter for HDL Particles",
                "Phosphoglycerides",
                "Triglycerides to Phosphoglycerides ratio",
                "Total Cholines",
                "Phosphatidylcholines",
                "Sphingomyelins",
                "Apolipoprotein B",
                "Apolipoprotein A1",
                "Apolipoprotein B to Apolipoprotein A1 ratio",
                "Total Fatty Acids",
                "Degree of Unsaturation",
                "Omega-3 Fatty Acids",
                "Omega-6 Fatty Acids",
                "Polyunsaturated Fatty Acids",
                "Monounsaturated Fatty Acids",
                "Saturated Fatty Acids",
                "Linoleic Acid",
                "Docosahexaenoic Acid",
                "Omega-3 Fatty Acids to Total Fatty Acids percentage",
                "Omega-6 Fatty Acids to Total Fatty Acids percentage",
                "Polyunsaturated Fatty Acids to Total Fatty Acids percentage",
                "Monounsaturated Fatty Acids to Total Fatty Acids percentage",
                "Saturated Fatty Acids to Total Fatty Acids percentage",
                "Linoleic Acid to Total Fatty Acids percentage",
                "Docosahexaenoic Acid to Total Fatty Acids percentage",
                "Polyunsaturated Fatty Acids to Monounsaturated Fatty Acids ratio",
                "Omega-6 Fatty Acids to Omega-3 Fatty Acids ratio",
                "Alanine",
                "Glutamine",
                "Glycine",
                "Histidine",
                "Total Concentration of Branched-Chain Amino Acids (Leucine + Isoleucine + Valine)",
                "Isoleucine",
                "Leucine",
                "Valine",
                "Phenylalanine",
                "Tyrosine",
                "Glucose",
                "Lactate",
                "Pyruvate",
                "Citrate",
                "3-Hydroxybutyrate",
                "Acetate",
                "Acetoacetate",
                "Acetone",
                "Creatinine",
                "Albumin",
                "Glycoprotein Acetyls",
                "Concentration of Chylomicrons and Extremely Large VLDL Particles",
                "Total Lipids in Chylomicrons and Extremely Large VLDL",
                "Phospholipids in Chylomicrons and Extremely Large VLDL",
                "Cholesterol in Chylomicrons and Extremely Large VLDL",
                "Cholesteryl Esters in Chylomicrons and Extremely Large VLDL",
                "Free Cholesterol in Chylomicrons and Extremely Large VLDL",
                "Triglycerides in Chylomicrons and Extremely Large VLDL",
                "Concentration of Very Large VLDL Particles",
                "Total Lipids in Very Large VLDL",
                "Phospholipids in Very Large VLDL",
                "Cholesterol in Very Large VLDL",
                "Cholesteryl Esters in Very Large VLDL",
                "Free Cholesterol in Very Large VLDL",
                "Triglycerides in Very Large VLDL",
                "Concentration of Large VLDL Particles",
                "Total Lipids in Large VLDL",
                "Phospholipids in Large VLDL",
                "Cholesterol in Large VLDL",
                "Cholesteryl Esters in Large VLDL",
                "Free Cholesterol in Large VLDL",
                "Triglycerides in Large VLDL",
                "Concentration of Medium VLDL Particles",
                "Total Lipids in Medium VLDL",
                "Phospholipids in Medium VLDL",
                "Cholesterol in Medium VLDL",
                "Cholesteryl Esters in Medium VLDL",
                "Free Cholesterol in Medium VLDL",
                "Triglycerides in Medium VLDL",
                "Concentration of Small VLDL Particles",
                "Total Lipids in Small VLDL",
                "Phospholipids in Small VLDL",
                "Cholesterol in Small VLDL",
                "Cholesteryl Esters in Small VLDL",
                "Free Cholesterol in Small VLDL",
                "Triglycerides in Small VLDL",
                "Concentration of Very Small VLDL Particles",
                "Total Lipids in Very Small VLDL",
                "Phospholipids in Very Small VLDL",
                "Cholesterol in Very Small VLDL",
                "Cholesteryl Esters in Very Small VLDL",
                "Free Cholesterol in Very Small VLDL",
                "Triglycerides in Very Small VLDL",
                "Concentration of IDL Particles",
                "Total Lipids in IDL",
                "Phospholipids in IDL",
                "Cholesterol in IDL",
                "Cholesteryl Esters in IDL",
                "Free Cholesterol in IDL",
                "Triglycerides in IDL",
                "Concentration of Large LDL Particles",
                "Total Lipids in Large LDL",
                "Phospholipids in Large LDL",
                "Cholesterol in Large LDL",
                "Cholesteryl Esters in Large LDL",
                "Free Cholesterol in Large LDL",
                "Triglycerides in Large LDL",
                "Concentration of Medium LDL Particles",
                "Total Lipids in Medium LDL",
                "Phospholipids in Medium LDL",
                "Cholesterol in Medium LDL",
                "Cholesteryl Esters in Medium LDL",
                "Free Cholesterol in Medium LDL",
                "Triglycerides in Medium LDL",
                "Concentration of Small LDL Particles",
                "Total Lipids in Small LDL",
                "Phospholipids in Small LDL",
                "Cholesterol in Small LDL",
                "Cholesteryl Esters in Small LDL",
                "Free Cholesterol in Small LDL",
                "Triglycerides in Small LDL",
                "Concentration of Very Large HDL Particles",
                "Total Lipids in Very Large HDL",
                "Phospholipids in Very Large HDL",
                "Cholesterol in Very Large HDL",
                "Cholesteryl Esters in Very Large HDL",
                "Free Cholesterol in Very Large HDL",
                "Triglycerides in Very Large HDL",
                "Concentration of Large HDL Particles",
                "Total Lipids in Large HDL",
                "Phospholipids in Large HDL",
                "Cholesterol in Large HDL",
                "Cholesteryl Esters in Large HDL",
                "Free Cholesterol in Large HDL",
                "Triglycerides in Large HDL",
                "Concentration of Medium HDL Particles",
                "Total Lipids in Medium HDL",
                "Phospholipids in Medium HDL",
                "Cholesterol in Medium HDL",
                "Cholesteryl Esters in Medium HDL",
                "Free Cholesterol in Medium HDL",
                "Triglycerides in Medium HDL",
                "Concentration of Small HDL Particles",
                "Total Lipids in Small HDL",
                "Phospholipids in Small HDL",
                "Cholesterol in Small HDL",
                "Cholesteryl Esters in Small HDL",
                "Free Cholesterol in Small HDL",
                "Triglycerides in Small HDL",
                "Phospholipids to Total Lipids in Chylomicrons and Extremely Large VLDL percentage",
                "Cholesterol to Total Lipids in Chylomicrons and Extremely Large VLDL percentage",
                "Cholesteryl Esters to Total Lipids in Chylomicrons and Extremely Large VLDL percentage",
                "Free Cholesterol to Total Lipids in Chylomicrons and Extremely Large VLDL percentage",
                "Triglycerides to Total Lipids in Chylomicrons and Extremely Large VLDL percentage",
                "Phospholipids to Total Lipids in Very Large VLDL percentage",
                "Cholesterol to Total Lipids in Very Large VLDL percentage",
                "Cholesteryl Esters to Total Lipids in Very Large VLDL percentage",
                "Free Cholesterol to Total Lipids in Very Large VLDL percentage",
                "Triglycerides to Total Lipids in Very Large VLDL percentage",
                "Phospholipids to Total Lipids in Large VLDL percentage",
                "Cholesterol to Total Lipids in Large VLDL percentage",
                "Cholesteryl Esters to Total Lipids in Large VLDL percentage",
                "Free Cholesterol to Total Lipids in Large VLDL percentage",
                "Triglycerides to Total Lipids in Large VLDL percentage",
                "Phospholipids to Total Lipids in Medium VLDL percentage",
                "Cholesterol to Total Lipids in Medium VLDL percentage",
                "Cholesteryl Esters to Total Lipids in Medium VLDL percentage",
                "Free Cholesterol to Total Lipids in Medium VLDL percentage",
                "Triglycerides to Total Lipids in Medium VLDL percentage",
                "Phospholipids to Total Lipids in Small VLDL percentage",
                "Cholesterol to Total Lipids in Small VLDL percentage",
                "Cholesteryl Esters to Total Lipids in Small VLDL percentage",
                "Free Cholesterol to Total Lipids in Small VLDL percentage",
                "Triglycerides to Total Lipids in Small VLDL percentage",
                "Phospholipids to Total Lipids in Very Small VLDL percentage",
                "Cholesterol to Total Lipids in Very Small VLDL percentage",
                "Cholesteryl Esters to Total Lipids in Very Small VLDL percentage",
                "Free Cholesterol to Total Lipids in Very Small VLDL percentage",
                "Triglycerides to Total Lipids in Very Small VLDL percentage",
                "Phospholipids to Total Lipids in IDL percentage",
                "Cholesterol to Total Lipids in IDL percentage",
                "Cholesteryl Esters to Total Lipids in IDL percentage",
                "Free Cholesterol to Total Lipids in IDL percentage",
                "Triglycerides to Total Lipids in IDL percentage",
                "Phospholipids to Total Lipids in Large LDL percentage",
                "Cholesterol to Total Lipids in Large LDL percentage",
                "Cholesteryl Esters to Total Lipids in Large LDL percentage",
                "Free Cholesterol to Total Lipids in Large LDL percentage",
                "Triglycerides to Total Lipids in Large LDL percentage",
                "Phospholipids to Total Lipids in Medium LDL percentage",
                "Cholesterol to Total Lipids in Medium LDL percentage",
                "Cholesteryl Esters to Total Lipids in Medium LDL percentage",
                "Free Cholesterol to Total Lipids in Medium LDL percentage",
                "Triglycerides to Total Lipids in Medium LDL percentage",
                "Phospholipids to Total Lipids in Small LDL percentage",
                "Cholesterol to Total Lipids in Small LDL percentage",
                "Cholesteryl Esters to Total Lipids in Small LDL percentage",
                "Free Cholesterol to Total Lipids in Small LDL percentage",
                "Triglycerides to Total Lipids in Small LDL percentage",
                "Phospholipids to Total Lipids in Very Large HDL percentage",
                "Cholesterol to Total Lipids in Very Large HDL percentage",
                "Cholesteryl Esters to Total Lipids in Very Large HDL percentage",
                "Free Cholesterol to Total Lipids in Very Large HDL percentage",
                "Triglycerides to Total Lipids in Very Large HDL percentage",
                "Phospholipids to Total Lipids in Large HDL percentage",
                "Cholesterol to Total Lipids in Large HDL percentage",
                "Cholesteryl Esters to Total Lipids in Large HDL percentage",
                "Free Cholesterol to Total Lipids in Large HDL percentage",
                "Triglycerides to Total Lipids in Large HDL percentage",
                "Phospholipids to Total Lipids in Medium HDL percentage",
                "Cholesterol to Total Lipids in Medium HDL percentage",
                "Cholesteryl Esters to Total Lipids in Medium HDL percentage",
                "Free Cholesterol to Total Lipids in Medium HDL percentage",
                "Triglycerides to Total Lipids in Medium HDL percentage",
                "Phospholipids to Total Lipids in Small HDL percentage",
                "Cholesterol to Total Lipids in Small HDL percentage",
                "Cholesteryl Esters to Total Lipids in Small HDL percentage",
                "Free Cholesterol to Total Lipids in Small HDL percentage",
                "Triglycerides to Total Lipids in Small HDL percentage"
            ]
        else:
            raise ValueError("Invalid model type. Choose '168' or '249'.")

        # 当初 intended 特征数量（例如 168）和
        # 由于训练时错误地将 eid 列也纳入，模型实际输入维度应比 intended 多 1。
        self.num_features = len(self.columns)  # 如 168
        self.input_dim = self.num_features + 1  # 实际输入维度，169

        self.hidden_size = 768
        self.num_layers = 12
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 根据实际输入维度构造模型
        self.model = MetaboliteBERTModel(self.hidden_size, self.num_layers, self.input_dim)
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Pretrained weights not found: {self.weight_path}")
        self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded pretrained model from {self.weight_path} on {self.device}")

    def impute(self, input_csv, batch_size=96):
        logger.info(f"Loading data from {input_csv}")
        df = pd.read_csv(input_csv)
        # 输入 CSV 文件应包含：第一列 eid，后续 self.num_features 列（如 168 个特征）
        if df.shape[1] != self.num_features + 1:
            raise ValueError(
                f"Expected {self.num_features + 1} columns in CSV (1 id + {self.num_features} features), got {df.shape[1]}")
        csv_eid = df.iloc[:, 0].copy()  # 标识符，不作为模型输入
        features_df = df.iloc[:, 1:].copy()  # 原始特征：形状 (n, 168)

        # 添加虚拟列，全为 0，以匹配训练时包含的多余 eid 列
        dummy_column = np.zeros((features_df.shape[0], 1), dtype=np.float32)
        # 构造模型输入数据：将 dummy_column 置于前端，后跟原始特征
        X = np.concatenate([dummy_column, features_df.values.astype(np.float32)], axis=1)  # shape: (n, input_dim)

        # 计算归一化参数
        means_arr = np.nanmean(X, axis=0)
        stds_arr = np.nanstd(X, axis=0)
        # 对虚拟列（第 0 列）手动设置均值和标准差，确保全为 0
        means_arr[0] = 0.0
        stds_arr[0] = 1.0
        stds_arr[stds_arr == 0] = 1.0

        logger.info("Normalizing data")
        norm_data = (X - means_arr) / stds_arr
        norm_data[np.isnan(norm_data)] = 0.0

        tensor_input = torch.tensor(norm_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_input)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger.info("Performing imputation")
        imputed_preds = []
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0].to(self.device)
                attn_mask = torch.ones(batch_data.size(), dtype=torch.long, device=self.device)
                preds = self.model(batch_data, attn_mask)
                imputed_preds.append(preds.detach().cpu().numpy())
        preds_all = np.concatenate(imputed_preds, axis=0)  # shape: (n, input_dim)

        # 将归一化数据反归一化
        imputed_all = preds_all * stds_arr + means_arr
        # 舍弃第 0 列（即虚拟 eid 列），保留后面的 self.num_features 列
        imputed_features = imputed_all[:, 1:]
        logger.info("De-normalizing data")

        # 构造最终输出：第一列为原始 CSV 中的 eid，后续为插补后的特征（列名为 self.columns）
        imputed_df = pd.DataFrame(imputed_features, columns=self.columns)
        imputed_df.insert(0, "eid", csv_eid)
        logger.info("Imputation complete")
        return imputed_df