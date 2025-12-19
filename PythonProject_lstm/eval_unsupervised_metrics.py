# eval_unsupervised_metrics.py
import argparse
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=str, default="anomaly_log_results.csv",
                    help="LSTM AE 결과 파일(recon_error 포함)")
    args = ap.parse_args()

    df = pd.read_csv(args.res)

    if "recon_error" not in df.columns:
        raise ValueError("recon_error 컬럼이 없습니다.")

    errs = df["recon_error"].values

    # 1) 기본 통계
    mean_err = errs.mean()
    std_err = errs.std()
    q1, q3 = np.percentile(errs, [25, 75])
    iqr = q3 - q1

    # 2) Tail 비율
    top1 = np.mean(errs >= np.quantile(errs, 0.99))
    top5 = np.mean(errs >= np.quantile(errs, 0.95))

    # 3) 분포 특성
    skewness = skew(errs)
    kurt = kurtosis(errs)

    # 4) 시간 안정성 하락 구간 비율 (갑작스러운 스파이크 탐지)
    diffs = np.abs(np.diff(errs))
    instability = np.mean(diffs > np.percentile(diffs, 95))

    print("\n=== Unsupervised Metrics (No Fault Labels Needed) ===")
    print(f"Reconstruction Error Mean     : {mean_err:.6f}")
    print(f"Reconstruction Error STD      : {std_err:.6f}")
    print(f"IQR (Q3-Q1)                   : {iqr:.6f}")
    print(f"Top 1% Tail Ratio             : {top1:.4f}")
    print(f"Top 5% Tail Ratio             : {top5:.4f}")
    print(f"Skewness (right-tail)         : {skewness:.4f}")
    print(f"Kurtosis (peakiness)          : {kurt:.4f}")
    print(f"Temporal Instability Ratio    : {instability:.4f}")

    print("\nInterpretation Guide:")
    print("- Mean/STD 낮으면 좋음 (정상 패턴 학습 안정)")
    print("- IQR 좁으면 잘 학습된 상태")
    print("- Top1%/5% 비율이 너무 크면 모델이 불안정")
    print("- Skewness > 0 은 정상 (오른쪽 꼬리 있음)")
    print("- Temporal Instability 낮을수록 good")

if __name__ == "__main__":
    main()
