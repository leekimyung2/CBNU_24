# eval_anomaly_metrics.py
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

def build_labels_from_fault_times(
    df_res: pd.DataFrame,
    faults_df: pd.DataFrame,
    time_col: str,
    fault_col: str,
    pos_window_min: float,
):
    """
    fault_time 기준으로 [fault_time - pos_window_min, fault_time] 구간에 포함되는
    seq_start_time을 양성(1), 나머지를 음성(0)으로 라벨링.
    """
    df = df_res.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    faults_df[fault_col] = pd.to_datetime(faults_df[fault_col])

    df["y_true"] = 0

    pos_window = pd.to_timedelta(pos_window_min, unit="m")

    for ft in faults_df[fault_col]:
        start = ft - pos_window
        end = ft
        mask = (df[time_col] >= start) & (df[time_col] <= end)
        df.loc[mask, "y_true"] = 1

    return df


def compute_lead_time(
    df: pd.DataFrame,
    time_col: str,
    fault_times: pd.Series,
    y_pred_col: str = "y_pred",
):
    """
    Lead Time 계산:
    - 각 fault_time마다,
      fault_time 이전에 발생한 y_pred=1 중 가장 마지막 시점(=가장 최근 경고)을 찾고
      fault_time - 그 alert_time 을 lead time으로 정의.
    - 단위: 초/분 둘 다 계산해서 반환.
    """
    times = pd.to_datetime(df[time_col])
    y_pred = df[y_pred_col].values

    lead_times_sec = []

    for ft in pd.to_datetime(fault_times):
        # fault 이전의 알람(양성 예측)만 고려
        mask = (times <= ft) & (y_pred == 1)
        if not mask.any():
            # 이 고장은 탐지 못함 (lead time 없음)
            continue
        # fault 이전 알람 중 가장 최근 알람 시간
        last_alert_time = times[mask].max()
        delta = (ft - last_alert_time).total_seconds()
        if delta >= 0:
            lead_times_sec.append(delta)

    if len(lead_times_sec) == 0:
        return None, None

    arr = np.array(lead_times_sec)
    avg_sec = arr.mean()
    avg_min = avg_sec / 60.0
    return avg_sec, avg_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=str, default="anomaly_log_results.csv",
                    help="LSTM AutoEncoder 결과 파일 (seq_start_time, recon_error, anomaly_flag 포함)")
    ap.add_argument("--faults", type=str, required=True,
                    help="고장 시간 파일 (fault_time 컬럼 포함)")
    ap.add_argument("--time_col", type=str, default="seq_start_time",
                    help="결과 파일에서 시간 컬럼 이름")
    ap.add_argument("--fault_col", type=str, default="fault_time",
                    help="고장 파일에서 고장 시간 컬럼 이름")
    ap.add_argument("--pos_window_min", type=float, default=30.0,
                    help="고장 전 몇 분을 양성(고장 관련) 구간으로 볼지 (예: 30분)")
    ap.add_argument("--thr", type=float, default=None,
                    help="recon_error 임계값 (None이면 q=0.99 quantile 사용)")
    args = ap.parse_args()

    # 1) 예측 결과 읽기
    df_res = pd.read_csv(args.res)
    if args.time_col not in df_res.columns:
        raise ValueError(f"{args.res}에 {args.time_col} 컬럼이 없습니다.")
    if "recon_error" not in df_res.columns:
        raise ValueError(f"{args.res}에 'recon_error' 컬럼이 없습니다.")

    # 2) 고장 시간 읽기
    faults_df = pd.read_csv(args.faults)
    if args.fault_col not in faults_df.columns:
        raise ValueError(f"{args.faults}에 {args.fault_col} 컬럼이 없습니다.")

    # 3) 고장 전 pos_window_min 분을 양성으로 라벨링
    df = build_labels_from_fault_times(
        df_res,
        faults_df,
        time_col=args.time_col,
        fault_col=args.fault_col,
        pos_window_min=args.pos_window_min,
    )

    # y_true 준비
    y_true = df["y_true"].astype(int).values

    # 4) recon_error 기반 점수/예측 생성
    y_score = df["recon_error"].values

    if args.thr is None:
        thr = np.quantile(y_score, 0.99)
        print(f"[INFO] 임계값(thr)을 자동 설정했습니다 (q=0.99): {thr:.6f}")
    else:
        thr = args.thr
        print(f"[INFO] 사용자 지정 임계값(thr) 사용: {thr:.6f}")

    y_pred = (y_score >= thr).astype(int)
    df["y_pred"] = y_pred

    # 5) 시퀀스 단위 Confusion Matrix, Precision, Recall, F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # False Alarm Rate (여기서는 FP / (FP + TN) 정의)
    neg = fp + tn
    if neg > 0:
        false_alarm_rate = fp / neg
    else:
        false_alarm_rate = 0.0

    # 6) ROC-AUC, PR-AUC
    # y_true에 양성이 하나도 없거나 전부면 ROC-AUC 계산이 안 될 수 있음 → 예외 처리
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = None

    # 7) Lead Time 계산
    avg_lead_sec, avg_lead_min = compute_lead_time(
        df,
        time_col=args.time_col,
        fault_times=faults_df[args.fault_col],
        y_pred_col="y_pred",
    )

    # 8) 결과 출력
    print("\n=== Sequence-level Metrics ===")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score       : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC        : {roc_auc:.4f}")
    else:
        print("ROC-AUC        : 계산 불가 (양성/음성 클래스 구성이 한쪽으로 치우쳐 있음)")
    if pr_auc is not None:
        print(f"PR-AUC         : {pr_auc:.4f}")
    else:
        print("PR-AUC         : 계산 불가 (양성/음성 클래스 구성이 한쪽으로 치우쳐 있음)")

    print(f"False Alarm Rate (FP / (FP+TN)) : {false_alarm_rate:.4f}")
    print("\nConfusion Matrix [TN FP; FN TP]:")
    print(cm)

    if avg_lead_sec is not None:
        print(f"\nAverage Lead Time : {avg_lead_sec:.1f} sec  ({avg_lead_min:.2f} min)")
    else:
        print("\nAverage Lead Time : 계산 불가 (고장 이전에 탐지된 알람이 없음)")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
