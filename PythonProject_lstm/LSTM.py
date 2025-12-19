# LSTM_log_ae.py
import argparse
import os
import random
import re
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 재현성 설정
# =========================
SEED = 42
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =========================
# 2. 데이터셋 정의
# =========================
class SeqDataset(Dataset):
    def __init__(self, X3d: np.ndarray):
        # X3d: (N_seq, T, F)
        self.X3d = X3d.astype(np.float32)

    def __len__(self):
        return self.X3d.shape[0]

    def __getitem__(self, idx: int):
        x = self.X3d[idx]
        # AutoEncoder라서 input = target
        return torch.from_numpy(x), torch.from_numpy(x)

# =========================
# 3. LSTM AutoEncoder
# =========================
class LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 32,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.enc2lat = nn.Linear(hidden, latent)
        self.lat2dec = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        # x: (B, T, F)
        enc_out, (h, c) = self.encoder(x)    # enc_out: (B, T, H)
        # 시간축 평균으로 시퀀스 임베딩
        h_seq = enc_out.mean(dim=1)          # (B, H)
        z = torch.tanh(self.enc2lat(h_seq))  # (B, Z)
        h_dec = torch.tanh(self.lat2dec(z))  # (B, H)

        B, T, _ = x.shape
        dec_in = h_dec.unsqueeze(1).repeat(1, T, 1)  # (B, T, H)
        dec_out, _ = self.decoder(dec_in)            # (B, T, H)
        x_hat = self.out(dec_out)                    # (B, T, F)
        return x_hat

# =========================
# 4. 로그 파싱 유틸
# =========================
# 예: [07/21 00:00:00.602] [PF]END - X Side 2 , START- Pressor 2-1 up
LOG_PATTERN = re.compile(
    r"\[(\d{2}/\d{2}) (\d{2}:\d{2}:\d{2}\.\d{3})\]\s+\[PF\](.*)"
)

def normalize_msg(msg: str) -> str:
    """
    메시지에서 숫자(시간/Loadcell 등)를 <num>으로 치환해서
    패턴 중심으로 보도록 한다.
    """
    msg = msg.strip()
    msg = re.sub(r"\d+\.\d+|\d+", "<num>", msg)
    # 공백 정리
    msg = re.sub(r"\s+", " ", msg)
    return msg

def parse_pf_log(path: str) -> pd.DataFrame:
    """
    설비 PF 로그 텍스트 파일을 읽어서
    time, raw_msg, norm_msg 컬럼을 가진 DataFrame으로 반환
    """
    rows: List[Tuple[str, str, str]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m = LOG_PATTERN.match(line)
            if not m:
                continue
            md, hms, msg = m.groups()
            # 연도는 로그에 없으므로 dummy year(2025) 부여
            ts_str = f"2025/{md} {hms}"
            norm = normalize_msg(msg)
            rows.append((ts_str, msg.strip(), norm))

    if not rows:
        raise ValueError(f"파싱된 로그 라인이 없습니다. 패턴을 확인하세요: {path}")

    df = pd.DataFrame(rows, columns=["time", "raw_msg", "norm_msg"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

# =========================
# 5. 시퀀스 생성
# =========================
def to_sequences(X2d: np.ndarray, win: int, stride: int):
    """
    X2d: (T, F)  ->  (N_seq, win, F)
    """
    T, F = X2d.shape
    idx = list(range(0, max(1, T - win + 1), stride))
    if T >= win:
        valid_starts = [i for i in idx if i + win <= T]
        seqs = np.stack([X2d[i:i + win] for i in valid_starts], axis=0)
        starts = np.array(valid_starts, dtype=int)
    else:
        seqs = np.empty((0, win, F), dtype=np.float32)
        starts = np.empty((0,), dtype=int)
    return seqs, starts  # starts: 원본 타임스텝 인덱스

# =========================
# 6. 학습/평가 루프
# =========================
def train_epoch(model, loader, opt, device, clip: float = 1.0):
    model.train()
    total = 0.0
    crit = nn.MSELoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        yhat = model(xb)
        loss = crit(yhat, yb)
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    crit = nn.MSELoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        loss = crit(yhat, yb)
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def anomaly_score(model, X3d: np.ndarray, device):
    """
    X3d: (N_seq, T, F)
    반환: errs (N_seq, ) - 시퀀스 단위 reconstruction error
    """
    model.eval()
    ds = SeqDataset(X3d)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    errs = []
    crit = nn.MSELoss(reduction="none")
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        yhat = model(xb)
        e = crit(yhat, yb).mean(dim=(1, 2)).cpu().numpy()
        errs.append(e)
    errs = np.concatenate(errs, axis=0)
    return errs

# =========================
# 7. 메인 파이프라인
# =========================
def main():
    set_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=str, help="설비 PF 로그 텍스트 파일 경로 (예: 20250721_PeelerF.txt)")
    ap.add_argument("--win", type=int, default=64, help="시퀀스 윈도우 길이 (로그 라인 수 기준)")
    ap.add_argument("--stride", type=int, default=16, help="시퀀스 슬라이딩 간격")
    ap.add_argument("--train_ratio", type=float, default=0.6,
                    help="학습 구간 비율 (시간 순으로 앞쪽 비율)")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="검증 구간 비율 (시퀀스 기준)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--quantile", type=float, default=0.99,
                    help="이상 판별에 사용할 quantile (예: 0.99 상위 1%)")
    ap.add_argument("--out", type=str, default="anomaly_log_results.csv",
                    help="결과 저장 파일명")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {args.log}")

    print(f"[INFO] 로그 파싱 중... {args.log}")
    df = parse_pf_log(args.log)
    print(f"[INFO] 파싱된 로그 라인 수: {len(df)}")

    # -------- 이벤트 → one-hot 코딩 --------
    # 정규화된 메시지(norm_msg)를 이벤트 종류로 사용
    events = df["norm_msg"].tolist()
    vocab = sorted(set(events))
    event2id = {e: i for i, e in enumerate(vocab)}
    V = len(vocab)
    print(f"[INFO] 이벤트 종류 개수(vocab size): {V}")

    # 각 로그 라인을 (T, V) 형태의 one-hot으로 변환
    ids = np.array([event2id[e] for e in events], dtype=np.int64)
    T = len(ids)
    X = np.zeros((T, V), dtype=np.float32)
    X[np.arange(T), ids] = 1.0

    # -------- 시퀀스 생성 --------
    X3d, starts = to_sequences(X, args.win, args.stride)
    if len(X3d) == 0:
        raise ValueError("시퀀스가 생성되지 않았습니다. win/stride를 줄이거나 데이터 길이를 확인하세요.")
    print(f"[INFO] 생성된 시퀀스 개수: {len(X3d)}")

    # -------- 학습/검증/테스트 분할 (시퀀스 기준) --------
    N = len(X3d)
    trN = int(N * args.train_ratio)
    valN = int(N * args.val_ratio)
    if trN <= 0:
        raise ValueError("train_ratio가 너무 작아서 학습 시퀀스가 없습니다.")
    if valN <= 0:
        valN = 1
    teN = N - trN - valN
    if teN < 0:
        raise ValueError("train_ratio와 val_ratio가 너무 커서 나머지가 없습니다. 값을 조정하세요.")

    Xtr = X3d[:trN]
    Xval = X3d[trN:trN + valN]
    Xte  = X3d[trN + valN:]  # 필요시 활용 가능

    print(f"[INFO] 시퀀스 분할: train={len(Xtr)}, val={len(Xval)}, test={len(Xte)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAE(
        n_features=V,
        hidden=args.hidden,
        latent=args.latent,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dl_tr = DataLoader(SeqDataset(Xtr), batch_size=args.batch, shuffle=True, drop_last=True)
    dl_val = DataLoader(SeqDataset(Xval), batch_size=args.batch, shuffle=False)

    # -------- 학습 루프 --------
    best_val = float("inf")
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, dl_tr, opt, device, clip=1.0)
        val_loss = eval_epoch(model, dl_val, device)
        print(f"[{epoch:03d}] train={tr_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("[INFO] Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # -------- 이상 점수 계산 --------
    print("[INFO] 전체 시퀀스에 대한 이상 점수 계산 중...")
    errs = anomaly_score(model, X3d, device)  # (N_seq,)
    thr = np.quantile(errs, args.quantile)
    flags = (errs >= thr).astype(int)

    # 시퀀스 시작 인덱스 → 시작 시간 매핑
    start_times = df.loc[starts, "time"].values

    out_df = pd.DataFrame({
        "seq_index": np.arange(N),
        "seq_start_row": starts,
        "seq_start_time": start_times,
        "recon_error": errs,
        "anomaly_flag": flags,
    })

    out_df.to_csv(args.out, index=False)
    print(f"[INFO] 결과 저장 완료 -> {args.out}")
    print(f"[INFO] 이상 임계값 @q={args.quantile}: {thr:.6f}")
    print("[INFO] 상위 일부 이상 시퀀스:")
    print(out_df.sort_values("recon_error", ascending=False).head(10))

if __name__ == "__main__":
    main()
