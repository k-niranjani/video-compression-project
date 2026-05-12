#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mstcn
from transformer2_3_1 import Transformer2_3_1


class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q):
        super().__init__()
        self.num_f_maps = mstcn_f_maps
        self.dim = mstcn_f_dim
        self.num_classes = out_features
        self.len_q = len_q

        self.transformer = Transformer2_3_1(
            d_model=out_features,
            d_ff=mstcn_f_maps,
            d_k=mstcn_f_maps,
            d_v=mstcn_f_maps,
            n_layers=1,
            n_heads=8,
            len_q=len_q,
        )
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)

    def forward(self, x, long_feature):
        # x is expected to be [1, 7, T]
        out_features = x.transpose(1, 2)  # [1, T, 7]
        inputs = []
        for i in range(out_features.size(1)):
            if i < self.len_q - 1:
                pad = torch.zeros(
                    (1, self.len_q - 1 - i, self.num_classes),
                    device=out_features.device,
                )
                current = torch.cat([pad, out_features[:, 0 : i + 1]], dim=1)
            else:
                current = out_features[:, i - self.len_q + 1 : i + 1]
            inputs.append(current)

        inputs = torch.stack(inputs, dim=0).squeeze(1)  # [T, len_q, 7]
        feas = torch.tanh(self.fc(long_feature).transpose(0, 1))  # [T, 1, 7] or similar
        output = self.transformer(inputs, feas)
        return output


def load_test_pickle(test_pkl: Path):
    with test_pkl.open("rb") as f:
        payload = pickle.load(f)
    test_paths = payload[0]
    test_labels = np.asarray(payload[1], dtype=np.int64)
    test_num_each = payload[2]
    return test_paths, test_labels, test_num_each


def build_video_metadata(test_paths, test_num_each):
    video_names = []
    start_vidx = []
    count = 0

    for n in test_num_each:
        start_vidx.append(count)
        first_path = Path(test_paths[count])
        video_names.append(first_path.parent.name)
        count += n

    return video_names, start_vidx


def save_phase_txt(out_file: Path, preds: np.ndarray):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        f.write("Frame\tPhase\n")
        for i, p in enumerate(preds):
            f.write(f"{i}\t{int(p)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pkl", default="test_paths_labels.pkl")
    parser.add_argument("--lfb-test", default="./LFB/g_LFB50_test.pkl")
    parser.add_argument("--tecno-model", required=True)
    parser.add_argument("--trans-model", required=True)
    parser.add_argument("--out-dir", default="./eval/phase")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--mstcn-layers", type=int, default=8)
    parser.add_argument("--mstcn-f-maps", type=int, default=32)
    parser.add_argument("--mstcn-f-dim", type=int, default=2048)
    parser.add_argument("--mstcn-stages", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    test_paths, test_labels, test_num_each = load_test_pickle(Path(args.test_pkl))
    video_names, test_start_vidx = build_video_metadata(test_paths, test_num_each)

    with open(args.lfb_test, "rb") as f:
        g_LFB_test = pickle.load(f)

    print("test videos:", len(test_num_each))
    print("test frames:", len(test_paths))
    print("g_LFB_test shape:", g_LFB_test.shape)

    tecno = mstcn.MultiStageModel(
        args.mstcn_stages,
        args.mstcn_layers,
        args.mstcn_f_maps,
        args.mstcn_f_dim,
        args.num_classes,
        True,
    )
    tecno.load_state_dict(torch.load(args.tecno_model, map_location=device))
    tecno.to(device)
    tecno.eval()

    trans_model = Transformer(
        args.mstcn_f_maps,
        args.mstcn_f_dim,
        args.num_classes,
        args.sequence_length,
    )
    trans_model.load_state_dict(torch.load(args.trans_model, map_location=device))
    trans_model.to(device)
    trans_model.eval()

    out_dir = Path(args.out_dir)
    pkl_dir = out_dir.parent / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    all_video_acc = []

    with torch.no_grad():
        for i, video_name in enumerate(video_names):
            start = test_start_vidx[i]
            length = test_num_each[i]
            end = start + length

            labels_phase = torch.LongTensor(test_labels[start:end, 0]).to(device)

            long_feature_np = g_LFB_test[start:end]
            if long_feature_np.shape[0] != length:
                raise RuntimeError(
                    f"LFB length mismatch for {video_name}: "
                    f"expected {length}, got {long_feature_np.shape[0]}"
                )

            long_feature = torch.tensor(long_feature_np, dtype=torch.float32, device=device).unsqueeze(0)
            video_fe = long_feature.transpose(2, 1)  # [1, 2048, T]

            out_features = tecno.forward(video_fe)[-1]
            out_features = out_features.squeeze(1)  # harmless if dim != 1

            logits = trans_model(out_features, long_feature)
            logits = logits.squeeze()

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            acc = (preds == labels_phase).float().mean().item()
            all_video_acc.append(acc)

            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()

            txt_path = out_dir / f"{video_name}-phase.txt"
            save_phase_txt(txt_path, preds_np)

            with (pkl_dir / f"{video_name}_pred.pkl").open("wb") as f:
                pickle.dump(preds_np, f)
            with (pkl_dir / f"{video_name}_score.pkl").open("wb") as f:
                pickle.dump(probs_np, f)

            print(f"{video_name}: frames={length}, acc={acc:.4f}, wrote {txt_path}")

    print()
    print(f"mean video accuracy: {np.mean(all_video_acc):.4f}")
    print(f"std video accuracy : {np.std(all_video_acc):.4f}")
    print(f"prediction txt dir : {out_dir}")


if __name__ == "__main__":
    main()