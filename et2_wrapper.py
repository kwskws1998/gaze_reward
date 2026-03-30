"""
et2_wrapper.py
==============
eyetrackpy의 FixationsPredictor_2를 대체하는 래퍼.
사용자가 학습한 RoBERTa 체크포인트(.pt / .safetensors)를 로드해서
reward_model_base.py가 기대하는 인터페이스를 그대로 제공.

환경변수 ET2_CHECKPOINT_PATH로 체크포인트 경로 지정.
setup_et_models.py를 실행하면 자동으로 설정됨.
"""

import os
import numpy as np
import torch
import transformers

# safetensors 선택적 import
try:
    from safetensors.torch import load_file as st_load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
WINDOW_SIZE = 512
OVERLAP = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 내부 RoBERTa 회귀 모델 (노트북 model.py와 동일)
# ──────────────────────────────────────────────

class _RobertaRegressionModel(torch.nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(model_name)
        embed_size = 1024 if "large" in model_name else 768
        self.decoder = torch.nn.Linear(embed_size, 5)

    def forward(self, input_ids, attention_mask, predict_mask):
        hidden = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        Y_pred = self.decoder(hidden)
        mask = (predict_mask == 0).unsqueeze(-1).expand_as(Y_pred).to(Y_pred.device)
        Y_pred = Y_pred.masked_fill(mask, -1.0)
        return Y_pred


# ──────────────────────────────────────────────
# 공개 인터페이스: reward_model_base.py가 임포트하는 클래스
# ──────────────────────────────────────────────

class FixationsPredictor_2:
    """
    reward_model_base.py 호환 인터페이스.

    reward_model_base.py의 호출 패턴:
        fp = FixationsPredictor_2(modelTokenizer=rm_tokenizer, remap=False)
        fixations, fix_attn, _, _, _, _ = fp._compute_mapped_fixations(input_ids, attention_mask)

    fixations shape: (1, seq_len, 5)  — 5 ET features per RM token
    """

    def __init__(self, modelTokenizer, remap=False,
                 checkpoint_path=None, roberta_model_name="roberta-base"):
        self.rm_tokenizer = modelTokenizer
        self.roberta_model_name = roberta_model_name

        # 체크포인트 경로 결정: 인자 > 환경변수 > 기본값
        ckpt = (
            checkpoint_path
            or os.environ.get("ET2_CHECKPOINT_PATH", "./checkpoints/et_predictor2_seed123")
        )
        self.checkpoint_path = ckpt

        # RoBERTa 토크나이저 (예측용 — RM 토크나이저와 다름)
        self.roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained(
            roberta_model_name, add_prefix_space=True
        )

        # 모델 로드
        self.model = _RobertaRegressionModel(roberta_model_name).to(device)
        self._load_checkpoint(ckpt)
        self.model.eval()

        print(f"[et2_wrapper] FixationsPredictor_2 로드 완료: {ckpt}")

    # ── 체크포인트 로드 ──────────────────────────

    def _load_checkpoint(self, path):
        """확장자 없어도 .safetensors → .pt 순으로 탐색"""
        # 확장자가 이미 붙어 있는 경우
        if os.path.isfile(path):
            return self._load_from_file(path)

        # 확장자 없이 넘어온 경우
        for ext in [".safetensors", ".pt", ".bin"]:
            candidate = path + ext
            if os.path.isfile(candidate):
                return self._load_from_file(candidate)

        raise FileNotFoundError(
            f"[et2_wrapper] 체크포인트를 찾을 수 없습니다: {path}[.safetensors/.pt]\n"
            "ET2_CHECKPOINT_PATH 환경변수를 확인하거나 setup_et_models.py를 다시 실행하세요."
        )

    def _load_from_file(self, path):
        if path.endswith(".safetensors"):
            if not HAS_SAFETENSORS:
                raise ImportError("pip install safetensors")
            state = st_load_file(path, device=str(device))
        else:
            state = torch.load(path, map_location=device)
        self.model.load_state_dict(state, strict=True)
        print(f"[et2_wrapper] 가중치 로드: {path}")

    # ── reward_model_base.py 호환 메인 인터페이스 ──

    def _compute_mapped_fixations(self, input_ids_rm, attention_mask_rm=None):
        """
        input_ids_rm:      (1, seq_len) LongTensor  — RM tokenizer token ids
        attention_mask_rm: (1, seq_len) LongTensor  — 1=real token, 0=pad

        반환 (reward_model_base.py 6-tuple 규격):
            fixations               (1, seq_len, 5)  float32
            fixations_attention_mask (1, seq_len)    long
            None, None, None, None
        """
        if attention_mask_rm is None:
            attention_mask_rm = torch.ones_like(input_ids_rm)

        ids  = input_ids_rm[0].cpu().tolist()
        mask = attention_mask_rm[0].cpu().tolist()

        # 패딩 제거 후 RM 토크나이저로 텍스트 복원
        pad_id = self.rm_tokenizer.pad_token_id or 0
        ids_no_pad = [i for i, m in zip(ids, mask) if m == 1 and i != pad_id]
        text = self.rm_tokenizer.decode(ids_no_pad, skip_special_tokens=True)

        # RoBERTa로 단어 수준 ET 특징 예측
        word_features, words = self._predict_words(text)

        # RM 토큰 공간으로 리매핑
        remapped = self._remap_to_rm_tokens(word_features, words, ids, mask)

        fixations = remapped.unsqueeze(0).to(input_ids_rm.device)
        fix_attn  = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(input_ids_rm.device)

        return fixations, fix_attn, None, None, None, None

    # ── 내부 예측 로직 ────────────────────────────

    def _predict_words(self, text):
        """텍스트를 받아 단어 수준 5-feature 예측 반환"""
        words = text.strip().split()
        if not words:
            return np.zeros((0, 5), dtype=np.float32), words

        enc = self.roberta_tokenizer(
            [words],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        input_ids   = enc["input_ids"].to(device)
        attn_mask   = enc["attention_mask"].to(device)

        token_preds = self._sliding_window_predict(input_ids, attn_mask)

        # 각 단어의 첫 번째 서브워드(Ġ) 예측값을 단어 대표값으로 사용
        word_features = self._aggregate_to_words(token_preds, input_ids.squeeze(0))
        return word_features, words

    def _sliding_window_predict(self, input_ids, attn_mask):
        """512 초과 시퀀스를 슬라이딩 윈도우로 처리 (overlap 50, linear blending)"""
        seq_len = input_ids.shape[1]

        if seq_len <= WINDOW_SIZE:
            predict_mask = attn_mask.clone()
            with torch.no_grad():
                pred = self.model(input_ids, attn_mask, predict_mask)
            return pred.squeeze(0).cpu().numpy()

        preds   = np.zeros((seq_len, 5), dtype=np.float32)
        weights = np.zeros(seq_len, dtype=np.float32)
        stride  = WINDOW_SIZE - OVERLAP
        start   = 0

        while start < seq_len:
            end      = min(start + WINDOW_SIZE, seq_len)
            ids_win  = input_ids[:, start:end]
            mask_win = attn_mask[:, start:end]
            win_len  = end - start

            linear_w = np.ones(win_len, dtype=np.float32)
            if start > 0:
                rl = min(OVERLAP, win_len)
                linear_w[:rl] = np.linspace(0, 1, rl)
            if end < seq_len:
                rl = min(OVERLAP, win_len)
                linear_w[-rl:] = np.linspace(1, 0, rl)

            with torch.no_grad():
                pred_win = self.model(ids_win, mask_win, mask_win.clone())
            pred_np = pred_win.squeeze(0).cpu().numpy()

            for fi in range(5):
                preds[start:end, fi] += pred_np[:, fi] * linear_w
            weights[start:end] += linear_w

            if end == seq_len:
                break
            start += stride

        nz = weights > 0
        preds[nz] /= weights[nz, None]
        return preds

    def _aggregate_to_words(self, token_preds, input_ids_1d):
        """토큰 수준 예측 → 단어 수준 (첫 서브워드 값 사용)"""
        tokens = [self.roberta_tokenizer.convert_ids_to_tokens(int(i))
                  for i in input_ids_1d]

        word_feats = []
        for idx, tok in enumerate(tokens):
            if tok in ("<s>", "</s>", "<pad>"):
                continue
            # Ġ로 시작하면 새 단어의 첫 서브워드
            if tok.startswith("Ġ"):
                pred = np.clip(token_preds[idx], 0, None)
                word_feats.append(pred)

        return np.array(word_feats, dtype=np.float32) if word_feats else np.zeros((0, 5), dtype=np.float32)

    # ── RM 토큰 공간으로 리매핑 ──────────────────

    def _remap_to_rm_tokens(self, word_features, words, rm_ids, rm_mask):
        """
        단어 수준 ET 특징을 RM 토크나이저의 토큰 공간으로 매핑.
        논문 Table 8 방식: 단어의 첫 번째 RM 토큰에만 값 할당, 나머지는 0.
        """
        seq_len = len(rm_ids)
        output  = torch.zeros(seq_len, 5, dtype=torch.float32)

        if len(word_features) == 0 or len(words) == 0:
            return output

        rm_tokens = self.rm_tokenizer.convert_ids_to_tokens(rm_ids)
        word_to_rm = _align_words_to_rm_tokens(words, rm_tokens, self.rm_tokenizer)

        n_words = min(len(words), len(word_features))
        for w_idx in range(n_words):
            if w_idx >= len(word_to_rm):
                break
            indices = word_to_rm[w_idx]
            if not indices:
                continue
            first = indices[0]
            if first < seq_len and rm_mask[first] == 1:
                output[first] = torch.tensor(word_features[w_idx], dtype=torch.float32)

        return output


def _align_words_to_rm_tokens(words, rm_tokens, rm_tokenizer):
    """
    단어 리스트를 RM 토큰 인덱스 리스트로 매핑.
    특수 토큰은 스킵. 각 단어에 속하는 토큰 인덱스 목록 반환.
    """
    special_ids = set(rm_tokenizer.all_special_ids)
    word_to_indices = []
    tok_idx = 0

    for word in words:
        indices        = []
        chars_remaining = len(word)

        while tok_idx < len(rm_tokens) and chars_remaining > 0:
            tok    = rm_tokens[tok_idx]
            tok_id = rm_tokenizer.convert_tokens_to_ids(tok)

            if tok_id in special_ids:
                tok_idx += 1
                continue

            # Llama/GPT 계열: Ġ, ▁, 공백 prefix 제거 후 실제 글자 수 계산
            tok_clean = tok.lstrip("Ġ▁ ")
            indices.append(tok_idx)
            chars_remaining -= len(tok_clean)
            tok_idx += 1

        word_to_indices.append(indices)

    return word_to_indices
