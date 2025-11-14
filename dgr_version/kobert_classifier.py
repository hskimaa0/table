"""
KoBERT 기반 텍스트 분류기
제목 / 설명 / 제목아님 3-class 분류
"""
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import numpy as np


class KoBERTClassifier(nn.Module):
    """KoBERT 기반 3-class 분류 모델"""

    def __init__(self, num_classes=3, dropout_rate=0.1):
        super(KoBERTClassifier, self).__init__()

        # KoBERT 모델 로드
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')

        # 분류 헤드
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_classes)  # KoBERT hidden size = 768

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # BERT 인코딩 (token_type_ids 제외)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [CLS] 토큰의 출력 사용
        pooled_output = outputs.pooler_output

        # 드롭아웃 + 분류
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class TableTextClassifier:
    """표 텍스트 분류 wrapper 클래스"""

    # 클래스 레이블
    LABELS = {
        0: "제목",
        1: "설명",
        2: "제목아님"
    }

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: 학습된 모델 경로 (None이면 미학습 모델)
            device: 'cuda' 또는 'cpu'
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🔧 디바이스: {self.device}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')

        # 모델 초기화
        self.model = KoBERTClassifier(num_classes=3)

        # 학습된 가중치 로드
        if model_path:
            print(f"📂 모델 로드: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("⚠️  미학습 모델 (fine-tuning 필요)")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, return_probs=False):
        """
        단일 텍스트 분류

        Args:
            text: 분류할 텍스트
            return_probs: True면 확률값도 반환

        Returns:
            예측 레이블 (str) 또는 (레이블, 확률) 튜플
        """
        # 토크나이징
        encoded = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        token_type_ids = encoded.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        # 추론
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            pred_prob = probs[0, pred_class].item()

        label = self.LABELS[pred_class]

        if return_probs:
            return label, pred_prob, probs[0].cpu().numpy()
        else:
            return label

    def predict_batch(self, texts, batch_size=16):
        """
        배치 예측

        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기

        Returns:
            [(텍스트, 레이블, 확률), ...] 리스트
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # 배치 토크나이징
            encoded = self.tokenizer(
                batch_texts,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            token_type_ids = encoded.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            # 추론
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, token_type_ids)
                probs = torch.softmax(logits, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1)

            # 결과 저장
            for j, text in enumerate(batch_texts):
                pred_class = pred_classes[j].item()
                pred_prob = probs[j, pred_class].item()
                label = self.LABELS[pred_class]

                results.append((text, label, pred_prob))

        return results


# ========== 학습용 함수 ==========
def train_kobert_classifier(train_data, val_data=None, epochs=3, batch_size=16, lr=2e-5, save_path='kobert_classifier.pt', pretrained_model_path=None):
    """
    KoBERT 분류기 학습

    Args:
        train_data: [(텍스트, 레이블), ...] 형태의 학습 데이터
                    레이블: 0=제목, 1=설명, 2=제목아님
        val_data: 검증 데이터 (선택)
        epochs: 학습 에폭
        batch_size: 배치 크기
        lr: 학습률
        save_path: 모델 저장 경로
        pretrained_model_path: 기존 학습된 모델 경로 (추가 학습용, None이면 처음부터 학습)
    """
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 학습 디바이스: {device}")

    # 데이터셋 클래스
    class TextDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text, label = self.data[idx]

            encoded = self.tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # token_type_ids가 없으면 0으로 채운 텐서 생성
            token_type_ids = encoded.get('token_type_ids')
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(encoded['input_ids'])

            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'token_type_ids': token_type_ids.squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }

    # 토크나이저 & 모델
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    model = KoBERTClassifier(num_classes=3)

    # 기존 모델 로드 (있으면)
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"📂 기존 모델 로드: {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print("✅ 기존 모델에서 이어서 학습합니다!")
    else:
        if pretrained_model_path:
            print(f"⚠️  모델 파일이 없습니다: {pretrained_model_path}")
        print("🆕 처음부터 학습을 시작합니다")

    model.to(device)

    # 데이터로더
    train_dataset = TextDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data:
        val_dataset = TextDataset(val_data, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 옵티마이저 & 손실함수
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 학습 루프
    print(f"\n🚀 학습 시작 (에폭: {epochs}, 배치: {batch_size}, 학습률: {lr})")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # 통계
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%")

        # 검증
        if val_data:
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    labels = batch['label'].to(device)

                    logits = model(input_ids, attention_mask, token_type_ids)
                    preds = torch.argmax(logits, dim=-1)

                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = 100 * val_correct / val_total
            print(f"  검증 정확도: {val_acc:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 모델 저장 완료: {save_path}")


# ========== 사용 예시 ==========
if __name__ == '__main__':
    # 1. 학습 데이터 예시 (실제로는 라벨링된 데이터 필요)
    train_data = [
        # (텍스트, 레이블: 0=제목, 1=설명, 2=제목아님)
        ("표 3.2 연간 실적", 0),
        ("□ 추진조직 구성", 0),
        ("표 B.8 월별 기온", 0),
        ("ㅇ 사업 개요", 0),
        ("① 주요 내용", 0),

        ("상세한 내용은 다음 표 B.12와 같다", 1),
        ("표 A.20에 의하면 다음과 같은 결과를 얻었다", 1),
        ("※ 본 자료는 참고용입니다", 1),

        ("123", 2),
        ("© 2024 All Rights Reserved", 2),
        ("단위: ℃", 2),
    ]

    # 2. 모델 학습 (주석 해제하여 사용)
    # train_kobert_classifier(train_data, epochs=3, save_path='kobert_table_classifier.pt')

    # 3. 추론 예시
    print("\n========== 추론 예시 ==========")
    classifier = TableTextClassifier(model_path=None)  # 미학습 모델

    test_texts = [
        "표 4.21 예산 집행 현황",
        "상세한 내용은 아래 표와 같다",
        "단위: %"
    ]

    for text in test_texts:
        label, prob, all_probs = classifier.predict(text, return_probs=True)
        print(f"'{text}' → {label} ({prob:.3f})")
        print(f"  전체 확률: 제목={all_probs[0]:.3f}, 설명={all_probs[1]:.3f}, 제목아님={all_probs[2]:.3f}")
