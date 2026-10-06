# Gemma 4 · Colab 대화

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm/blob/c915734/gemma4_colab.ipynb)

Gemma 4 GGUF 모델을 llama.cpp로 실행하고, Gradio에서 텍스트·이미지 대화를 하는 Colab 노트북입니다. 계산, 현재 시각 조회, 웹 검색, 첨부 텍스트 파일 읽기를 지원합니다.

## 실행

1. **Open in Colab**을 눌러 노트북을 엽니다.
2. **런타임 → 런타임 유형 변경**에서 GPU를 선택합니다. 기존 모델 서버가 실행 중이면 런타임을 삭제한 뒤 새로 시작합니다.
3. 셀을 위에서부터 실행하고 대화창의 Gradio 공유 링크를 엽니다.
4. 사용을 마치면 대화창 셀을 중지하고 종료 셀을 실행한 뒤 Colab 런타임을 삭제합니다.

## 런타임·주요 라이브러리

| 구성 | 환경·버전 | 용도 |
|---|---|---|
| 런타임 | Google Colab GPU · Python 3 | 노트북 실행 |
| 추론 엔진 | llama.cpp b11433 | GGUF 모델 실행 |
| 서버 바이너리 | Ubuntu 24.04 x64 · CUDA 12.8 | 사전 빌드 서버와 CUDA 라이브러리 |
| 패키지 설치 | uv 0.12.23 | Colab Python 환경에 패키지 설치 |
| Gradio | 6.29.1 | 대화창·이미지 및 파일 첨부 |
| huggingface-hub | 2.1.1 | 모델 파일 다운로드 |
| requests | 2.34.2 | 모델 서버 요청·답변 스트리밍 |
| DDGS | 9.16.0 | API 키 없이 웹 검색 |
| tqdm | Colab 환경에서 사용 | 다운로드 진행률 표시 |

## 모델·기본 설정

현재 모델 저장소는 `HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced`입니다. 모델과 실행 설정은 노트북 첫 코드 셀에서 변경합니다.

| 항목 | 기본값 |
|---|---|
| 모델 파일 | `Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf` |
| 이미지 처리 파일 | `mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf` |
| MTP 파일 | `mtp-gemma-4-12B-it.gguf` |
| 추측 디코딩 | `draft-mtp` |
| 문맥 / 최대 답변 | 4,096 / 2,048토큰 |
| 동시 생성 | 1개 |
| Flash Attention / 사고 모드 | 꺼짐 / 꺼짐 |

## 대화·도구

이미지와 UTF-8 텍스트 파일(`.txt`, `.md`, `.csv`, `.json`, `.log`)을 첨부할 수 있습니다. 메시지를 편집하면 다음 요청에 수정된 대화 기록이 반영됩니다.

| 도구 | 기능 |
|---|---|
| 계산기 | 사칙연산·거듭제곱·괄호가 포함된 수식 계산 |
| 현재 시각 | 지정한 시간대의 날짜와 시각 조회 |
| 웹 검색 | 최대 3개 결과의 제목·URL·요약 조회 |
| 파일 읽기 | 현재 대화에 첨부한 파일을 한 번에 최대 6,000자 읽기 |

모델이 필요에 따라 도구를 호출하며, 요청당 최대 3차례·6개 도구를 실행합니다.

## HF Transformers · PyTorch 실험

별도 노트북 [gemma4_hf_colab.ipynb](gemma4_hf_colab.ipynb)은 Google 공식 Gemma 4 12B QAT 가중치를 bitsandbytes NF4 4비트로 로딩합니다. 기존 GGUF 노트북과 독립적으로 실행합니다.

| 항목 | HF 노트북 설정 |
|---|---|
| 원본 가중치 | `google/gemma-4-12B-it-qat-q4_0-unquantized` |
| 다운로드 | 반정밀도 가중치 약 23.9GB, 로딩 중 NF4로 양자화 |
| 실행 | Transformers 5.18.0 · bitsandbytes 0.50.2 · Accelerate 1.15.0 |
| 계산 정밀도 | T4는 FP16, BF16 지원 GPU는 BF16 |
| 내부 접근 | 같은 런타임의 `model`, `processor` |
| 직접 대화 | HF 셀에서 텍스트·이미지·파일·도구 사용 |
| 내부 관찰 | 마지막 위치 logits, hidden states, 선택적 attention·모듈 출력 hook |
| Gradio | 독립된 UI 셀에서 이미지·파일 첨부, 도구 사용, 메시지 편집, 답변 스트리밍 |

### 실행 순서

1. `gemma4_hf_colab.ipynb`을 Colab에 업로드하고 GPU 런타임을 선택합니다.
2. 설정, 패키지 설치, 모델 로딩, 대화·도구 함수 셀을 실행합니다.
3. 직접 대화 셀이나 Gradio UI 셀을 실행합니다. 두 경로는 같은 모델을 사용합니다.
4. 생성이 끝난 뒤 내부 관찰 셀을 실행합니다. 결과는 `inspection_outputs`와 `captured_module_output`에 CPU 텐서로 남습니다.
5. 사용 종료 셀을 실행하고 Colab 런타임을 삭제합니다.

### 메모리와 관찰 설정

NF4는 기존 Balanced의 Q4_K_M 및 공식 GGUF의 Q4_0과 다른 양자화 방식입니다. 공식 가중치의 답변 성향도 Balanced와 다릅니다. 모델 다운로드 크기와 실제 GPU 사용량은 다르며, VRAM 8GB·RAM 10~12GB 이내 실행을 보장하지 않습니다. 로딩 및 실행 후 GPU 할당·예약·최대 할당 메모리와 Python 프로세스 최대 RAM을 표시합니다.

기본 관찰 입력 한도는 256토큰입니다. `INSPECT_ATTENTIONS=True`이면 해당 forward 동안 eager attention을 사용합니다. `INSPECT_MODULE_NAME`은 `model.named_modules()`에서 확인한 이름으로 지정합니다. 최근 대화 입력과 응답은 `last_messages`, `last_inputs`, `last_raw_response`, `last_response`에서 확인합니다. `last_inputs`와 관찰 결과는 CPU에 보관합니다.

Gradio 셀 실행 후에도 다른 HF 셀을 실행할 수 있습니다. 모델 실행은 한 번에 하나씩 처리합니다. `GRADIO_SHARE=True`이면 외부 접속 가능한 공유 링크를 엽니다. 사용 종료 시 UI와 모델·관찰 결과를 해제합니다.

[공식 QAT 가중치](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) · [HF bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) · [HF 응답 파서](https://huggingface.co/docs/transformers/en/chat_response_parsing)
