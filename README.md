# Gemma 4 12B QAT · HF Transformers 실험과 대화

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm/blob/hf-gemma4-12b-4bit/gemma4_hf_colab.ipynb)

**26B-A4B IQ4_XS · T4:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm/blob/hf-gemma4-12b-4bit/gemma4_26b_iq4xs_colab.ipynb)

Google 공식 Gemma 4 12B QAT 가중치를 HF Transformers와 bitsandbytes NF4 4비트로 실행하는 Colab 노트북입니다. 공식 MTP 보조 모델로 speculative decoding을 사용합니다. HF 셀에서 직접 텍스트·이미지 대화와 도구를 사용하고, PyTorch 내부 출력을 관찰할 수 있습니다. Gradio 대화창은 별도 셀에서 같은 모델과 생성 함수를 사용합니다.

## 실행

1. **Open in Colab**을 눌러 [gemma4_hf_colab.ipynb](gemma4_hf_colab.ipynb)을 엽니다.
2. **런타임 → 런타임 유형 변경**에서 GPU를 선택합니다. 이전 모델이 남아 있으면 런타임을 삭제한 뒤 새로 시작합니다.
3. **실행 설정 → 패키지 설치 → HF 모델 로딩 → MTP 보조 모델 로딩 → 대화·이미지·도구 함수** 셀을 순서대로 실행합니다.
4. **HF 셀에서 직접 대화** 또는 **Gradio 대화창** 셀을 실행합니다. 내부 출력을 보려면 생성이 끝난 뒤 **PyTorch 내부 출력 관찰** 셀을 실행합니다.
5. 사용을 마치면 Gradio 셀을 중지한 뒤 **사용 종료** 셀을 실행하고 Colab 런타임을 삭제합니다.

Gradio 셀은 실행 상태를 유지하며 로그와 오류를 출력합니다. 다른 HF 셀을 사용하려면 대화가 끝난 뒤 Gradio 셀을 중지하세요. 모델 실행은 한 번에 하나씩 처리합니다. 셀을 계속 실행해도 Colab의 세션 종료 정책을 막지는 못합니다. 설치 전에 패키지를 이미 import한 런타임에서 버전을 바꿨다면 런타임을 다시 시작하고 설정 셀부터 실행하세요.

## 다운로드 가속

첫 설정 셀의 `XET_HIGH_PERFORMANCE=True`로 `HF_XET_HIGH_PERFORMANCE=1`을 설정합니다. HF 라이브러리를 불러오기 전에 적용하며, Xet이 네트워크·디스크·CPU를 더 적극적으로 사용합니다. 실제 속도 향상은 Colab과 서버 상태에 따라 다릅니다.

이미 HF 라이브러리를 불러온 Python 세션에는 설정 변경이 즉시 적용되지 않습니다. 변경 후에는 Python 세션을 다시 시작하고 첫 셀부터 실행하세요. 이미 진행 중인 다운로드의 속도를 즉시 바꾸는 옵션은 아닙니다.

## 런타임·주요 라이브러리

| 구성 | 환경·버전 | 용도 |
|---|---|---|
| 런타임 | Google Colab GPU · Python 3 | 노트북 실행 |
| PyTorch | Colab의 CUDA 설치 버전 | 텐서 연산·모델 실행·내부 출력 관찰 |
| Transformers | 5.17.0 · vision 의존성 포함 | 모델·processor 로딩, 생성, 도구 응답 파싱 |
| bitsandbytes | 0.50.2 | 언어 모델 Linear 레이어의 NF4 4비트 로딩 |
| Accelerate | 1.15.0 | 모델 로딩·장치 배치 |
| huggingface-hub | 1.33.0 | 공식 가중치 다운로드 |
| 패키지 설치 | uv 0.12.23 | Colab Python 환경에 패키지 설치 |
| Gradio | 6.29.1 | 별도 대화창·이미지 및 파일 첨부 |
| DDGS | 9.16.0 | API 키 없이 웹 검색 |

Transformers 5.18.0의 Gemma MTP 후보 생성기 인수 누락 오류를 피하기 위해, 실제 NF4+MTP 생성이 통과한 5.17.0으로 고정합니다.

## 모델·기본 설정

원본 가중치는 `google/gemma-4-12B-it-qat-q4_0-unquantized`입니다. 반정밀도 가중치 약 **23.9GB**를 다운로드한 뒤 로딩 과정에서 NF4로 양자화합니다. 모델과 실행 설정은 노트북 첫 코드 셀에서 변경합니다.

| 항목 | 기본값 |
|---|---|
| 모델 | Google 공식 Gemma 4 12B QAT |
| MTP 보조 모델 | `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` · 가중치 약 846MB |
| speculative decoding | 켜짐 · 후보 3토큰 · 보조 모델은 반정밀도 |
| 양자화 | NF4 4비트 · double quantization 켜짐 |
| 계산 정밀도 | T4는 FP16, 하드웨어 BF16 지원 GPU는 BF16 |
| 비전·오디오 인코더·출력층 | 반정밀도 유지 · 출력층은 입력 임베딩과 가중치 공유 |
| 모델 배치 | GPU 한 개에 로딩 |
| attention / 사고 모드 | SDPA / 꺼짐 |
| 문맥 / 최대 답변 | 4,096 / 2,048토큰 |
| 생성 설정 | temperature 0.6 · top_p 0.9 · top_k 64 · min_p 0.05 · repetition_penalty 1.1 |
| 동시 생성 | 1개 |

QAT는 학습 방식입니다. 이 노트북의 실행 양자화 형식 NF4는 기존 Balanced의 Q4_K_M 및 공식 GGUF의 Q4_0과 다릅니다. 공식 모델의 답변 성향도 Balanced와 다릅니다.

## MTP speculative decoding

**MTP 보조 모델 로딩** 셀에서 `assistant_model`을 주 모델과 같은 계산 정밀도로 GPU에 올립니다. 공통 `stream_model_events()`가 `model.generate()`에 전달하므로 HF 직접 대화와 Gradio에서 모두 사용합니다. 후보 토큰은 주 모델이 검증하고, 내부 관찰 셀의 단일 forward는 기존대로 실행합니다.

`NUM_ASSISTANT_TOKENS`로 제안할 후보 토큰 수를 조절합니다. `ENABLE_SPECULATIVE_DECODING=False`로 바꾸고 보조 모델 로딩 셀을 실행하면 보조 모델을 해제하고 일반 생성으로 돌아갑니다. 보조 모델 셀을 다시 실행해도 주 모델을 다시 로딩하지 않습니다. 추가 VRAM 사용량과 실제 가속 효과는 실행 후 확인하세요.

## HF 직접 대화·Gradio

HF 직접 실행 셀에서 `HF_PROMPT`, `HF_IMAGE_PATH`, `HF_TEXT_FILE_PATH`, `HF_HISTORY`를 지정합니다. 첨부 경로는 Colab에 업로드한 파일의 로컬 경로입니다. `generate_reply(message, history)`를 직접 호출할 수도 있습니다.

Gradio 대화창은 이미지와 UTF-8 텍스트 파일(`.txt`, `.md`, `.csv`, `.json`, `.log`) 첨부, 메시지 편집, 답변 스트리밍을 지원합니다. 메시지를 편집하면 다음 요청에 수정된 대화 기록이 반영됩니다. 두 실행 경로는 같은 모델과 도구 함수를 사용합니다.

| 도구 | 기능 |
|---|---|
| 계산기 | 사칙연산·거듭제곱·괄호가 포함된 수식 계산 |
| 현재 시각 | 지정한 시간대의 날짜와 시각 조회 |
| 웹 검색 | 최대 3개 결과의 제목·URL·요약 조회 |
| 파일 읽기 | 현재 대화에 첨부한 파일을 한 번에 최대 6,000자 읽기 |

모델의 도구 호출을 공식 tokenizer의 응답 파서로 읽고 실행 결과를 다시 모델에 전달합니다. 요청당 최대 3차례·6개 도구를 실행합니다. `GRADIO_SHARE=True`이면 외부 접속 가능한 공유 링크를 엽니다.

## PyTorch 내부 출력 관찰

관찰 셀은 짧은 입력으로 한 번 forward를 실행하고 결과를 CPU 텐서로 보관합니다. 기본 입력 한도는 256토큰이며, 이미지 입력은 필요에 따라 `INSPECT_MAX_TOKENS`를 조절합니다.

| 설정·변수 | 용도 |
|---|---|
| `INSPECT_TEXT` / `INSPECT_IMAGE_PATH` | 관찰할 텍스트·이미지 입력 |
| `INSPECT_HIDDEN_STATES` | hidden states 저장 여부 · 기본 켜짐 |
| `INSPECT_ATTENTIONS` | attention 저장 여부 · 기본 꺼짐 |
| `INSPECT_MODULE_NAME` | `model.named_modules()`에서 확인한 모듈 이름을 지정해 출력 hook 등록 |
| `inspection_outputs.logits` | 마지막 입력 위치에서 다음 토큰을 예측한 점수 |
| `inspection_outputs.hidden_states` | 전체 입력 위치의 hidden states |
| `inspection_outputs.attentions` | attention 관찰을 켰을 때의 값 |
| `captured_module_output` | 선택한 모듈의 출력 |

attention 관찰을 켜면 해당 forward 동안 eager attention을 사용하고 원래 설정으로 복원합니다. 저장량은 입력 길이의 제곱에 비례합니다. 관찰 결과를 해제하려면 `inspection_outputs = captured_module_output = None`을 실행합니다.

최근 대화의 입력·응답은 `last_messages`, `last_inputs`, `last_raw_response`, `last_response`에서 확인합니다. `last_inputs`는 CPU에 보관합니다.

## 메모리·검증 범위

모델 다운로드 크기와 실제 GPU 사용량은 다릅니다. MTP 보조 모델·hidden states·후보 검증, 이미지·긴 대화·내부 출력 저장에는 추가 메모리가 필요하며, VRAM 8GB·RAM 10~12GB 이내 실행을 보장하지 않습니다. 모델은 GPU 한 개에 배치하므로 공간이 부족하면 로딩 오류가 납니다.

모델 로딩 및 직접 대화·관찰 후 GPU 할당·예약·최대 할당 메모리와 Python 프로세스 최대 RAM을 표시합니다. 사용 종료 셀은 Gradio와 주 모델·보조 모델·관찰 결과를 해제합니다.

도구 호출·결과 전달·스트리밍 연결, 로컬 이미지 전처리, Gradio 구성과 셀 실행을 유지하는 launch 경로, 작은 Gemma 4 CPU 모델의 NF4+반정밀도 MTP 생성·샘플링·스트리밍 및 logits·hidden states·attention·hook과 설정 복원을 확인했습니다. 실제 12B 가중치의 GPU 실행·속도·메모리는 Colab에서 확인해야 합니다.

## 관련 파일·문서

- [기존 README](README_GGUF.md)
- [기존 GGUF 노트북](gemma4_colab.ipynb)
- [공식 QAT 가중치](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)
- [공식 MTP 보조 모델](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized-assistant)
- [HF bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [HF 응답 파서](https://huggingface.co/docs/transformers/en/chat_response_parsing)
- [Xet 고성능 다운로드](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hf_xet_high_performance)
