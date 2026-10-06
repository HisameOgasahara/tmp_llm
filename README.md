# Gemma 4 E4B · Colab 대화

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm/blob/3ea79354e245e6f1af29fe13409dd6c55f45e229/gemma4_colab.ipynb)

배지는 실행 코드가 저장된 커밋에 고정되어 있습니다. 새 버전 배포 시 커밋 주소도 갱신해 이전 버전 캐시와 구분합니다. 배지 클릭으로 GPU 런타임이나 다운로드 파일이 삭제되지는 않습니다.

Google 공식 Gemma 4 E4B QAT Q4_0 GGUF와 Gradio로 한국어 대화를 해보는 Colab 노트북입니다. 공식 llama.cpp CUDA 사전 빌드 파일을 받아 실행합니다. Ubuntu 24.04용 배포 파일이므로 Colab의 최신 기본 런타임을 사용하세요.

## 실행

1. 위 **Open in Colab** 배지를 누릅니다.
2. 기존 vLLM 또는 llama.cpp 서버가 실행 중이면 **런타임 → 연결 해제 및 런타임 삭제** 후 새 런타임에서 시작합니다.
3. **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
4. 셀을 위에서부터 실행합니다. 서버 파일 약 172MB, CUDA 라이브러리 약 594MB, GGUF 모델 약 5.15GB를 다운로드합니다.
5. 대화창 셀을 실행하고 Gradio 공유 링크를 엽니다. 로그인 없이 바로 대화할 수 있습니다. 입력창의 **전송 아이콘**을 누르거나 **Shift + Enter**로 전송합니다.
   사용자 메시지와 모델 답변은 편집 아이콘으로 수정할 수 있으며, 수정한 내용은 다음 요청의 대화 기록에 반영됩니다. 수정만으로 답변을 다시 생성하지는 않습니다.
6. 대화창 셀은 실행 상태로 유지되며 대화 중 서버 로그와 오류 상세를 표시합니다.
7. 사용을 마치면 대화창 셀의 중지 버튼을 누른 뒤 종료 셀을 실행하고 Colab 런타임을 삭제합니다.

## 기본 구성

| 항목 | 설정 |
|---|---|
| 모델 | Google Gemma 4 E4B QAT Q4_0 GGUF |
| 입력 | 텍스트 |
| 실행 엔진 | llama.cpp b11433, Ubuntu x64 CUDA 12.8 사전 빌드 |
| 설치 | 바이너리와 CUDA 라이브러리 다운로드·압축 해제 |
| 대화창 | Gradio 6.29.1 |
| 문맥 / 답변 길이 | 4,096 / 최대 2,048토큰 |
| 동시 생성 | 1개 |
| Flash Attention | 꺼짐 |
| 사고 모드 | 꺼짐 |

첫 셀에서 실행 설정을 변경할 수 있습니다. 문맥 한도를 넘으면 대화를 지우고 새로 시작하세요. 텍스트 대화용 GGUF 파일 하나를 사용하며 이미지용 mmproj는 다운로드하지 않습니다.

설치·다운로드·압축 해제·모델 로딩 로그를 표시합니다. 다운로드에는 진행률을 표시하고, 별도의 경과 시간 메시지는 출력하지 않습니다. 대화 중에도 대화창 셀에서 서버 로그와 요청 처리 단계를 계속 표시하며, 실패 시 모델 서버 오류 응답과 Python 예외를 표시합니다. 소스 컴파일과 vLLM·PyTorch·Transformers 설치 단계는 없습니다.

## Colab 이용 조건

무료 Colab은 GPU 종류와 사용 시간을 보장하지 않습니다. 노트북 UI를 벗어나 웹 UI를 주로 사용하는 방식은 제한되며 세션이 종료될 수 있습니다. Gradio 공유 링크를 만드는 이 노트북에도 해당 조건이 적용됩니다.

대화 중 요청 처리 단계와 실패 상세는 Colab 대화창 셀에 표시됩니다.

## 참고 자료

- [Google 공식 GGUF 모델](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)
- [llama.cpp 사전 빌드 배포 파일](https://github.com/ggml-org/llama.cpp/releases/tag/b11433)
- [llama.cpp 서버 사용법](https://github.com/ggml-org/llama.cpp/blob/b11433/tools/server/README.md)
- [Gradio](https://www.gradio.app/)
- [Colab 이용 조건](https://research.google.com/colaboratory/faq.html)

## 실행 확인 범위

노트북 형식과 코드 구문, 사전 빌드 배포 파일 구성, 로그 전달 및 서버 준비 상태 확인을 로컬에서 확인했습니다. 대화 기록 전달과 답변 스트리밍, HTTP 오류 상세 및 예외 로그, 로딩 완료 후 서버 로그 표시를 모의 응답으로 확인했습니다.

공식 CUDA 12.8 배포 설정에 T4의 CUDA 아키텍처 7.5가 포함되어 있습니다. Colab T4에서 바이너리 실행, 모델 로딩 및 실제 답변 생성까지는 직접 검증하지 않았습니다.
