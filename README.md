# Gemma 4 E4B · Colab 대화

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/tmp_llm/blob/630f0c75396e04dd945ef62e36107bcda73260c9/gemma4_colab.ipynb)

배지는 실행 코드가 저장된 커밋에 고정되어 있습니다. 노트북을 수정해 배포할 때 링크의 커밋도 갱신하면 이전 버전 캐시와 구분되는 새 주소로 열립니다. GPU 런타임과 다운로드 파일은 배지 클릭으로 초기화되지 않습니다.

Google 공식 Gemma 4 E4B QAT Q4_0 모델과 Gradio로 한국어 대화를 해보는 Colab 노트북입니다.

## 실행

1. 위 **Open in Colab** 배지를 누릅니다.
2. **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
3. 셀을 위에서부터 실행합니다. 처음에는 CUDA 빌드와 약 5.15GB 모델 다운로드가 필요합니다.
4. 대화창 셀에서 비밀번호를 입력합니다.
5. 표시되는 Gradio 공유 링크를 열고 사용자 이름 `gemma`와 입력한 비밀번호로 로그인합니다.
6. 사용을 마치면 종료 셀을 실행하고 Colab 런타임을 삭제합니다.

## 기본 구성

| 항목 | 설정 |
|---|---|
| 모델 | Google Gemma 4 E4B QAT Q4_0 GGUF |
| 입력 | 텍스트 |
| 실행 엔진 | llama.cpp v0.6.0, CUDA |
| 대화창 | Gradio 6.29.1 |
| 문맥 / 답변 길이 | 4,096 / 최대 512토큰 |
| 동시 생성 | 1개 |
| 사고 모드 | 꺼짐 |

첫 셀에서 실행 설정을 변경할 수 있습니다. 문맥 한도를 넘으면 대화를 지우고 새로 시작하세요. E4B의 E는 유효 파라미터를 뜻하므로 파일 크기를 4B만으로 계산할 수 없습니다.

## Colab 이용 조건

무료 Colab은 GPU 종류와 사용 시간을 보장하지 않습니다. 노트북 UI를 벗어나 웹 UI를 주로 사용하는 방식은 제한되며 세션이 종료될 수 있습니다. Gradio 공유 링크를 만드는 이 노트북에도 해당 조건이 적용됩니다.

공유 링크에는 비밀번호가 필요합니다. 링크와 비밀번호를 본인만 보관하고, Hugging Face 토큰을 대화창 비밀번호로 사용하지 마세요. 대화 기록은 서버 파일에 저장하지 않으며, 노트북에 실행 출력이나 개인 대화를 저장한 뒤 공개하지 않도록 주의하세요.

## 참고 자료

- [Google 공식 양자화 모델](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)
- [Gemma 4 메모리 요구량](https://ai.google.dev/gemma/docs/core)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Gradio](https://www.gradio.app/)
- [Colab 이용 조건](https://research.google.com/colaboratory/faq.html)

## 실행 확인 범위

노트북 형식과 모든 코드 셀의 Python 구문을 확인했습니다. 대화 기록 전달과 답변 스트리밍은 모의 모델 응답으로 확인했고, Gradio 6.29.1을 로컬에서 실행해 비밀번호 로그인과 익명 접속 차단을 확인했습니다.

Colab T4에서 CUDA 빌드, 모델 로딩 및 실제 답변 생성까지 실행한 결과는 아직 확인하지 않았습니다.
