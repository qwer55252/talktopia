# Talktopia results viewer

저장된 시뮬레이션과 여러 평가기의 결과를 한 에피소드 단위로 확인하는 로컬 페이지다.
실험 실행, 모델 호출, DB 저장 기능은 없다. 결과와 프로필은 읽기 전용으로 사용한다.

## 실행

저장소 루트에서 실행한다. 기존 Talktopia 환경에는 필요한 의존성이 이미 설치돼 있다.
실험 중에는 공유 `.venv`에 패키지를 설치하거나 업데이트하지 않는다.

```bash
cp demo/config.example.json demo/config.local.json
.venv/bin/python -m demo --config demo/config.local.json
```

브라우저에서 http://127.0.0.1:8765 를 연다. 포트를 바꾸려면 `--port 8766`을 추가한다.
서버는 localhost에서만 실행된다. 원격 서버에서는 SSH 또는 VS Code의 포트 전달을 사용한다.

음성·모델 실행 환경 없이 조회 페이지만 설치할 수도 있다. Python 3.12 이상을 사용한다.
이 환경은 저장소 루트의 공유 `.venv`와 별개다.

```bash
python3 -m venv demo/.venv
demo/.venv/bin/python -m pip install -r demo/requirements.txt
demo/.venv/bin/python -m demo --config demo/config.local.json
```

의존성 버전은 저장소의 `requirements.lock`을 따른다. 프런트엔드 빌드는 필요 없다.

## 결과 추가

`config.local.json`은 Git에서 제외된다. 예시 설정의 경로를 실제 결과 경로로 바꾼다.

```json
{
  "runs": [
    {
      "id": "pair-01",
      "label": "Pair 01",
      "simulation_dir": "/path/to/simulation-run",
      "evaluation_dirs": ["/path/to/qwen-evaluation", "/path/to/glm-evaluation"]
    }
  ]
}
```

- `id`는 실행마다 달라야 하며 영문·숫자·밑줄·하이픈을 사용할 수 있다.
- `label`은 선택 항목이다. 생략하면 `id`를 표시한다.
- 경로는 절대경로 또는 **설정 파일의 디렉터리 기준** 상대경로다. `~`도 허용한다.
- 다른 시뮬레이션은 `runs`에 항목을 추가한다. 평가는 해당 실행의 `evaluation_dirs`에 추가한다.
- 파일을 저장한 뒤 페이지의 **새로고침**을 누른다. 서버 재시작은 필요 없다.
- 부모 matrix 디렉터리가 아니라 `03_simulation.json`이 있는 실행 디렉터리를 지정한다.
  평가는 `evaluation_manifest.json`과 `04_sotopia_eval_reevaluate_existing.json`이 있는 디렉터리다.
- 등록하지 않은 형제 디렉터리를 자동으로 찾아 추가하지 않는다.

전체 목록은 manifest와 진행 요약을 합쳐 읽는다. 실패·대기 항목도 유지한다.
상세 화면은 선택한 에피소드의 파일만 읽으며, 요약에 저장된 재시도 경로를 따른다.
자동 폴링은 하지 않는다. 실행 중인 결과는 새로고침으로 갱신한다.

## 읽는 내용과 연결 규칙

상단의 세 profile 토글에서 저장된 필드를 확인할 수 있다. 원본 파일 내용은 수정하거나 번역하지 않는다.
Agent 1과 Agent 2의 순서를 대화, 목표, 모델, 평가표에서 동일하게 유지한다.
모델 이름은 manifest보다 원본 EpisodeLog의 `models`를 우선한다.

프로필은 `run_config.json`의 `input_fingerprints`에서 찾고 SHA-256을 확인한다.
Relationship profile은 두 에이전트 ID와 환경의 관계 유형이 모두 일치해야 한다.
DB를 옮겼다면 해당 실행 설정에 `"profiles_dir": "/new/profile-db"`를 추가한다.
그 디렉터리 아래에 `AgentProfile/`, `EnvironmentProfile/`, `RelationshipProfile/`이 있어야 한다.
이 경우도 원래 기록된 해시와 비교한다. 프로필이 없거나 달라졌으면 표시하지 않으며,
Agent profile 안에서 에피소드 시작 시 전달된 원래 맥락을 확인할 수 있다.

평가를 비교표에 넣기 전에 원본 JSON의 SHA-256, 환경 ID와 에이전트 순서를 확인한다.
Surface5 평가에 이벤트 해시가 있으면 그 파일도 확인한다.
과거 절대경로가 현재 서버와 달라도 내용이 같으면 연결한다.
평가 시점의 원본이 다르거나 확인할 수 없으면 점수 대신 `—`를 표시한다.
평가 보고서는 이 상태를 표시한 채 별도로 읽을 수 있다. 실패·제외·미평가도 0점으로 바꾸지 않는다.
점수는 저장값이며, 표시만 소수 둘째 자리까지 반올림한다.

대화의 기본 텍스트는 상대가 받은 ASR 기록이다. 생성 문장이나 TTS 문장으로 바꾸지 않는다.
전체 음성을 재생하면 현재 발화를 강조하고, 시간 정보가 있는 대사를 누르면 그 시점으로 이동한다.
클릭 시 재생·일시정지 상태는 유지한다. 에피소드를 바꾸면 이전 재생을 멈춘다.

- Round-robin: 완료된 발화 WAV 길이와 발화 사이 0.3초 간격으로 시간을 계산한다.
  TTS 생략 기록은 제외한다. 전체 WAV 길이와 맞지 않거나 발화 WAV가 없으면 클릭 이동을 해제한다.
- Surface5: 저장된 시작·종료 시각을 사용하며 겹치는 발화도 함께 강조한다.
  비음성 행동은 이벤트의 확정 시각에 표시한다. 미확정 발화는 별도로 표시한다.
- 음성이 없는 에피소드도 대화와 보고서는 확인할 수 있다.

시뮬레이션과 평가의 readable Markdown 전체를 펼쳐 읽을 수 있다.
문서 보기와 원문 보기를 전환할 수 있으며 HTML 태그는 실행하지 않고 텍스트로 표시한다.

## 검증

기존 개발 환경에서 다음을 실행한다. 임시 결과 파일만 사용하는 테스트다.

```bash
.venv/bin/python -m pytest -q demo/tests
node --check demo/static/app.js
```

독립 환경에서 테스트하려면 pytest와 httpx가 추가로 필요하다.
기존 실험 환경에는 설치하지 말고 독립 환경에서만 설치한다.

```bash
demo/.venv/bin/python -m pip install -c requirements.lock pytest==9.1.1 httpx==0.28.1
demo/.venv/bin/python -m pytest -q demo/tests
```

브라우저 확인 항목:

1. 에피소드를 선택하고 두 에이전트·환경·관계 토글을 연다.
2. 음성을 재생하고 탐색한다. 대사를 클릭해 해당 시점과 강조 표시를 확인한다.
3. 재생 중 다른 에피소드를 선택해 이전 음성이 멈추는지 확인한다.
4. 여러 평가 경로를 추가하고 새로고침한다. 점수표와 각 보고서, Markdown 원문을 확인한다.
5. 좁은 화면에서 페이지 전체가 가로로 넘치지 않는지 확인한다.

실제 예시 데이터로 브라우저 확인을 자동 실행하려면 별도 Playwright 설치를 사용할 수 있다.
시스템에 Chromium 실행에 필요한 라이브러리가 있어야 한다. 이 도구는 데모 실행에는 필요 없다.

```bash
npm install --prefix /tmp/talktopia-demo-browser playwright@1.58.2
/tmp/talktopia-demo-browser/node_modules/.bin/playwright install chromium
PLAYWRIGHT_MODULE=/tmp/talktopia-demo-browser/node_modules/playwright \
  node demo/tests/browser_smoke.cjs http://127.0.0.1:8765
```

이 검사는 완료된 에피소드가 3개 이상이고 첫 에피소드에 음성과 readable 문서가 있는 실행을 대상으로 한다.
예시 Pair 01의 450개 목록, 실제 음성 재생·탐색, 토글·보고서·빠른 선택·모바일 화면을 검사한다.
검증 설정에 `id: "duplex"`인 실제 Surface5 실행을 추가하면 첫 에피소드의 2.5초 지점에서
겹치는 발화가 강조되는지도 검사한다.

구현·검증·별도 Agent 리뷰가 완료된 기능은 `git merge --no-ff feature/results-demo`로
`dev`에 병합해 분기와 합류 이력을 남긴다.
