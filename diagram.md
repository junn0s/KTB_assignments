```mermaid
flowchart TB
    A((시작)) --> B[메인 메뉴 표시]
    B --> C{메뉴 선택}
    C -->|1. 모험 떠나기| D{이벤트 발생?}
    D -->|몬스터 출현| E{전투 행동 선택?}
    E -->|공격| F[플레이어가 몬스터 공격]
    F --> G{몬스터 사망?}
    G -->|Yes| H[골드/경험치 획득<br/>및 레벨업 체크]
    H --> M[메인 메뉴로 복귀]
    G -->|No| I[몬스터가 플레이어 공격]
    I --> J{플레이어 사망?}
    J -->|Yes| X((게임 오버))
    J -->|No| E
    E -->|도망| M

    D -->|아무 일도 없음| M
    C -->|2. 정보 확인| K[플레이어 정보 표시]
    K --> M[메인 메뉴로 복귀]
    C -->|3. 게임 종료| Z((게임 종료))
    M --> B


> 주의: 코드 블록 시작과 끝의 ```(백틱)은 3개지만, GitHub Markdown 상에서 보이지 않도록 처리한 것입니다.

---

## 3. 마크다운 미리보기 열기

- 파일을 저장한 뒤, VSCode 좌측 상단 메뉴에서 **View** → **Command Palette** (또는 `Ctrl+Shift+P`)를 연 뒤  
- “**Markdown: Open Preview to the Side**”를 검색/선택하면,  
  - 마크다운 미리보기가 오른쪽에 열립니다.  
- 혹은 마크다운 파일에서 **우클릭** → **Open Preview** (또는 **Open Preview to the Side**)를 선택해도 됩니다.

만약 Mermaid 확장이 제대로 설치되어 있다면, 해당 코드 블록이 텍스트가 아닌 **다이어그램(Flowchart)** 형태로 렌더링되어 보일 것입니다.

---

## 4. 확장에 따라 설정 조정

일부 확장은 **추가 설정**을 요구하기도 합니다. 예를 들어 **Markdown Preview Enhanced**에서는 `settings.json`에서 Mermaid 관련 플래그를 켜야 할 수도 있습니다.  

확장 마켓플레이스 페이지나 GitHub 리포지토리에 안내된 방법을 참고해 설정을 조정하세요.

---

### 그 외 팁

- **Mermaid Live Editor** (온라인 도구)  
  - [https://mermaid.live/](https://mermaid.live/) 같은 사이트에서 코드를 붙여 넣으면 즉시 다이어그램을 확인할 수 있습니다.  
- **다른 마크다운 뷰어** 사용  
  - VSCode 외에도 **Typora** 같은 마크다운 편집기나 GitHub의 웹 인터페이스에서도 Mermaid가 지원됩니다.

이 과정을 거치면 VSCode에서 Mermaid 순서도를 마크다운 프리뷰로 쉽게 확인할 수 있습니다.  