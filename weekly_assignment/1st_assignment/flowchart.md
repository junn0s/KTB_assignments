```mermaid
flowchart TB

%% ======== 메인 시작 ========
A((프로그램 시작)) --> B["플레이어 이름 & 직업 입력"]
B --> C["stage = 1 (리스 항구)"]

%% ======== 메인 루프 ========
C --> D{"플레이어 레벨 체크"}
D -->|"레벨 > 30"| X[["모든 스테이지 클리어 후 종료"]]
D -->|"레벨 > 20"| S3["stage = 3 (커닝시티)"]
D -->|"레벨 > 10"| S2["stage = 2 (엘리니아)"]
D -->|"그 외"| M["메인 메뉴 표시"]

M --> MN{"메뉴 선택"}
MN -->|"1) 모험"| ADV["go_adventure() 실행"]
MN -->|"2) 정보 확인"| ST["show_status() 실행"]
ST --> D

MN -->|"3) 인벤토리"| INV["show_inventory() 및 use_item()"]
INV --> D

MN -->|"4) 상점 방문"| GSH["go_shopping() 실행"]
MN -->|"5) 미니게임"| MGM["start_minigame() 실행"]
MN -->|"0) 종료"| X2[["게임 종료"]]

S2 --> D
S3 --> D

%% ======== 모험 & 전투 ========
subgraph "Adventure: go_adventure()"
direction TB
    ADV --> CR["create_random_monster() 호출"]
    CR --> BTL["battle() 호출"]
end

BTL --> BR{"전투 루프"}

BR -->|"플레이어 공격"| AttackMon["몬스터 HP 감소 처리"]
AttackMon --> CheckMonHP{"몬스터 HP <= 0 ?"}
CheckMonHP -->|"Yes"| Win["전리품 획득 및 level_up()"]
CheckMonHP -->|"No"| MonAttack["몬스터 공격 실행"]

MonAttack --> CheckPlayerHP{"플레이어 HP <= 0 ?"}
CheckPlayerHP -->|"Yes"| X3[["게임 오버"]]
CheckPlayerHP -->|"No"| BR

BR -->|"2) 인벤토리"| InvUse["show_inventory() 및 use_item()"]
InvUse --> BR

BR -->|"3) 도망"| D
Win --> D

%% ======== 상점 ========
subgraph "Shop: go_shopping() / shop()"
direction TB
    GSH --> SMsg["'상점으로 가는 중...' 출력"]
    SMsg --> SH["shop() 호출"]

    SH --> SC{"아이템 종류 선택: 무기, 방어구, 포션"}
    SC -->|"1,2,3"| BuyMenu["아이템 구매 로직 실행"]
    BuyMenu --> SC
    SC -->|"0"| EndShop["상점 종료"]
    EndShop --> ShowInv["player.show_inventory()"]
    ShowInv --> D
end

%% ======== 미니게임 ========
subgraph "Minigame: start_minigame()"
direction TB
    MGM --> MGChoice{"가위바위보, 운빨게임, 종료"}

    MGChoice -->|"가위바위보"| RSP["rock_scissors_paper() 실행"]
    RSP --> MGChoice

    MGChoice -->|"운빨게임"| Lucky["are_you_lucky() 실행"]
    Lucky --> MGChoice

    MGChoice -->|"종료"| EndMini["미니게임 종료"]
    EndMini --> D
end
