import random
import time

########## 캐릭터 및 몬스터 ##########

# 플레이어 기본 객체
class Player:
    def __init__(self, name, hp=0, attack=0, gold=0, exp=0, level=1, job=''):
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.attack = attack
        self.gold = gold
        self.exp = exp
        self.level = level
        self.job = job
        self.inventory = []

    def add_inventory(self, item):
        self.inventory.append(item)
    
    def show_status(self, exp_needed):
        bar_length = 20
        exp_percentage = self.exp / exp_needed
        hp_percentage = self.hp / self.max_hp
        filled = int(bar_length * exp_percentage)
        filled2 = int(bar_length * hp_percentage)
        empty = bar_length - filled
        empty2 = bar_length - filled2
        exp_bar = '■' * filled + '-' * empty
        hp_bar = '■' * filled2 + '-' * empty2
        print(f"\n======== {self.name}의 상태 ========")
        print(f"레벨: {self.level}")
        print(f"HP: {self.hp} / {self.max_hp}")
        print(f"[{hp_bar}] {(hp_percentage * 100):.2f}%")
        print(f"공격력: {self.attack}")
        print(f"보유 골드: {self.gold}")
        print(f"경험치: {self.exp}")
        print(f"[{exp_bar}] {(exp_percentage * 100):.2f}%")
        name_len = len(self.name) - 1
        tmp = ''
        for _ in range(name_len):
            tmp += '='
        print(f"=========================={tmp}\n")  # 양식 맞추기 위함

    def show_inventory(self):
        if not self.inventory:
            print("\n인벤토리가 비어 있습니다.")
            return False
        else:
            print("\n=== 인벤토리 ===")
            for idx, item in enumerate(self.inventory, 1):
                print(f"{idx}. {item.show_status()}")
            return True

# 직업 - 전사
class Warrior(Player):
    def __init__(self, name):
        super().__init__(name, hp=70, attack=12, job='Warrior')  # 전사는 HP가 높음

    def show_status(self):
        exp_needed = self.level * 15
        super().show_status(exp_needed)

    # 레벨업 로직
    def level_up(self):
        exp_needed = self.level * 15
        while self.exp >= exp_needed:
            self.exp -= exp_needed
            self.max_hp += self.level * 12
            self.attack += self.level * 3
            self.level += 1
            self.hp = self.max_hp  # 레벨업 시 HP를 가득 회복
            exp_needed += self.level * 15
            print(f"\n*** 축하합니다! 레벨이 {self.level}이(가) 되었습니다! ***")
            super().show_status(exp_needed)

# 직업 - 마법사
class Mage(Player):
    def __init__(self, name):
        super().__init__(name, hp=50, attack=15, job='Mage')  # 마법사는 공격력이 좀 더 높음

    def show_status(self):
        exp_needed = self.level * 15
        super().show_status(exp_needed)

    def level_up(self):
        exp_needed = self.level * 15
        while self.exp >= exp_needed:
            self.exp -= exp_needed
            self.max_hp += self.level * 7
            self.attack += self.level * 5
            self.level += 1
            self.hp = self.max_hp  
            exp_needed += self.level * 15
            print(f"\n*** 축하합니다! 레벨이 {self.level}이(가) 되었습니다! ***")
            super().show_status(exp_needed)

# 직업 - 궁수
class Archer(Player):
    def __init__(self, name):
        super().__init__(name, hp=60, attack=13, job='Archer')  # 궁수는 평균적인 느낌

    def show_status(self):
        exp_needed = self.level * 15
        super().show_status(exp_needed)

    def level_up(self):
        exp_needed = self.level * 15
        while self.exp >= exp_needed:
            self.exp -= exp_needed
            self.max_hp += self.level * 10
            self.attack += self.level * 4
            self.level += 1
            self.hp = self.max_hp 
            exp_needed += self.level * 15
            print(f"\n*** 축하합니다! 레벨이 {self.level}이(가) 되었습니다! ***")
            super().show_status(exp_needed)

# 직업 - 도적
class Bandit(Player):
    def __init__(self, name):
        super().__init__(name, hp=55, attack=14, job='Bandit')  # 도적은 레벨업이 더 빠름

    def show_status(self):
        exp_needed = self.level * 10
        super().show_status(exp_needed)

    def level_up(self):
        exp_needed = self.level * 10
        while self.exp >= exp_needed:
            self.exp -= exp_needed
            self.max_hp += self.level * 10
            self.attack += self.level * 3
            self.level += 1
            self.hp = self.max_hp  
            exp_needed += self.level * 10
            print(f"\n*** 축하합니다! 레벨이 {self.level}이(가) 되었습니다! ***")
            super().show_status(exp_needed)

# 몬스터 기본 객체         
class Monster:
    def __init__(self, name, hp, attack, gold_reward, exp_reward):
        self.name = name
        self.hp = hp
        self.max_hp = hp
        self.attack = attack
        self.gold_reward = gold_reward
        self.exp_reward = exp_reward

    def show_status(self):
        bar_length = 20
        hp_percentage = self.hp / self.max_hp
        filled2 = int(bar_length * hp_percentage)
        empty2 = bar_length - filled2
        hp_bar = '■' * filled2 + '-' * empty2
        print(f"\n                                                            === {self.name} 상태 ===")
        print(f"                                                            HP: {self.hp} / {self.max_hp}")
        print(f"                                                            [{hp_bar}] {(hp_percentage * 100):.2f}%")
        print(f"                                                            공격력: {self.attack}")
        print("                                                            =====================\n")

# 몬스터 종류 리스트 (레벨 1)
monster_list = [
    # 레벨 3까지
    [
        Monster("달팽이", 8, 3, 2, 2),
        Monster("슬라임", 20, 5, 5, 5),
        Monster("리본돼지", 40, 8, 10, 12),
        Monster("주황버섯", 55, 12, 15, 17)
    ],
    # 레벨 6까지
    [
        Monster("이블아이", 70, 15, 20, 20),
        Monster("좀비버섯", 100, 20, 25, 25),
        Monster("아이언 호그", 140, 26, 30, 30),
        Monster("레이스", 200, 30, 40, 35)
    ],
    # 레벨 9까지
    [
        Monster("와일드보어", 260, 40, 40, 40),
        Monster("주니어 예티", 380, 51, 45, 45),
        Monster("다크 골렘", 450, 60, 50, 51),
        Monster("크림슨 발록", 565, 70, 60, 58)
    ],
    # 마지막 레벨 10
    [
        Monster("혼테일", 750, 100, 100, 160)
    ]
]

# 몬스터 종류 리스트 (레벨 2)
monster_list2 = [
    # 레벨 13까지
    [
        Monster("블루 슬라임", 500, 120, 60, 60),
        Monster("그린 머쉬맘", 800, 156, 70, 70),
        Monster("페어리", 900, 167, 80, 80),
        Monster("플라워 피쉬", 1000, 197, 90, 90)
    ],
    # 레벨 16까지
    [
        Monster("오렌지 버섯", 990, 180, 70, 70),
        Monster("루팡", 1280, 215, 85, 85),
        Monster("화이트 펑", 1450, 239, 95, 95),
        Monster("푸퍼즈", 1720, 307, 110, 110)
    ],
    # 레벨 19까지
    [
        Monster("주니어 루팡", 1560, 280, 85, 85),
        Monster("마노", 1890, 385, 100, 95),
        Monster("킹 슬라임", 2100, 400, 115, 110),
        Monster("주니어 발록", 2320, 445, 140, 130)
    ],
    # 마지막 레벨 20
    [
        Monster("자쿰", 3000, 500, 250, 315)
    ]
]

# 몬스터 종류 리스트 (레벨 3)
monster_list3 = [
    # 레벨 23까지
    [
        Monster("플라잉 아이즈", 2200, 480, 70, 70),
        Monster("스톤 골렘", 2720, 575, 75, 75),
        Monster("마시토", 3010, 600, 85, 85),
        Monster("캡틴 블랙슬라임", 3270, 630, 110, 100)
    ],
    # 레벨 26까지
    [
        Monster("화이트 웬디고", 3300, 635, 110, 90),
        Monster("블러드 하프", 3440, 700, 130, 110),
        Monster("다크 예티", 3580, 745, 150, 120),
        Monster("레드 와이번", 3920, 850, 200, 130)
    ],
    # 레벨 29까지
    [
        Monster("블러드 드래곤", 4500, 755, 220, 130),
        Monster("다크 와이번", 5550, 790, 230, 150),
        Monster("네크로스", 6600, 830, 260, 170),
        Monster("카파 드레이크", 7650, 870, 330, 200)
    ],
    # 마지막 레벨 30
    [
        Monster("핑크빈", 10000, 850, 10000, 10000)
    ]
]

########## 아이템 ##########

# 아이템 기본 객체
class Item:
    def __init__(self, name, effect, price):
        self.name = name
        self.effect = effect  # 공격력, 방어력 등 효과
        self.price = price
    
    def show_status(self):
        return f"{self.name} 가격: {self.price}골드, 효과: {self.effect}"

# 아이템 - 무기
class Weapon(Item):
    def __init__(self, name, attack_increase, price):
        super().__init__(name, attack_increase, price)

    def show_status(self):
        return f"{self.name} 가격: {self.price}골드, 효과: {self.effect}만큼 공격력 증가"      

# 아이템 - 방어구
class Armor(Item):
    def __init__(self, name, defense_increase, price):
        super().__init__(name, defense_increase, price)

    def show_status(self):
        return f"{self.name} 가격: {self.price}골드, 효과: {self.effect}만큼 최대 체력 증가"

# 아이템 - 포션
class Potion(Item):
    def __init__(self, name, hp_increase, price):
        super().__init__(name, hp_increase, price)

    def show_status(self):
        return f"{self.name} 가격: {self.price}골드, 효과: {self.effect}만큼 체력 회복" 

# 직업별 아이템 종류 리스트(무기구)
job_items_weapon = {
    'Warrior': [
        Weapon("낡은 단검", 4, 20),
        Weapon("일반 칼", 15, 75),
        Weapon("검사의 칼", 40, 225),
        Weapon("화염의 검", 115, 700),
        Weapon("장월도", 225, 1490),
        Weapon("엑스칼리버", 545, 3150)
    ],
    'Mage': [
        Weapon("오래된 지팡이", 5, 20),
        Weapon("일반 지팡이", 18, 75),
        Weapon("마법사의 지팡이", 52, 225),
        Weapon("썬더블레이드", 125, 700),
        Weapon("마스터완드", 255, 1490),
        Weapon("라이넬 완드", 655, 3150)
    ],
    'Archer': [
        Weapon("낡은 활", 4, 20),
        Weapon("일반 활", 15, 70),
        Weapon("궁수의 활", 40, 225),
        Weapon("석궁", 115, 700),
        Weapon("화염의 활", 225, 1400),
        Weapon("아모스의 활", 545, 3130)
    ],
    'Thief': [
        Weapon("낡은 표창", 4, 18),
        Weapon("초보 도적의 표창", 18, 65),
        Weapon("월비 표창", 50, 225),
        Weapon("화랑 표창", 115, 625),
        Weapon("뇌전 수리검", 255, 1470),
        Weapon("플레임 표창", 645, 3140)
    ]
}

# 공통 아이템 종류 리스트(방어구)
common_items_armor = [
    # 신발
    Armor("여행자의 장화", 210, 160),
    Armor("신속의 장화", 1025, 630),
    Armor("마법 부여된 장화", 2050, 1065),

    # 하의
    Armor("여행자의 다리 보호구", 210, 160),
    Armor("병사의 다리 보호구", 1025, 630),
    Armor("신수의 바지", 2050, 1065),

    # 상의
    Armor("여행자의 갑옷", 210, 160),
    Armor("병사의 갑옷", 1025, 630),
    Armor("전설의 갑옷", 2050, 1065),

    # 투구
    Armor("여행자의 투구", 210, 160),
    Armor("영웅의 투구", 1025, 630),
    Armor("황혼의 투구", 2050, 1065)
]

# 공통 아이템 종류 리스트(포션)
common_items_potion = [
    # 물약
    Potion("체력 물약", 20, 10),
    Potion("거대한 물약", 100, 45),
    Potion("더욱 거대한 물약", 300, 120),
    Potion("더더더더더욱 거대한 물약", 1000, 360)
]


########## 아이템 생성 ##########

# 직업별 아이템 생성
def create_product_for_job(player):
    job = player.job
    if job in job_items_weapon:
        return job_items_weapon[job]
    else:
        return [] # 아이템이 없을 경우

# 공통 아이템 생성(방어구)
def create_product_for_common_armor(player):
    return common_items_armor

# 공통 아이템 생성(포션)
def create_product_for_common_potion(player):
    return common_items_potion


########## 모험 시작, 몬스터 생성, 전투 ##########

# 몬스터 레벨별 랜덤 생성
def create_random_monster(player):
    level = player.level

    # 레벨 범위에 따른 몬스터 리스트 선택
    if level <= 10:
        monster_source = monster_list
    elif 11 <= level <= 20:
        monster_source = monster_list2
    elif 21 <= level <= 30:
        monster_source = monster_list3
    else:
        return None

    # 레벨에 따라 몬스터 그룹에서 선택
    if level <= 3 or (11 <= level <= 13) or (21 <= level <= 23):
        return random.choice(monster_source[0])
    elif level <= 6 or (14 <= level <= 16) or (24 <= level <= 26):
        return random.choice(monster_source[1])
    elif level <= 9 or (17 <= level <= 19) or (27 <= level <= 29):
        return random.choice(monster_source[2])
    else:
        return random.choice(monster_source[3])

# 전투 로직(승리 시 레벨업 여부, 패배 시 사망 여부)
def battle(player, monster):
    print(f"\n*** 야생의 {monster.name}이(가) 등장했다! ***")
    # 전투 루프
    while True:
        monster.show_status()
        player.show_status()

        print("1) 공격하기")
        print("2) 인벤토리 열기")
        print("3) 도망가기")

        choice = input("행동을 선택하세요: ")

        if choice == '1':
            # 플레이어 공격
            damage_to_monster = player.attack
            monster.hp -= damage_to_monster
            print(f"{player.name}의 공격! {monster.name}에게 {damage_to_monster}의 피해!")
            time.sleep(1)

            # 몬스터가 살아있는지 체크
            if monster.hp <= 0:
                print(f"\n{monster.name}을(를) 물리쳤다!")
                # 전리품(경험치 및 골드)
                player.gold += monster.gold_reward
                player.exp += monster.exp_reward
                print(f"{monster.gold_reward}골드와 {monster.exp_reward} 경험치를 획득했다!")
                # 레벨업 체크
                player.level_up()
                return True

            # 몬스터 공격
            damage_to_player = monster.attack
            player.hp -= damage_to_player
            print(f"{monster.name}의 공격! {player.name}은(는) {damage_to_player}의 피해를 입었다!")
            time.sleep(1)

            # 플레이어 체력 체크
            if player.hp <= 0:
                print(f"\n{player.name}이(가) 쓰러졌다...")
                print("게임 오버")
                return False  # 사망 시에만 False출력하여 루프 벗어나도록

        elif choice == '2':
            state = player.show_inventory()
            if state:
                use_item(player)
        elif choice == '3':
            print(f"\n{player.name}은(는) {monster.name}에게서 도망쳤다!")
            return True
        else:
            print("잘못된 입력입니다. 다시 선택하세요.")

# 모험 시작 로직(몬스터 생성 및 전투 시작)
def go_adventure(player):
    print("\n=== 모험을 떠납니다! ===")
    monster = create_random_monster(player)
    monster.hp = monster.max_hp
    result = battle(player, monster)
    return result  # True이면 계속, False이면 게임 종료


########## 상점 시스템 ##########

# 상점 로직
def shop(player):
    while True:
        items = create_product_for_job(player)
        common_items_armor = create_product_for_common_armor(player)
        common_items_potion = create_product_for_common_potion(player)

        print("\n구매할 아이템:")
        print("1. 무기")
        print("2. 방어구")
        print("3. 포션")
        print("0. 나가기")
        print()
        num = int(input("구매할 아이템 번호를 선택하세요: "))

        # 아이템을 구매할 아이템 묶음으로 전환..후 표시
        if num == 0:
            print("상점을 나갑니다.")
            break
        elif num == 1:
            items = items
        elif num == 2:
            items = common_items_armor
        elif num == 3:
            items = common_items_potion
        else:
            print("잘못된 선택입니다.")
            continue

        print()
        for idx, item in enumerate(items, 1):
            print(f"{idx}. {item.show_status()}")

        print("0. 나가기")
        print()
        choice = int(input("구매할 아이템 번호를 선택하세요: "))

        if choice == 0:
            print("구매를 취소합니다.")
            continue
        
        # 구매 로직
        if 1 <= choice <= len(items):
            selected_item = items[choice - 1]
            if not isinstance(selected_item, Potion):
                if selected_item in player.inventory:  # 무기와 방어구는 똑같은 상품 선택 불가, 포션은 가능
                    print("이미 구매하셨습니다! 다른 아이템을 선택해 주세요")
                    continue

            if player.gold >= selected_item.price:
                player.gold -= selected_item.price
                print(f"{selected_item.name}을(를) 구매했습니다!")
                if num == 1:
                    player.attack += selected_item.effect  # 공격력 증가
                    print(f"공격력이 {selected_item.effect}만큼 증가하였습니다!")
                elif num == 2:
                    player.max_hp += selected_item.effect  # 최대체력 증가
                    print(f"최대 체력이 {selected_item.effect}만큼 증가하였습니다!")
                player.add_inventory(selected_item)
            else:
                print("골드가 부족합니다.")
        else:
            print("잘못된 선택입니다.")

# 상점 가는 길 출력 로직
def go_shopping(player):
    message = "상점으로 가는 중입니다...."
    text = message.rstrip('.')
    dots = message[len(text):]
    print()
    print(text, end='', flush=True)
    
    for dot in dots:
        print(dot, end='', flush=True)
        time.sleep(0.5)

    print()
    print("\n=== 상점에 도착했습니다! ===")
    shop(player)
    player.show_inventory()

# 아이템 사용 로직
def use_item(player):
    while True:
        print()
        print("나가시려면 0번을 입력해주세요")
        index = int(input("사용할 아이템 번호를 입력하세요: "))
        print()
        if index == 0:
            return
        index -= 1
        if 0 <= index < len(player.inventory):
            item = player.inventory[index]
            if isinstance(item, Potion):
                player.hp = min(player.max_hp, player.hp + item.effect)
                player.inventory.pop(index)  # 아이템 삭제(포션)
                print(f"{item.name}을(를) 사용하여 체력이 {item.effect}만큼 회복되었습니다! (HP: {player.hp}/{player.max_hp})")
            elif isinstance(item, Weapon):  # 무기 및 방어구는 사용은 불가능(이미 장착됨)
                print(f"{item.name}은(는) 이미 장착되어 사용할 수 없습니다.")
            elif isinstance(item, Armor):
                print(f"{item.name}은(는) 이미 장착되어 사용할 수 없습니다.")
        else:
            print("잘못된 아이템 번호입니다.")


########## 미니게임 시스템 ##########

# 미니게임 시작 함수  
def start_minigame(player):
    while True:
        print("\n미니게임을 시작합니다!")
        print("아래에서 미니게임을 선택해 주세요!\n")

        print("1) 가위바위보 한판?")
        print("2) 운빨게임")
        print("0) 미니게임 종료")

        cmd = int(input("\n입력: "))
        if cmd == 0:
            print("\n미니게임에서 나갑니다.")
            break

        if cmd == 1:
            rock_scissors_paper(player)
        elif cmd == 2:
            are_you_lucky(player)

# 가위바위보 게임
def rock_scissors_paper(player):
    print("\n가위바위보를 시작합니다!")
    while True:
        print(f"\n현재 보유 골드: {player.gold}")
        money = int(input("배팅 금액을 걸어주세요! "))
        current_money = player.gold
        if money > current_money:
            print("현재 가지고 있는 돈보다 더 배팅할 수 없습니다! ")
            continue

        print("\n아래에서 하실 행동을 선택해 주세요")
        print("1. 주먹")
        print("2. 가위")
        print("3. 보자기")
        print("0. 가위바위보 나가기")

        action = int(input("입력: "))
        if action == 0:
            print("\n가위바위보에서 나갑니다.")
            break

        # 배팅금액 차감
        player.gold -= money
        betting_money = money * 2

        ai_action = random.choice([1,2,3])
        actions = {1: "주먹", 2: "가위", 3: "보자기"}
        print(f"\n플레이어 선택: {actions[action]}")
        print(f"컴퓨터 선택: {actions[ai_action]}")

        if action == ai_action:
            print("비겼습니다")
            player.gold += money 
        elif (action == 1 and ai_action == 2) or (action == 2 and ai_action == 3) or (action == 3 and ai_action == 1):
            print("축하합니다! 승리하셨습니다!")
            player.gold += betting_money
        else:
            print("졌습니다 ㅋ")

# 확률 배팅 게임
def are_you_lucky(player):
    print("\n확률 게임을 시작합니다!")
    while True:
        print(f"\n현재 보유 골드: {player.gold}")
        money = int(input("배팅 금액을 걸어주세요! "))
        current_money = player.gold
        if money > current_money:
            print("현재 가지고 있는 돈보다 더 배팅할 수 없습니다!")
            continue

        # 초기 배팅 금액 차감
        player.gold -= money
        total_reward = money  # 배당금 초기화

        stages = [2, 3, 4]  # 확률 단계
        for stage in stages:
            print(f"\n{stage}분의 1 확률 도전!")
            print("1) 도전하기")
            print("2) 포기하고 배당금 받기")
            choice = int(input("입력: "))

            if choice == 2:
                # 포기 시 보상 지급
                player.gold += total_reward
                print(f"\n포기를 선택하셨습니다! 배당금 {total_reward}골드를 획득하셨습니다.")
                return

            # 도전 진행
            print(f"{stage}분의 1 확률 도전 중...")
            time.sleep(1)
            lucky_number = random.randint(1, stage)
            player_choice = random.randint(1, stage) 

            if lucky_number == player_choice:
                # 성공 시 배당금 배수로 증가
                total_reward *= stage
                print(f"축하합니다! {stage}분의 1 확률을 성공하셨습니다!")
                print(f"현재 배당금: {total_reward}골드")
            else:
                print(f"\n아쉽습니다. {stage}분의 1 확률에 실패하셨습니다!")
                print(f"배팅 금액 {money}골드가 사라졌습니다.")
                return

        # 모든 단계 성공 시 보상 지급
        player.gold += total_reward
        print(f"\n모든 단계를 성공하셨습니다! 배당금 {total_reward}골드를 획득하셨습니다.")
        return


########## 메인 ##########

# 메인 함수
def main():
    print("=== milo의 즐거운 rpg 게임 ===")
    player_name = input("플레이어 이름을 입력하세요: ")
    print("직업을 선택하세요.")
    print("1) 전사")
    print("2) 마법사")
    print("3) 궁수")
    print("4) 도적")
    while True:
        job_choice = input("입력: ")

        job_classes = {    # 직업 선택에 따른 클래스 매핑
            '1': Warrior,
            '2': Mage,
            '3': Archer,
            '4': Bandit
        }
        if job_choice not in job_classes:
            print("잘못된 입력입니다. 다시 선택해 주세요.")
        else:
            player = job_classes[job_choice](player_name)
            break
    
    stage = 1
    print("\n*** 스테이지 1 : 리스 항구 ***")

    while True:
        if player.level > 10 and stage == 1:
            stage = 2
            print("\n*** 스테이지 2 : 엘리니아 ***")

        if player.level > 20 and stage == 2:
            stage = 3
            print("\n*** 스테이지 3 : 커닝시티 ***")

        if player.level > 30:
            print("축하합니다, 모든 스테이지를 클리어했습니다!")
            break

        print("\n메뉴를 선택하세요.")
        print("1) 모험을 떠난다")
        print("2) 플레이어 정보 확인")
        print("3) 인벤토리 확인")
        print("4) 상점 방문")
        print("5) 미니 게임")
        print("0) 게임 종료")
        cmd = input("입력: ")
        if cmd == '1':
            # 모험 시작
            result = go_adventure(player)
            if not result:  
                # result가 False면 플레이어 사망으로 게임 오버
                break
        elif cmd == '2':
            player.show_status()
        elif cmd == '3':
            state = player.show_inventory()
            if state:
                use_item(player)
        elif cmd == '4':
            go_shopping(player)
        elif cmd == '5':
            start_minigame(player)
        elif cmd == '0':
            print("게임을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다. 다시 입력해주세요.")

if __name__ == "__main__":
    main()
