# CarsRL - Unity ML-Agents (PPO)

Projekt polega na wytrenowaniu agentów AI sterujących samochodami na torze wyścigowym przy użyciu uczenia ze wzmocnieniem (Reinforcement Learning) w środowisku Unity ML-Agents.

**Autorzy:** Michał Nocek, Jakub Wawro, Wiktor Piwowar

---

# Spis treści

- [CarsRL - Unity ML-Agents (PPO)](#carsrl---unity-ml-agents-ppo)
- [Spis treści](#spis-treści)
- [1. Wprowadzenie](#1-wprowadzenie)
- [2. Podstawy teoretyczne](#2-podstawy-teoretyczne)
  - [2.1 Uczenie ze wzmocnieniem](#21-uczenie-ze-wzmocnieniem)
  - [2.2 Proces decyzyjny Markowa (MDP)](#22-proces-decyzyjny-markowa-mdp)
  - [2.3 Polityka (policy)](#23-polityka-policy)
    - [Schemat intuicyjny](#schemat-intuicyjny)
  - [2.4 Obserwacje, akcje, nagroda i kara](#24-obserwacje-akcje-nagroda-i-kara)
  - [2.5 PPO i TRPO](#25-ppo-i-trpo)
    - [Dlaczego TRPO było ważne?](#dlaczego-trpo-było-ważne)
    - [Co robi PPO?](#co-robi-ppo)
    - [Związek PPO z TRPO](#związek-ppo-z-trpo)
  - [2.6 PPO jako metoda Actor-Critic z siecią neuronową](#26-ppo-jako-metoda-actor-critic-z-siecią-neuronową)
    - [Intuicja działania](#intuicja-działania)
  - [2.7 Unity ML-Agents](#27-unity-ml-agents)
- [3. Opis projektu](#3-opis-projektu)
  - [3.1 Środowisko](#31-środowisko)
  - [3.2 Modele 3D wykonane w Blenderze](#32-modele-3d-wykonane-w-blenderze)
  - [3.3 Tory i przeszkody](#33-tory-i-przeszkody)
    - [Tory](#tory)
      - [Widok torów z oddalenia](#widok-torów-z-oddalenia)
    - [Przeszkody](#przeszkody)
    - [Nawierzchnia lodowa](#nawierzchnia-lodowa)
  - [3.4 Obserwacje i akcje agenta](#34-obserwacje-i-akcje-agenta)
    - [Obserwacje](#obserwacje)
    - [Akcje](#akcje)
  - [3.5 System nagród](#35-system-nagród)
    - [Najważniejsze sygnały nagrody](#najważniejsze-sygnały-nagrody)
    - [Dlaczego to działa?](#dlaczego-to-działa)
- [4. Konfiguracja i przebieg treningu](#4-konfiguracja-i-przebieg-treningu)
  - [4.1 Plik konfiguracyjny](#41-plik-konfiguracyjny)
  - [4.2 Proces treningu](#42-proces-treningu)
  - [4.3 Wykresy uczenia](#43-wykresy-uczenia)
  - [4.4 Opis kluczowych fragmentów kodu](#44-opis-kluczowych-fragmentów-kodu)
- [5. Przykładowe nagrania](#5-przykładowe-nagrania)
    - [Nagranie 1 - zbyt wolna jazda](#nagranie-1---zbyt-wolna-jazda)
    - [Nagranie 2 - dojechanie auta do końca toru](#nagranie-2---dojechanie-auta-do-końca-toru)
    - [Nagranie 3 - zablokowanie się na przeszkodzie](#nagranie-3---zablokowanie-się-na-przeszkodzie)
    - [Nagranie 4 - jazda po innym torze](#nagranie-4---jazda-po-innym-torze)
- [6. Napotkane problemy i rozwiązania](#6-napotkane-problemy-i-rozwiązania)
  - [Problem 1: Agent stał w miejscu](#problem-1-agent-stał-w-miejscu)
  - [Problem 2: Auta spawnowały się jedno na drugim](#problem-2-auta-spawnowały-się-jedno-na-drugim)
  - [Problem 3: Kolizje z podłożem kończyły epizod](#problem-3-kolizje-z-podłożem-kończyły-epizod)
  - [Problem 4: Agent ignorował checkpointy](#problem-4-agent-ignorował-checkpointy)
  - [Problem 5: Agent wywracał się na zakrętach](#problem-5-agent-wywracał-się-na-zakrętach)
  - [Problem 6: Agent nie wyrabiał na ostrych zakrętach](#problem-6-agent-nie-wyrabiał-na-ostrych-zakrętach)
- [7. Wnioski](#7-wnioski)
- [8. Bibliografia](#8-bibliografia)

---

# 1. Wprowadzenie

Celem projektu było przygotowanie agentów sterujących samochodami w środowisku wyścigowym oraz wytrenowanie ich tak, aby potrafiły samodzielnie poruszać się po torze, omijać przeszkody i zaliczać checkpointy w odpowiedniej kolejności. Do realizacji zadania wykorzystano uczenie ze wzmocnieniem oraz algorytm PPO, zaimplementowany w Unity ML-Agents.

W dokumentacji najpierw przedstawiono podstawy teoretyczne, następnie opisano środowisko, strukturę obserwacji i akcji, system nagród oraz przebieg treningu. Na końcu umieszczono przykładowe nagrania i omówienie problemów, które pojawiły się podczas pracy.

---

# 2. Podstawy teoretyczne

## 2.1 Uczenie ze wzmocnieniem

Uczenie ze wzmocnieniem (Reinforcement Learning, RL) to paradygmat uczenia maszynowego, w którym agent uczy się zachowania poprzez interakcję ze środowiskiem. W każdym kroku:

1. agent obserwuje stan środowiska,
2. podejmuje akcję,
3. otrzymuje nagrodę lub karę,
4. aktualizuje sposób podejmowania decyzji tak, aby w przyszłości maksymalizować sumaryczny zwrot.

W odróżnieniu od uczenia nadzorowanego, w RL nie podaje się poprawnych odpowiedzi „z góry”. Agent sam musi odkryć, które decyzje prowadzą do lepszego wyniku [9].

## 2.2 Proces decyzyjny Markowa (MDP)

Formalny opis wielu problemów RL przyjmuje postać **procesu decyzyjnego Markowa** (*Markov Decision Process*, MDP) - matematycznego modelu sekwencyjnego podejmowania decyzji w warunkach niepewności [6].

MDP opisuje się zwykle przez pięć elementów [6][8]:

- **S** - zbiór stanów lub obserwacji,
- **A** - zbiór akcji,
- **P(s' \| s, a)** - prawdopodobieństwo przejścia do nowego stanu,
- **R(s, a)** - funkcja nagrody,
- **γ** - współczynnik dyskontowania przyszłych nagród (w projekcie γ = 0.99).

Założenie Markowa oznacza, że to, co wydarzy się w kolejnym kroku, zależy wyłącznie od bieżącego stanu i bieżącej akcji, a nie od całej historii wstecz [4]. Dzięki temu agent może podejmować decyzje na podstawie samej aktualnej obserwacji.

## 2.3 Polityka (policy)

W RL **polityka** (*policy*) to strategia, którą agent stosuje, dążąc do realizacji swoich celów - reguła wiążąca stan środowiska z wyborem akcji [7][8]. Najprościej można ją opisać jako mapowanie:

**obserwacja → akcja**

Polityka odpowiada na pytanie: *„co agent powinien zrobić, widząc dany stan środowiska?”*

W ujęciu procesu decyzyjnego Markowa politykę zapisuje się zwykle jako funkcję π(s), która każdemu stanowi s ∈ S przyporządkowuje akcję (lub rozkład prawdopodobieństwa nad akcjami) zalecaną w tym stanie [8]. Celem uczenia jest znalezienie takiej polityki, która maksymalizuje oczekiwaną sumę (zdyskontowanych) nagród.

Można rozróżnić dwa podstawowe typy polityki [7]:

- **deterministyczną** - dla danego stanu zawsze zwraca tę samą akcję,
- **stochastyczną** - zwraca rozkład prawdopodobieństwa nad akcjami, a akcja jest losowana z tego rozkładu.

W praktyce PPO najczęściej uczy się korzystając z **polityki stochastycznej**, ponieważ ułatwia to eksplorację i stabilizuje trening [1].

### Schemat intuicyjny

```text
Środowisko → obserwacje → polityka → akcja → środowisko
                         ↑
                      nagroda
```

Polityka jest więc centralnym elementem całego układu. Celem uczenia jest znalezienie takiej polityki, która daje możliwie największą sumę nagród.

## 2.4 Obserwacje, akcje, nagroda i kara

W problemach RL występują cztery bardzo ważne pojęcia:

- **obserwacje** - informacje, które agent otrzymuje ze środowiska,
- **akcje** - decyzje podejmowane przez agenta,
- **nagroda** - dodatni sygnał wzmacniający pożądane zachowanie,
- **kara** - ujemny sygnał osłabiający niepożądane zachowanie.

Sygnał nagrody jest jedyną informacją zwrotną o jakości decyzji - agent dąży do maksymalizacji jego skumulowanej wartości w czasie [9].

W przypadku samochodu obserwacjami mogą być np. odległości od przeszkód, prędkość, kąt względem toru czy kontakt z lodem. Akcjami są np. skręt i gaz/hamulec. Nagroda może pojawić się za przejazd przez checkpoint, a kara za kolizję, utknięcie lub jazdę w złym kierunku.

Oprócz nagród zewnętrznych, agent korzysta z modułu ciekawości (*curiosity*) - dodatkowej nagrody za odkrywanie nowych stanów. Moduł uczy się przewidywać kolejny stan na podstawie obecnego i akcji; im trudniejszy do przewidzenia stan, tym wyższa nagroda intrinsiczna. Zapobiega to utknięciu agenta w bezpiecznych, ale nieproduktywnych zachowaniach (np. kręceniu się w kółko).

## 2.5 PPO i TRPO

**PPO** (*Proximal Policy Optimization*) jest algorytmem z rodziny metod policy gradient. Został zaproponowany jako prostsza i bardziej praktyczna alternatywa dla **TRPO** (*Trust Region Policy Optimization*). W oryginalnej publikacji autorzy podkreślają, że PPO zachowuje część zalet TRPO, ale jest łatwiejsze w implementacji i daje dobrą efektywność próbkowania. [1], [2]

### Dlaczego TRPO było ważne?

TRPO wprowadziło ideę ograniczania zbyt dużych zmian polityki w jednym kroku optymalizacji. Taki „obszar zaufania” miał poprawiać stabilność treningu i zmniejszać ryzyko, że nowa polityka stanie się wyraźnie gorsza od poprzedniej. Podejście to działa dobrze, ale jest obliczeniowo bardziej złożone i trudniejsze do wdrożenia. [2]

### Co robi PPO?

PPO korzysta z **clipped surrogate objective**, czyli funkcji celu, która ogranicza korzyść wynikającą ze zbyt dużej zmiany prawdopodobieństwa akcji. W publikacji [1] funkcję tę zapisuje się jako:

```text
L_CLIP(θ) = E[ min( r_t(θ) · Â_t ,  clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]
```

gdzie `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` to stosunek prawdopodobieństw akcji według nowej i starej polityki, `Â_t` to oszacowana przewaga (advantage), a `ε` to parametr przycinania (w projekcie ε = 0.2). W praktyce zamiast twardo rozwiązywać skomplikowane ograniczenie optymalizacyjne (jak TRPO), PPO „przycina” zbyt duże aktualizacje. Dzięki temu trening jest prostszy, a jednocześnie nadal stabilny. [1]

W uproszczeniu:
- gdy aktualizacja jest mała, algorytm działa normalnie,
- gdy zmiana polityki staje się zbyt duża, funkcja celu ją ogranicza,
- dzięki temu agent nie „przeskakuje” gwałtownie do dużo gorszego zachowania.

### Związek PPO z TRPO

PPO można traktować jako praktyczne uproszczenie idei stojącej za TRPO:
- **TRPO**: silniej opiera się na formalnym ograniczeniu kroku aktualizacji,
- **PPO**: zamiast tego używa prostszej funkcji celu z clippingiem.

To właśnie dlatego PPO jest tak popularne w zastosowaniach praktycznych: zachowuje stabilność treningu, ale nie wymaga tak ciężkiej optymalizacji jak TRPO. [1], [2]

## 2.6 PPO jako metoda Actor-Critic z siecią neuronową

W praktycznych implementacjach PPO najczęściej stosuje się układ **Actor-Critic**. Oznacza to, że jedna sieć lub dwa współpracujące moduły pełnią różne role:

- **Actor** - proponuje akcję na podstawie obserwacji,
- **Critic** - ocenia, jak dobra jest dana sytuacja albo dana decyzja.

W projekcie właśnie dlatego pojawia się sieć neuronowa: PPO nie działa „sam z siebie”, tylko potrzebuje modelu aproksymującego politykę i zwykle również funkcję wartości. W Unity ML-Agents agent przekazuje obserwacje do polityki, a polityka zwraca akcje; równolegle oceniana jest jakość decyzji przez sygnał wartości. [1], [10], [11]

### Intuicja działania

- Actor odpowiada: „jak skręcić i ile dodać gazu?”
- Critic odpowiada: „czy ten stan jest korzystny, czy nie?”
- PPO aktualizuje oba elementy na podstawie doświadczeń zebranych podczas interakcji ze środowiskiem.

Dzięki temu agent nie tylko uczy się, *co zrobić*, ale również *jak ocenić, czy dana decyzja była dobra*.

W tym projekcie sieć przyjmuje na wejściu 25 obserwacji i przepuszcza je przez trzy w pełni połączone warstwy ukryte po 256 neuronów (parametry `num_layers: 3`, `hidden_units: 256` z pliku konfiguracyjnego), a na wyjściu zwraca 2 ciągłe akcje (skręt oraz gaz/hamulec):

```text
Wejście (25 obserwacji)
        │
   [ Dense 256 ]  (warstwa ukryta 1)
        │
   [ Dense 256 ]  (warstwa ukryta 2)
        │
   [ Dense 256 ]  (warstwa ukryta 3)
        │
   Wyjście (2 akcje: skręt, gaz/hamulec)
```

Włączona normalizacja wejść (`normalize: true`) jest tu istotna, ponieważ obserwacje mają różne skale (np. 0..1 dla raycastów i -1..1 dla kątów), a bez normalizacji trening byłby mniej stabilny.

## 2.7 Unity ML-Agents

Unity ML-Agents to pakiet łączący środowisko Unity z uczeniem maszynowym. Zgodnie z dokumentacją Unity:
- agent generuje obserwacje,
- podejmuje akcje,
- otrzymuje nagrody,
- a zachowanie opisuje się przez **Behavior**, który może być uczony, heurystyczny lub inferencyjny. [3]

W praktyce trening wygląda następująco:

1. środowisko uruchamia się w Unity,
2. agent zbiera obserwacje,
3. Pythonowy trener ML-Agents odbiera dane,
4. algorytm PPO liczy aktualizację polityki,
5. model jest zapisywany jako checkpoint,
6. podczas treningu mogą być tworzone logi do TensorBoard. [3]

Takie podejście pozwala trenować złożone zachowania bez ręcznego programowania każdego ruchu samochodu.

---

# 3. Opis projektu

## 3.1 Środowisko

Projekt obejmuje wyścigowe środowisko 3D, w którym kilka samochodów może jednocześnie uczyć się jazdy po torze. Agent porusza się po mapie, reaguje na przeszkody, checkpointy i zmienne warunki nawierzchni.

Środowisko zostało przygotowane tak, aby wymuszać:
- jazdę po wyznaczonej trasie,
- omijanie przeszkód,
- stabilne pokonywanie zakrętów,
- reagowanie na lód i inne auta.

## 3.2 Modele 3D wykonane w Blenderze

Modele samochodów, przeszkód oraz elementów toru zostały wykonane samodzielnie w programie **Blender**, a następnie wyeksportowane do formatu **FBX**. Tory zostały wygenerowane proceduralnie przez podzielenie toru na małe odcinki (np. ostry skręt, przeszkoda, lód itd.) i złączenie ich poprzez skrypt w Blenderze wzdłuż krzywej. Dzięki temu możliwe było pełne dostosowanie geometrii do potrzeb projektu oraz przygotowanie środowiska treningowego zgodnie z założeniami eksperymentu. 

## 3.3 Tory i przeszkody

Projekt zawiera trzy tory o różnym stopniu trudności oraz proceduralny system rozmieszczania przeszkód.

### Tory

**Tor główny** - tor właściwy. Jest bardziej wymagający, ma węższe przejazdy, ostrzejsze zakręty i większą liczbę przeszkód.

**Tor nr 2 z lodem** - Charakteryzuje się szerszym pasem ruchu, łagodniejszymi zakrętami i mniejszą liczbą przeszkód. Został użyty do nauczenia agenta podstaw jazdy i zaliczania checkpointów.

**Tor nr 3 z lodem** - Charakteryzuje się dużą ilością prostych odcinków, przeszkodami na zakrętach i lodem.

#### Widok torów z oddalenia

Poniższe zrzuty ekranu pokazują tory z większej odległości, dzięki czemu widać ogólny układ mapy.

- **Tor główny - widok z oddalenia:**
<img width="898" height="699" alt="tor_glowny" src="https://github.com/user-attachments/assets/9fdf78bc-263c-4720-a8e9-d80d44f77cad" />

- **Tor nr 2 z lodem - widok z oddalenia:**
<img width="904" height="818" alt="tor_2" src="https://github.com/user-attachments/assets/c374d436-a8c8-42fd-852d-a97eacbd57e1" />

- **Tor nr 3 z lodem - widok z oddalenia:**
<img width="1468" height="785" alt="tor_3" src="https://github.com/user-attachments/assets/4f29fb5f-b351-4d46-825d-5fc1e12f12cf" />

### Przeszkody

W środowisku występują dwa główne typy przeszkód:
- **Opony** (`opona.fbx`) - tworzą slalomy, zwężenia i miejsca wymagające precyzyjnego manewrowania, ograniczają wyjazd poza tor,
- **Przeszkody** (`przeszkoda.fbx`) - statyczne obiekty blokujące fragmenty toru.

Rozmieszczenie przeszkód nie jest wykonywane ręcznie. Skrypt `oponygenerator.cs` automatycznie generuje obiekty na podstawie znaczników zapisanych w plikach map. Dzięki temu zmiana układu toru sprowadza się do przesunięcia znaczników i ponownego wygenerowania obiektów.

### Nawierzchnia lodowa

Na obu torach znajdują się fragmenty nawierzchni z materiałem `Ice.physicMaterial`. Lód znacznie obniża przyczepność, przez co samochód:
- wolniej hamuje,
- łatwiej wpada w poślizg,
- trudniej skręca na zakrętach.

Agent rozpoznaje lód przez osobną obserwację, która informuje, ile kół znajduje się na śliskiej nawierzchni.

## 3.4 Obserwacje i akcje agenta

### Obserwacje

Agent korzysta z zestawu 25 obserwacji. Obejmują one:
| # | Opis | Zakres |
|---|------|--------|
| 0–8 | Raycasts na ściany/przeszkody (kąty: -70°, -45°, -25°, -10°, 0°, 10°, 25°, 45°, 70°) | 0..1 (0 = tuż przy przeszkodzie) |
| 9–13 | Raycasts na inne auta (kąty: -60°, -25°, 0°, 25°, 60°) | 0..1 |
| 14 | Prędkość do przodu ze znakiem (ujemna = cofanie) | -1..1 |
| 15 | Prędkość bezwzględna znorm. | 0..1 |
| 16 | Kąt do następnego checkpointa | -1..1 |
| 17 | Dot product kierunku jazdy i kierunku do CP | -1..1 |
| 18–19 | Lokalna pozycja CP (x, z) | -1..1 |
| 20 | Dystans do CP | 0..1 |
| 21–22 | Kąt i dot do lookahead CP (+2 checkpoint do przodu) | -1..1 |
| 23 | Kontakt z lodem | 0..1 |
| 24 | Yaw rate (prędkość kątowa) | -1..1 |

Dzięki temu agent nie „widzi” całego świata, tylko dostaje skoncentrowany zestaw informacji potrzebnych do prowadzenia samochodu.

### Akcje

Agent steruje dwiema ciągłymi wartościami:
- **skręt**,
- **gaz / hamulec**.

Taki układ dobrze pasuje do symulacji samochodu, ponieważ ruch nie jest dyskretny, tylko płynny.

```csharp
float gas   = Mathf.Max(0f, _currentThrottle);   // [0, 1]
float brake = Mathf.Max(0f, -_currentThrottle);   // [0, 1]
```

Dzięki temu wartość `0` oznacza neutralną pozycję, a agent może samodzielnie przechodzić między przyspieszaniem i hamowaniem.

## 3.5 System nagród

W projekcie zastosowano system nagród oparty na zasadzie **zero tolerancji dla złych zachowań**. Główny cel agenta to ukończenie trasy możliwie szybko i bez kolizji.

### Najważniejsze sygnały nagrody

| Zdarzenie | Nagroda | Znaczenie |
|-----------|---------|-----------|
| Zaliczenie checkpointa | +3.0 | Główna motywacja do jazdy po trasie |
| Ukończenie całej trasy | +15.0 | Duży bonus za poprawny przejazd |
| Zbliżanie się do checkpointa | +0.05 za metr | Ciągłe wzmacnianie ruchu we właściwym kierunku |
| Szybka jazda w dobrym kierunku | +0.01/krok | Motywacja do płynnej jazdy |
| Każdy krok (existence penalty) | -0.002 | Kara za bezczynność, wymusza szybkość |
| Jazda zbyt wolno (< 2 m/s) | kara eskalująca -0.01·(1+t)/krok | Ogranicza „stanie w miejscu” |
| Cofanie | -0.01/krok | Eliminuje cofanie jako strategię |
| Kolizja (ściana / przeszkoda / auto) | -1.0 i koniec epizodu | Natychmiastowy reset |
| Wywrócenie się | -1.0 i koniec epizodu | Kara za utratę kontroli |
| Stuck przez dłuższy czas (> 1.5 s) | -1.5 i koniec epizodu | Silniejsza kara niż za kolizję |

Wszystkie powyższe wartości pochodzą bezpośrednio z pól skryptu `CarAgent.cs` (`checkpointReward = 3.0`, `trackCompleteReward = 15.0`, `collisionPenalty = -1.0`, `stuckPenalty = -1.5` itd.).

Kluczowy jest tu balans między karą za utknięcie a karą za kolizję. Stanie w miejscu przez 1.5 s kosztuje łącznie około **-2.4** (existence ≈ -0.15, low-speed ≈ -0.75, stuck -1.5), podczas gdy pojedyncza kolizja kosztuje tylko **-1.0**. Dzięki temu agentowi zawsze bardziej opłaca się próbować jechać (ryzykując zderzenie) niż biernie czekać.

### Dlaczego to działa?

Agent bardzo szybko odkrywa, że:
- stanie w miejscu jest kosztowne (kara eskaluje w czasie),
- cofanie nie rozwiązuje problemu,
- utknięcie (-2.4) jest mniej opłacalne niż nawet kolizja (-1.0), więc lepiej jechać niż stać,
- przejechanie checkpointa daje wyraźnie wyższą nagrodę.

W efekcie uczy się zachowania podobnego do ludzkiego kierowcy: przyspiesza na prostych, zwalnia przed zakrętami i unika przeszkód.

## 3.6 Użycie AI w projekcie
W projekcie użyto generatywnego AI w celach naprawy błędów w kodzie i w środowisku Unity. Wykorzystane zostało również przy doborze hiperparametrów i parametrów Reward/Penalty.
Wykorzystane modele: Claude Opus 4.7 | Gemma 4 31b:cloud | Qwen3 Coder:local

---

# 4. Konfiguracja i przebieg treningu

## 4.1 Plik konfiguracyjny

Poniżej pokazano przykładową konfigurację treningu:

```yaml
behaviors:
  CarDriver:
    trainer_type: ppo
    hyperparameters:
      batch_size: 2048
      buffer_size: 20480
      learning_rate: 3.0e-4
      num_epoch: 4
      epsilon: 0.2
      lambd: 0.95
      beta: 3.0e-2
    network_settings:
      hidden_units: 256
      num_layers: 3
      normalize: true
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
      curiosity:
        gamma: 0.99
        strength: 0.02       # mały wpływ — nie przesłania nagród zewnętrznych
        encoding_size: 128
    time_horizon: 256
    max_steps: 10000000
```

Najważniejsze parametry:
- `epsilon` - ogranicza zbyt duże zmiany polityki w PPO (clipping),
- `lambd` - parametr GAE,
- `beta` - wspiera eksplorację (entropia),
- `normalize` - pomaga, gdy obserwacje mają różne skale,
- `reward_signals.curiosity` - moduł ciekawości (intrinsic motivation) o niewielkiej sile (0.02), opisany w sekcji 2.4.

## 4.2 Proces treningu

Trening przebiegał w kilku etapach:

1. uruchomienie środowiska w Unity,
2. połączenie z trenerem ML-Agents w Pythonie,
3. zbieranie trajektorii z wielu agentów,
4. obliczanie przewag i aktualizacja polityki,
5. okresowe zapisywanie modelu,
6. podgląd wyników w TensorBoard.

Przykładowe polecenia:

```bash
mlagents-learn Assets/car_training.yaml --run-id=CarRun01
# wznowienie treningu:
mlagents-learn Assets/car_training.yaml --run-id=CarRun01 --resume
```

Podgląd logów:

```bash
tensorboard --logdir results/
```

## 4.3 Wykresy uczenia

- **Cumulative Reward** - czy agent faktycznie uczy się lepszej jazdy,
<img width="1140" height="504" alt="obraz" src="https://github.com/user-attachments/assets/a0a9db73-7470-4f74-89cc-1bcc81edec0f" />

- **Episode Length** - czy epizody stają się dłuższe i bardziej stabilne,
<img width="1140" height="504" alt="obraz" src="https://github.com/user-attachments/assets/46f508a0-b637-41e7-bbc9-93ad4bf95541" />

- **Policy Loss** - jak zachowuje się funkcja celu polityki,
<img width="1140" height="453" alt="obraz" src="https://github.com/user-attachments/assets/8a38b4cf-e284-4c39-998a-3070f49cfba7" />

- **Value Loss** - jakość estymacji funkcji wartości,
<img width="1140" height="453" alt="obraz" src="https://github.com/user-attachments/assets/0a35ac6b-be7c-4596-8d74-c9b6b4223a9a" />

- **Entropy** - stopień eksploracji.
<img width="1134" height="469" alt="Zrzut ekranu 2026-06-5 o 12 40 43" src="https://github.com/user-attachments/assets/f5aa5ac4-d0a1-4d31-a718-b06800d7abe5" />

## 4.4 Opis kluczowych fragmentów kodu

Logika agenta jest zaimplementowana głównie w dwóch skryptach: `CarAgent.cs` (obserwacje, akcje, nagrody, warunki końca epizodu) oraz `CarController.cs` (fizyka pojazdu oparta na `WheelCollider`). Poniżej omówiono najważniejsze fragmenty.

**Pętla decyzyjna agenta (`OnActionReceived`).** W każdym kroku fizyki agent najpierw sprawdza warunki zakończenia epizodu (przekroczenie czasu, wywrócenie, utknięcie), a następnie nalicza nagrody i kary. Kara za utknięcie jest celowo większa niż za kolizję:

```csharp
// Stuck detection — kara większa niż za kolizję
if (_car.SpeedMs < STUCK_SPEED)            // 0.5 m/s
{
    _stuckTimer += Time.fixedDeltaTime;
    if (_stuckTimer > STUCK_TIME)          // 1.5 s
    {
        AddReward(stuckPenalty);           // -1.5 (gorzej niż kolizja -1.0)
        EndEpisode();
        return;
    }
}
```

**Nagroda za postęp i jazdę w dobrym kierunku.** Agent jest stale wzmacniany za zbliżanie się do kolejnego checkpointa oraz za szybką jazdę w jego stronę:

```csharp
// Progress reward — +0.05 za każdy metr zbliżenia do checkpointa
float distDelta = _previousDistToCheckpoint - currentDist;
AddReward(distDelta * progressRewardScale);

// Speed × kierunek — premia za szybką jazdę w stronę CP
float dot = Vector3.Dot(transform.forward, toCP);
float speedNorm = Mathf.Clamp01(_car.ForwardSpeed / _car.maxSpeed);
if (dot > 0f && speedNorm > 0.05f)
    AddReward(dot * speedNorm * speedRewardScale);   // +0.01
```

**Obsługa kolizji — „zero tolerancji”.** Każda kolizja ze ścianą, przeszkodą lub innym autem natychmiast kończy epizod. Kolizje z podłożem są ignorowane (rozpoznawane po tagu lub po kierunku normalnej kontaktu skierowanej w górę):

```csharp
private void OnCollisionEnter(Collision col)
{
    if (_episodeEnding) return;
    if (IsGroundCollision(col)) return;   // podłoże ignorujemy

    if (IsOtherCarCollision(col) ||
        col.gameObject.CompareTag("Obstacle") ||
        col.gameObject.CompareTag("Wall"))
    {
        AddReward(collisionPenalty);       // -1.0
        EndEpisode();
    }
}
```

**Fizyka pojazdu (`CarController.cs`).** Samochód korzysta z czterech komponentów `WheelCollider` (napęd 4WD), zamiast prostego `Rigidbody` z colliderem pudełkowym. Dzięki temu uwzględniane są zawieszenie, trakcja wzdłużna i siła boczna, co daje realistyczne i przewidywalne zachowanie. Najważniejsze parametry (wartości z kodu):
- moment obrotowy `motorTorque = 3500` malejący kwadratowo z prędkością (przy prędkości maksymalnej spada do ~10%),
- prędkość maksymalna `maxSpeed = 25` m/s,
- maksymalny kąt skrętu `maxSteeringAngle = 35°`, redukowany o 40% przy wysokiej prędkości (`speedSteerReduction = 0.4`),
- docisk aerodynamiczny rosnący z prędkością (`downforceCoefficient = 8`),
- środek masy obniżony o 0.5 m (`centerOfMassYOffset = -0.5`) dla większej stabilności.

**Wykrywanie lodu.** Metoda `GetSurfaceContactRatio()` sprawdza, ile z czterech kół styka się z nawierzchnią lodową, i zwraca proporcję (0, 0.25, 0.5, 0.75 lub 1.0). Wynik trafia do obserwacji nr 23, dzięki czemu agent „wie”, że znajduje się na śliskim odcinku.

---

# 5. Przykładowe nagrania

Pełny materiał wideo został podzielony na krótsze fragmenty. Poniżej każdy z nich jest opisany wraz z wyjaśnieniem, co dokładnie widać na danym nagraniu.

### Nagranie 1 - zbyt wolna jazda
Na nagraniu 1 widać, że agent porusza się wolno przez co nie dojeżdża do końca na czas i epizod się kończy.
[https://youtu.be/edlB-GY1ixQ]

### Nagranie 2 - dojechanie auta do końca toru
Na nagraniu 2 widać, że agent w pewnych momentach jedzie niepewnie, ale dojeżdża do samego końca toru. Dlaczego tak się dzieję? Pojawia się zatruwanie uczenia pojazdu we wcześniejszych epizodach poprzez kolizję aut lub ich bliskość względem siebie. Dlatego auto w pewnych momentach zwalnia choć nic przed nim nie ma.
[https://youtu.be/QArM2jbXj2c]

### Nagranie 3 - zablokowanie się na przeszkodzie
Na nagraniu 3 można zobaczyć, jak samochód zachowuje się gdy nie wyrobi na ostrym zakręcie. Agent staje przed przeszkodą i epizod się kończy przez idle state.
[https://youtu.be/tsjyHNGL0r0]

### Nagranie 4 - jazda po innym torze
Na nagraniu 4 widać wcześniejszą wersję agenta jadącą po innym torze niż główny.
[https://youtu.be/5jXMSAZTWbM]

---

# 6. Napotkane problemy i rozwiązania

## Problem 1: Agent stał w miejscu

**Objaw:** Na początku treningu agent prawie się nie ruszał.

**Rozwiązanie:**
- dodanie kary za każdy krok,
- kara za niską prędkość,
- nagroda za ciekawość i eksplorację,
- mocniejsza kara za utknięcie niż za kolizję.

## Problem 2: Auta spawnowały się jedno na drugim

**Objaw:** Przy dużej liczbie agentów auta pojawiały się w tych samych punktach.

**Rozwiązanie:** Wprowadzono system Grid Stagger z kontrolą zajętości pozycji przed spawnem.

## Problem 3: Kolizje z podłożem kończyły epizod

**Objaw:** Ruch po nawierzchni był błędnie traktowany jako kolizja.

**Rozwiązanie:** Dodano rozpoznawanie kolizji podłoża na podstawie tagu i normalnej kontaktu.

## Problem 4: Agent ignorował checkpointy

**Objaw:** Agent próbował jechać na skróty.

**Rozwiązanie:** Checkpointy muszą być zaliczane w odpowiedniej kolejności, a dodatkowe obserwacje typu lookahead pomagają planować zakręty wcześniej.

## Problem 5: Agent wywracał się na zakrętach

**Objaw:** Przy dużej prędkości auto traciło stabilność.

**Rozwiązanie:** Obniżono środek masy, zmniejszono skręt przy wyższych prędkościach i wprowadzono docisk.

## Problem 6: Agent nie wyrabiał na ostrych zakrętach
**Objaw:** Przy dużej prędkości auto nie miało czasu aby zareagować na raycasty ukazujące ostry skręt.

**Rozwiązanie:** Dodano obserwacje kierunku w którym jest checkpoint aby auto wiedziało że checkpoint jest z którejś ze stron i może być skręt.

---

# 7. Wnioski

Projekt pokazał, że uczenie ze wzmocnieniem jest skuteczną metodą trenowania agentów do prowadzenia pojazdów w złożonym środowisku 3D. Wytrenowany agent potrafi samodzielnie zaliczać checkpointy, unikać części przeszkód i kończyć trasę w poprawny sposób.

Najważniejsze obserwacje:
- PPO dobrze sprawdza się w zadaniach z ciągłymi akcjami,
- system nagród ma ogromny wpływ na jakość uczenia,
- odpowiednio przygotowane obserwacje są kluczowe,
- stabilność środowiska fizycznego ma duże znaczenie dla końcowego efektu.

Do poprawy pozostają przede wszystkim:
- zachowanie na lodzie,
- omijanie innych aut,
- generalizacja na inne tory.

---

# 8. Bibliografia

Numeracja odpowiada odwołaniom w tekście (np. [1]). Wszystkie źródła internetowe były dostępne w czerwcu 2026 r.

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. https://arxiv.org/pdf/1707.06347
2. Schulman, J., Levine, S., Moritz, P., Jordan, M. I., Abbeel, P. (2015). *Trust Region Policy Optimization*. arXiv:1502.05477. https://arxiv.org/abs/1502.05477
3. Unity Technologies. *ML-Agents Toolkit Overview* (v4.0). https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/ML-Agents-Overview.html
4. *Markov chain*. Wikipedia (wersja angielska). https://en.wikipedia.org/wiki/Markov_chain
5. *Łańcuch Markowa*. Wikipedia (wersja polska). https://pl.wikipedia.org/wiki/Łańcuch_Markowa
6. *Markov decision process*. Wikipedia (wersja angielska). https://en.wikipedia.org/wiki/Markov_decision_process
7. *What is Policy in Reinforcement Learning?*. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/what-is-policy-in-reinforcement-learning/
8. De Luca, G. *What Is a Policy in Reinforcement Learning?*. Baeldung on Computer Science (aktualizacja: 2025). https://www.baeldung.com/cs/ml-policy-reinforcement-learning
9. Nikolopoulou, K. (2023). *Easy Introduction to Reinforcement Learning*. Scribbr. https://www.scribbr.com/ai-tools/reinforcement-learning/
10. Baptista, J. (2024). *Actor-Critic Methods: SAC and PPO*. https://joel-baptista.github.io/phd-weekly-report/posts/ac/
11. *Is PPO a policy-based method or an actor-critic-based method?*. Artificial Intelligence Stack Exchange. https://ai.stackexchange.com/questions/43313/is-ppo-a-policy-based-method-or-an-actor-critique-based-method

