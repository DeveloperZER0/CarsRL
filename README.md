# Car AI — Unity ML-Agents (PPO)

Projekt polega na wytrenowaniu agentów AI sterujących samochodami na torze wyścigowym przy użyciu uczenia ze wzmocnieniem (Reinforcement Learning) w środowisku Unity ML-Agents.

**Autorzy:** Michał Nocek, Jakub Wawro, Wiktor Piwowar

---

## 1. Podstawy teoretyczne

### Uczenie ze wzmocnieniem (Reinforcement Learning)

Uczenie ze wzmocnieniem to paradygmat uczenia maszynowego, w którym agent uczy się podejmować decyzje przez interakcję ze środowiskiem. Agent obserwuje stan środowiska, wykonuje akcję, otrzymuje nagrodę (lub karę) i aktualizuje swoją politykę decyzyjną tak, by w przyszłości maximalizować skumulowaną nagrodę.

Formalnie problem opisuje się jako **Markowski Proces Decyzyjny (MDP)**:
- **S** — przestrzeń stanów (obserwacje agenta)
- **A** — przestrzeń akcji (skręt, gaz)
- **R(s, a)** — funkcja nagrody
- **γ** — współczynnik dyskontowania przyszłych nagród (tu: 0.99)

### Algorytm PPO (Proximal Policy Optimization)

PPO to algorytm klasy *policy gradient* opracowany przez OpenAI (Schulman et al., 2017). Zamiast bezpośrednio optymalizować politykę (co może prowadzić do destabilizujących skoków), PPO ogranicza każdą aktualizację tak, by nowa polityka nie odbiegała zbyt daleko od starej. Realizuje to przez tzw. **clipped surrogate objective**:

```
L_CLIP(θ) = E[ min( r_t(θ) · Â_t ,  clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]
```

gdzie `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` to stosunek prawdopodobieństw akcji według nowej i starej polityki, a `ε` (epsilon) ogranicza maksymalną zmianę (tu: 0.2).

**Dlaczego PPO dla samochodów?**
- Naturalna obsługa **ciągłych przestrzeni akcji** (skręt i gaz to liczby rzeczywiste, nie dyskretne wybory)
- Stabilność treningu — małe, bezpieczne aktualizacje polityki
- Efektywność przy **wielu równoległych agentach** (10–50 aut jednocześnie na scenie)
- Wbudowana obsługa GAE (Generalized Advantage Estimation, λ=0.95) — dokładniejsza ocena jakości akcji

### Curiosity (intrinsic motivation)

Oprócz nagród zewnętrznych, agent korzysta z modułu ciekawości (*curiosity*) — dodatkowej nagrody za odkrywanie nowych stanów. Moduł uczy się przewidywać kolejny stan na podstawie obecnego i akcji; im trudniejszy do przewidzenia stan, tym wyższa nagroda intrinsiczna. Zapobiega to utknięciu agenta w bezpiecznych, ale nieproduktywnych zachowaniach (np. kręceniu się w kółko).

```yaml
# car_training.yaml
reward_signals:
  extrinsic:
    gamma: 0.99
    strength: 1.0
  curiosity:
    gamma: 0.99
    strength: 0.02       # mały wpływ — nie przesłania nagród zewnętrznych
    encoding_size: 128
```

---

## 2. Architektura systemu

### Sieć neuronowa

```
Wejście (25 obserwacji)
         │
  ┌──────▼──────┐
  │  Dense 256  │  ReLU
  └──────┬──────┘
  ┌──────▼──────┐
  │  Dense 256  │  ReLU
  └──────┬──────┘
  ┌──────▼──────┐
  │  Dense 256  │  ReLU
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Output (2) │  tanh → [-1, 1]
  └─────────────┘
  [steer, throttle]
```

Normalizacja wejść (`normalize: true`) jest kluczowa — obserwacje mają różne skale (0..1 dla raycastów, -180..180 dla kątów) i bez normalizacji trening byłby niestabilny.

### Obserwacje agenta (25 wartości)

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

### Akcje (2 wartości ciągłe)

| # | Opis | Zakres |
|---|------|--------|
| 0 | Skręt (lewo/prawo) | -1..1 |
| 1 | Gaz (>0) lub hamulec (<0) | -1..1 |

Remapowanie throttle w `CarController.cs`:
```csharp
float gas   = Mathf.Max(0f, _currentThrottle);   // [0, 1]  — gaz
float brake = Mathf.Max(0f, -_currentThrottle);  // [0, 1]  — hamulec
```
Dzięki temu wartość 0 oznacza neutralną — auto toczy się bez oporu silnika, a agent musi aktywnie hamować. Upraszcza to przestrzeń akcji i pozwala agentowi uczyć się płynnej jazdy.

---

## 3. System nagród — filozofia "ZERO TOLERANCJI"

Kluczowe założenie: **każda kolizja natychmiast kończy epizod**. Agent nie ma szansy „przyzwyczaić się" do jazdy po ścianach.

### Balans kar i nagród

| Zdarzenie | Nagroda | Efekt |
|-----------|---------|-------|
| Zaliczenie checkpointa | +3.0 | Główna motywacja do jazdy po trasie |
| Ukończenie całej trasy | +15.0 | Mega bonus za pełny okrążenie |
| Zbliżanie się do CP (za każdy metr) | +0.05 | Ciągły sygnał kierunkowy |
| Szybka jazda w kierunku CP | +0.01/krok | Motywuje do przyspieszania |
| Każdy krok (existence penalty) | -0.002 | Wymusza szybkość, karze guzdranie |
| Jazda poniżej 2 m/s (eskaluje) | -0.01·(1+t)/krok | Im dłużej wolno, tym gorzej |
| Cofanie | -0.01/krok | Eliminuje cofanie jako strategię |
| Kolizja ze ścianą/przeszkodą/autem | -1.0 + EndEpisode | Natychmiastowy reset |
| Wywrócenie się | -1.0 + EndEpisode | |
| Stuck (stanie >1.5 s) | **-1.5** + EndEpisode | Gorsze niż kolizja! |

Kluczowy balans: stanie przez 1.5 s kosztuje łącznie ~-2.4 (existence + lowSpeed + stuck), podczas gdy kolizja kosztuje tylko -1.0. Agent **zawsze woli spróbować jechać** (ryzykując kolizję) niż stać w miejscu.

### Implementacja w kodzie

```csharp
// CarAgent.cs — OnActionReceived()

// Existence penalty — każdy krok jest "płatny"
AddReward(existencePenalty);  // -0.002

// Low speed penalty — eskaluje im dłużej agent jedzie wolno
if (_car.SpeedMs < lowSpeedThreshold && _episodeTimer > 0.5f)
{
    _lowSpeedTimer += Time.fixedDeltaTime;
    float escalation = 1f + _lowSpeedTimer;  // im dłużej, tym gorzej
    AddReward(lowSpeedPenalty * escalation);  // -0.01 * (1 + t)
}

// Progress reward — nagroda za zbliżanie się do checkpointa
float distDelta = _previousDistToCheckpoint - currentDist;
AddReward(distDelta * progressRewardScale);  // +0.05 za każdy metr

// Speed × Direction — nagroda za szybką jazdę w dobrym kierunku
float dot = Vector3.Dot(transform.forward, toCP);
float speedNorm = Mathf.Clamp01(_car.ForwardSpeed / _car.maxSpeed);
if (dot > 0f && speedNorm > 0.05f)
    AddReward(dot * speedNorm * speedRewardScale);  // +0.01
```

```csharp
// Kolizja — zero tolerancji
private void OnCollisionEnter(Collision col)
{
    if (_episodeEnding) return;
    if (IsGroundCollision(col)) return;  // podłoże ignorujemy

    if (IsOtherCarCollision(col) ||
        col.gameObject.CompareTag("Obstacle") ||
        col.gameObject.CompareTag("Wall"))
    {
        _episodeEnding = true;
        AddReward(collisionPenalty);  // -1.0
        EndEpisode();
    }
}
```

---

## 4. Przykładowe sytuacje z treningu

### Sytuacja 1: Prawidłowe zaliczenie checkpointa

Agent zbliża się do bramki pod właściwym kątem. Obserwacja nr 16 (kąt do CP) zbliża się do 0, obserwacja nr 17 (dot product) zbliża się do 1.0. Agent otrzymuje sygnał progress reward za każdy metr oraz +3.0 przy przejechaniu przez trigger bramki.

```
[CarAgent] CP reached: got=3, expecting=3, total=8
[CarAgent] ✓ CP 3 ZALICZONY! Reward +3.0. Next=4/8
```

### Sytuacja 2: Kolizja ze ścianą — błąd agenta

Na wczesnym etapie treningu agent często wchodzi w zakręt za szybko i uderza w barierkę. Epizod natychmiast się kończy z karą -1.0. Ponieważ agent nie ma szansy "odratować" sytuacji po kolizji, szybko uczy się, że takie zakręty się nie opłacają.

```
// Raycasts przed kolizją:
// Kąt 0°  (prosto): 0.85  — droga wolna
// Kąt -25°(lewo):  0.12  — ŚCIANA BLISKO!
// Kąt -45°(lewo):  0.05  — ŚCIANA BARDZO BLISKO!

// OnCollisionEnter wywołany → Reward -1.0 → EndEpisode()
```

Poprawne zachowanie po treningu: agent widząc niskie wartości lewostronnych raycastów zawczasu skręca w prawo i hamuje.

### Sytuacja 3: Utknięcie (stuck) — najgorszy scenariusz

Agent zablokowany między przeszkodami przez ponad 1.5 sekundy otrzymuje karę -1.5 — wyższą niż za kolizję. Mechanizm ten eliminuje strategię "stoję i czekam na lepszy moment", która blokowała wczesne wersje agenta.

```csharp
// Stuck detection w OnActionReceived()
if (_car.SpeedMs < STUCK_SPEED)  // 0.5 m/s
{
    _stuckTimer += Time.fixedDeltaTime;
    if (_stuckTimer > STUCK_TIME)  // 1.5 s
    {
        _episodeEnding = true;
        AddReward(stuckPenalty);   // -1.5 — GORSZE niż kolizja!
        EndEpisode();
        return;
    }
}
```

### Sytuacja 4: Ukończenie trasy

Po ukończeniu wszystkich checkpointów agent otrzymuje bonus +15.0 i epizod kończy się sukcesem. W pełni wytrenowany agent kończy trasę szybciej niż pozwala timeout (90 s), konsekwentnie unikając przeszkód.

```
[CarAgent] CP reached: got=7, expecting=7, total=8
[CarAgent] ✓ CP 7 ZALICZONY! Reward +3.0. Next=8/8
[CarAgent] ★★★ TRASA UKOŃCZONA! Reward +15.0. EndEpisode().
```

---

## 5. Napotkane problemy i rozwiązania

### Problem 1: Agent stał w miejscu i nie eksplorował

**Objaw:** Na początku treningu agent prawie się nie ruszał — zerowy ruch był "bezpieczną" strategią, bo nie generował kar za kolizje.

**Rozwiązanie:**
- Dodanie *existence penalty* (-0.002/krok) — każda sekunda bezczynności kosztuje
- Dodanie eskalującej *low speed penalty* — im dłużej agent jedzie wolno, tym większa kara
- Dodanie *curiosity reward* (strength: 0.02) — wewnętrzna motywacja do eksplorowania nowych stanów
- Kara za stuck (-1.5) wyższa niż za kolizję (-1.0) — agent woli ryzykować zderzenie niż stać

### Problem 2: Auta spawnowały się jedno na drugim

**Objaw:** Przy wielu agentach (30+) część aut spawnowała się w tej samej pozycji, co powodowało natychmiastowe kolizje i kończenie epizodów.

**Rozwiązanie:** Implementacja systemu **Grid Stagger** w `CheckpointManager.cs` — każde auto dostaje deterministyczne miejsce w siatce wzdłuż kierunku trasy + mały losowy jitter:

```csharp
private Vector3 GetGridSpawnPosition(int index, Vector3 basePos, Vector3 forward, Vector3 right)
{
    float forwardOffset = index * gridSpawnSpacing;       // 5 m między autami
    int lateralPattern = (index % 3) - 1;                // wzór: -1, 0, +1, -1, 0, +1...
    float lateralOffset = lateralPattern * gridSpawnLateralRange;

    return basePos + forward * forwardOffset + right * lateralOffset + jitter;
}
```

Dodatkowo przed każdym spawnem sprawdzana jest kolizja `OverlapSphere` — jeśli miejsce jest zajęte, agent próbuje kolejnej pozycji (max 40 prób), a na końcu używa bezpiecznego fallbacku.

### Problem 3: Kolizje z podłożem kończyły epizod

**Objaw:** `OnCollisionEnter` był wywoływany przez koła toczące się po nawierzchni, co błędnie interpretowano jako kolizję ze ścianą.

**Rozwiązanie:** Metoda `IsGroundCollision()` klasyfikuje kolizję jako podłoże gdy:
1. Obiekt ma tag `Ground` lub `Ice`, albo
2. Normalna kontaktu wskazuje w górę (dot z `Vector3.up` > 0.7)

```csharp
if (col.contactCount > 0)
{
    Vector3 avgNormal = Vector3.zero;
    for (int i = 0; i < col.contactCount; i++)
        avgNormal += col.GetContact(i).normal;
    avgNormal /= col.contactCount;

    if (Vector3.Dot(avgNormal, Vector3.up) > 0.7f)
        return true;  // to jest podłoże — ignoruj
}
```

### Problem 4: Agent ignorował checkpointy i "skracał" trasę

**Objaw:** Agent uczył się jechać prosto w stronę mety zamiast po wyznaczonej trasie, pomijając bramki.

**Rozwiązanie:**
- Checkpoint pominięty = nie jest zaliczany (wymagana kolejność)
- Dodanie **lookahead obserwacji** — agent widzi nie tylko kolejny checkpoint, ale też ten za nim, co pozwala mu wcześniej planować zakręty

```csharp
int lookaheadIndex = Mathf.Clamp(
    _nextCheckpointIndex + lookaheadCheckpointOffset,  // +2 do przodu
    0, cpCount - 1);
Vector3 laPos = checkpointManager.GetCheckpointPosition(lookaheadIndex);
sensor.AddObservation(Vector3.SignedAngle(transform.forward, toLA, Vector3.up) / 180f);
sensor.AddObservation(Vector3.Dot(transform.forward, toLA));
```

### Problem 5: Agent wywracał się na zakrętach

**Objaw:** Przy agresywnym skręcaniu przy wysokiej prędkości auto się wywracało.

**Rozwiązanie:**
- Obniżenie środka masy (`centerOfMassYOffset = -0.5`) w `CarController.cs`
- Redukcja kąta skrętu przy wyższej prędkości (`speedSteerReduction = 0.4`)
- Downforce rosnący kwadratowo z prędkością (`downforceCoefficient = 8`)
- Detekcja wywrócenia (flip timer 1.5 s) i natychmiastowy reset

---

## 6. Konfiguracja treningu

```yaml
behaviors:
  CarDriver:
    trainer_type: ppo
    hyperparameters:
      batch_size: 2048
      buffer_size: 20480
      learning_rate: 3.0e-4
      num_epoch: 4
      epsilon: 0.2        # clipping PPO — ogranicza zmianę polityki
      lambd: 0.95         # GAE lambda
      beta: 3.0e-2        # entropia — wysoka = dużo eksploracji
    network_settings:
      hidden_units: 256
      num_layers: 3
      normalize: true     # normalizacja obserwacji — kluczowe!
    time_horizon: 256
    max_steps: 10000000
```

Trening uruchamiamy komendą:

```bash
mlagents-learn Assets/car_training.yaml --run-id=CarRun01
# Wznowienie:
mlagents-learn Assets/car_training.yaml --run-id=CarRun01 --resume
```

Postęp treningu można śledzić w TensorBoard:

```bash
tensorboard --logdir results/
# http://localhost:6006
```

---

## 7. Opis środowiska

Projekt zawiera dwa tory o różnym stopniu trudności oraz proceduralny system rozmieszczania przeszkód.

### Tory

**mapa-easy** — tor wprowadzający. Szerokie pasy, łagodne zakręty, minimalna liczba przeszkód. Używany do wstępnego treningu agenta — pozwala mu nauczyć się podstawowego sterowania i zaliczania checkpointów zanim trafi na trudniejsze warunki.

**supereleganckitor** — tor właściwy. Węższe pasy, ostrzejsze zakręty, większa gęstość przeszkód. Na tym torze przeprowadzano główny trening i oceniano finalną jakość agenta.

### Przeszkody

Przeszkody na torze to dwa rodzaje obiektów:
- **Opony** (`opona.fbx`) — stosy opon ustawionych na torze, tworzące slalom lub zwężenia pasa
- **Przeszkody** (`przeszkoda.fbx`) — statyczne bloki blokujące fragmenty drogi

Rozmieszczenie przeszkód nie jest ręczne — skrypt `oponygenerator.cs` automatycznie generuje obiekty na podstawie znaczników (pustych GameObject'ów) zapisanych w pliku `LOSTAKIZLASU.fbx`. Znacznik zawierający w nazwie `opon` staje się oponą, zawierający `przeszkoda` — przeszkodą. Dzięki temu zmiana układu toru sprowadza się do przesunięcia znaczników i kliknięcia "Wygeneruj Wszystko" w Inspectorze.

### Nawierzchnia lodowa

Na obu torach znajdują się fragmenty nawierzchni z materiałem `Ice.physicMaterial` — znacznie niższa tarcie boczne i wzdłużne. Agent wykrywa lód przez obserwację nr 23 (współczynnik 0..1 ile kół dotyka lodu) i w teorii powinien reagować wcześniejszym hamowaniem oraz mniejszym skrętem. W praktyce reakcja na lód jest jednym ze słabszych aspektów wytrenowanego modelu.

---

## 8. Fizyka samochodu — WheelColliders

Zamiast prostego Rigidbody z colliderem w kształcie pudełka, samochód korzysta z czterech komponentów `WheelCollider` — dedykowanego rozwiązania Unity do symulacji kół pojazdów.

**Dlaczego WheelCollider, nie zwykły Rigidbody?**

Zwykły BoxCollider nie symuluje zachowania opony — nie ma pojęcia o poślizgu, trakcji ani sile bocznej. WheelCollider modeluje każde koło osobno z uwzględnieniem:
- zawieszenia (sprężyna + tłumik)
- trakcji wzdłużnej (przyspieszanie i hamowanie)
- siły bocznej (opór przy skręcaniu)

Dzięki temu agent otrzymuje realistyczne sprzężenie zwrotne — auto zachowuje się przewidywalnie przy różnych prędkościach i na różnych nawierzchniach, co znacznie przyspiesza uczenie.

**Napęd 4WD i redukcja momentu:**

```csharp
// CarController.cs — ApplyMotor()
float speedRatio  = Mathf.Clamp01(Mathf.Max(ForwardSpeed, 0f) / maxSpeed);
float torqueCurve = Mathf.Lerp(1f, 0.1f, speedRatio * speedRatio);
float torque      = gas * motorTorque * torqueCurve;

wheelFL.motorTorque = torque;  // 4WD — wszystkie koła
wheelFR.motorTorque = torque;
wheelRL.motorTorque = torque;
wheelRR.motorTorque = torque;
```

Moment obrotowy maleje kwadratowo wraz z prędkością — przy starcie pełna siła (3500 Nm), przy max prędkości 10% wartości. Naśladuje to charakterystykę prawdziwego silnika i zapobiega "rakietowemu" przyspieszeniu przy każdym epizodzie.

**Stabilizacja:**
- Środek masy obniżony o 0.5 m — auto trudniej się wywraca
- Downforce rosnący kwadratowo z prędkością — przy 25 m/s docisk ~5000 N
- Redukcja kąta skrętu przy wyższych prędkościach (o 40% przy v_max)

---

## 9. Lód — wpływ na zachowanie agenta

Fragment toru pokryty materiałem `Ice.physicMaterial` drastycznie zmienia właściwości jazdy:
- tarcie boczne spada z ~2.0 do ~0.1 — auto ślizga się w zakrętach
- hamowanie wydłuża się kilkukrotnie
- przy agresywnym skręcaniu auto wpada w poślizg i traci panowanie

Agent obserwuje kontakt z lodem przez obserwację nr 23:

```csharp
// CarAgent.cs — CollectObservations()
_iceContactRatio = _car.GetSurfaceContactRatio(iceMaterial, iceTag, iceLayers);
sensor.AddObservation(_iceContactRatio);  // 0 = sucho, 1 = wszystkie koła na lodzie
```

Metoda `GetSurfaceContactRatio()` sprawdza każde z czterech kół — jeśli WheelCollider dotyka collidera z materiałem lodu (lub tagiem `Ice`), koło jest zaliczane jako "na lodzie". Wynik to proporcja kół na lodzie (0.25, 0.5, 0.75 lub 1.0).

**Zachowanie agenta na lodzie:**

W praktyce agent nauczył się częściowo radzić z lodem — zwalnia przed wejściem w poślizg, gdy lewa strona raycastów wskazuje blisko ściany. Jednak nie wypracował optymalnej strategii: zdarza mu się wejść w zakręt za szybko i utracić kontrolę, co kończy epizod kolizją ze ścianą. Jest to jeden z wyraźnych obszarów gdzie dłuższy trening lub większa nagroda za płynną jazdę na lodzie mogłaby poprawić wyniki.

---

## 10. Wnioski

Projekt pokazał, że uczenie ze wzmocnieniem jest skuteczną metodą trenowania agentów do prowadzenia pojazdów — po dostatecznej liczbie kroków agent samodzielnie kończy trasę, omija przeszkody i reaguje na zmieniające się warunki. Jednocześnie ujawnił szereg ograniczeń i niedoskonałości.

**Co działa dobrze:**
- Agent konsekwentnie zalicza checkpointy w odpowiedniej kolejności
- Unika kolizji ze ścianami na prostych i łagodnych zakrętach
- Przy dłuższym treningu ukończenia trasy stają się regularne

**Co nadal sprawia problemy:**
- **Ostrze zakręty przy wysokiej prędkości** — agent czasem nie hamuje wystarczająco wcześnie i uderza w barierkę. Sieć nie przewiduje przyszłości, a lookahead +2 checkpointy nie zawsze wystarczają na bardzo ciasnych łukach.
- **Lód** — agent nie wypracował wyraźnie różnego zachowania na lodzie w porównaniu do suchej nawierzchni. Obserwacja o kontakcie z lodem jest dostępna, ale sieć nie nauczyła się jej w pełni wykorzystywać.
- **Omijanie innych aut** — przy wielu agentach na jednej scenie zdarzają się kolizje car-to-car, szczególnie przy starcie gdy auta są blisko siebie.
- **Generalizacja** — model wytrenowany na jednym torze słabo radzi sobie na zupełnie nowym układzie. RL uczy się konkretnej trasy, nie ogólnych reguł jazdy.

**Co można by zrobić lepiej:**
- Dłuższy trening (10M kroków to relatywnie mało dla złożonego toru)
- Curriculum learning — zaczynać od prostszej mapy, stopniowo wprowadzać trudniejsze warunki
- Większa sieć lub LSTM — pamięć sekwencyjna mogłaby pomóc przy planowaniu zakrętów
- Bardziej precyzyjne nagrody za jazdę na lodzie — osobny sygnał za utrzymanie kontroli przy poślizgu

---

## 11. Wymagania i setup

**Zależności:**
- Unity 2021.3 LTS lub nowsze
- ML-Agents Package 2.0+ (`com.unity.ml-agents`)
- Python 3.9 lub 3.10
- `pip install mlagents torch`

**Tagi Unity (wymagane):**
| Tag | Obiekt |
|-----|--------|
| `Agent` | Główny GameObject samochodu |
| `Wall` | Ściany/granice toru |
| `Obstacle` | Przeszkody na trasie |
| `Ground` | Nawierzchnia drogi |
| `Ice` | Śliska nawierzchnia |
