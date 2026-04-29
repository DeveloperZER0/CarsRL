# 🚗 Car AI - Unity ML-Agents PPO Setup

## Algorytm: PPO (Proximal Policy Optimization)
Wybrałem PPO bo:
- Jest domyślnym i najlepiej przetestowanym algorytmem w ML-Agents
- Świetnie radzi sobie z ciągłymi przestrzeniami akcji (skręt, gaz)
- Stabilny i przewidywalny w uczeniu
- Obsługuje wiele agentów równolegle (10-100 aut jednocześnie)

---

## 📦 Wymagania

```bash
pip install mlagents torch
```
- Unity 2021.3 LTS lub nowsze
- ML-Agents Package 2.0+ (w Package Manager: `com.unity.ml-agents`)
- Python 3.9 lub 3.10

---

## 🛠️ Setup w Unity - krok po kroku

### 1. Przygotuj paczkę ML-Agents
Window → Package Manager → Add by name → `com.unity.ml-agents`

### 2. Stwórz samochód (Prefab)
1. Stwórz GameObject "Car" z:
   - `Rigidbody` (Mass: 1000, Angular Drag: 0.05)
   - `CarController.cs`
   - `CarAgent.cs`
   - `BehaviorParameters` (dodaj auto przez Add Component)
   - `DecisionRequester` (Decision Period: 5)

2. W `BehaviorParameters` ustaw:
   - **Behavior Name:** `CarDriver` ← WAŻNE! Musi zgadzać się z YAML
   - **Vector Observations → Space Size:** `12`
   - **Actions → Continuous Actions:** `2`
   - **Actions → Discrete Actions:** `0`
   - **Behavior Type:** `Default` (podczas treningu) / `Heuristic` (ręczny test)

3. Stwórz 4 child objects dla kół z WheelCollider:
   - `WheelFL`, `WheelFR`, `WheelRL`, `WheelRR`
   - Każdy WheelCollider: Radius 0.35, Mass 20

4. Stwórz meshes kół (zwykłe cylindry) jako osobne children
5. Przypisz WheelCollider i Wheel Mesh do CarController

### 3. Stwórz trasę
1. Stwórz teren/tor z przeszkodami
2. **Tagi** - ustaw:
   - Samochód: tag `Agent`
   - Ściany/granice: tag `Wall`
   - Przeszkody: tag `Obstacle`
3. Przeszkody i ściany muszą mieć **Collider** (niekinematyczny)

### 4. Setup CheckpointManager
1. Stwórz pusty GameObject "CheckpointManager" → dodaj `CheckpointManager.cs`
2. Stwórz bramki wzdłuż trasy:
   - Dla każdej bramki: pusty GO → dodaj `Checkpoint.cs` + `BoxCollider` (Is Trigger = ON)
   - Ustaw `checkpointIndex` kolejno: 0, 1, 2, 3...
   - BoxCollider powinien obejmować całą szerokość pasa
3. Przeciągnij wszystkie Transform bramek do tablicy `checkpoints[]` w CheckpointManager
4. Stwórz punkt startowy i przeciągnij do `spawnTransform`

### 5. Przypisz referencje w CarAgent
- `checkpointManager` → GameObject z CheckpointManager.cs

### 6. Dla wielu agentów (zalecane!)
- Zduplikuj samochód 10-50 razy (więcej = szybszy trening)
- Każdy używa tego samego prefaba, tego samego BehaviorParameters
- Mogą być na tej samej scenie lub w osobnych środowiskach

---

## 🚀 Uruchomienie treningu

```bash
# Otwórz projekt w Unity, uruchom scenę (Play)
# Potem w terminalu:

cd twój_projekt
mlagents-learn config/car_training.yaml --run-id=CarRun01

# Wznów trening:
mlagents-learn config/car_training.yaml --run-id=CarRun01 --resume
```

## 📊 Śledzenie postępów (TensorBoard)
```bash
tensorboard --logdir results/
# Otwórz http://localhost:6006
```

## 🎮 Testowanie bez AI (ręczne sterowanie)
W `BehaviorParameters` → Behavior Type → `Heuristic Only`
Sterowanie: WASD lub strzałki

---

## 🔍 Architektura sieci
```
Input (12): [ray0..ray8, speed, cp_angle, cp_dot]
         ↓
Hidden Layer 1: 128 neuronów (ReLU)
         ↓
Hidden Layer 2: 128 neuronów (ReLU)
         ↓
Output (2): [steer, throttle] (tanh → -1..1)
```

## ⚙️ Obserwacje (wejście sieci, 12 wartości)
| # | Opis | Zakres |
|---|------|--------|
| 0-8 | Raycasts (9 kątów) | 0..1 (0=przeszkoda blisko) |
| 9 | Prędkość znorm. | 0..1 |
| 10 | Kąt do checkpointa | -1..1 |
| 11 | Dot product do CP | -1..1 |

## 🎯 Akcje (wyjście sieci, 2 wartości)
| # | Opis | Zakres |
|---|------|--------|
| 0 | Skręt | -1..1 |
| 1 | Gaz/Hamulec | -1..1 |

---

## ❗ Troubleshooting

**Agent się nie rusza / stoi w miejscu:**
→ Sprawdź czy BehaviorParameters → Space Size = 12
→ Dodaj curiosity reward w YAML (odkomentuj sekcję curiosity)

**Agent uderza w ściany i nie uczy się:**
→ Upewnij się że Wall/Obstacle mają odpowiednie tagi
→ Zwiększ max_steps do 5M w YAML

**Python error `mlagents not found`:**
→ `pip install mlagents` (Python 3.9-3.10!)
→ Użyj venv: `python -m venv mlagents_env`

**Unity error `Behavior name not found`:**
→ BehaviorParameters → Behavior Name musi być dokładnie `CarDriver`
