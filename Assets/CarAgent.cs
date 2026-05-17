using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// ML-Agents Agent sterujący samochodem — v5 "ZERO TOLERANCJI".
///
/// Filozofia:
///   KAŻDA kolizja (ściana, przeszkoda, inne auto) = NATYCHMIASTOWY KONIEC EPIZODU + kara.
///   Agent NIE MA prawa się uczyć złych nawyków. Jedyne co się opłaca to czysta jazda.
///
/// OBSERWACJE (VectorSensor) — 25 wartości:
///   - 9 raycastów ścian/przeszkód (-70, -45, -25, -10, 0, 10, 25, 45, 70)
///   - 5 raycastów na inne auta (-60, -25, 0, 25, 60)
///   - 1 prędkość do przodu ze znakiem (-1..1)
///   - 1 prędkość bezwzględna (0..1)
///   - 2 kierunek do checkpointa (kąt + dot)
///   - 2 lokalna pozycja checkpointa (x, z)
///   - 1 dystans do checkpointa
///   - 2 kierunek do lookahead checkpointa (kąt + dot)
///   - 1 kontakt z lodem (0..1)
///   - 1 yaw rate
///
/// AKCJE (ciągłe):
///   - [0] Skręt:       -1 (lewo) do +1 (prawo)
///   - [1] Gaz/Hamulec: -1 (hamulec) do +1 (pełny gaz)
/// </summary>
[RequireComponent(typeof(CarController))]
public class CarAgent : Agent
{
    [Header("Sensors - Raycasts (Walls/Obstacles)")]
    [Tooltip("Maksymalna długość każdego raycastu na ściany")]
    public float raycastDistance = 25f;

    [Tooltip("Wysokość, na której emitowane są raycaste (lokalnie od środka auta)")]
    public float raycastHeight = 0.3f;

    [Tooltip("Warstwy Unity wykrywane przez raycaste ścian (NIE zawierać warstwy Cars!)")]
    public LayerMask raycastLayers = ~0;

    [Header("Sensors - Other Cars")]
    [Tooltip("Maksymalna odległość wykrywania innych aut")]
    public float carDetectionDistance = 20f;

    [Tooltip("Warstwy, na których znajdują się inne auta")]
    public LayerMask otherCarsLayers;

    [Tooltip("Tag innych aut")]
    public string otherCarsTag = "Agent";

    [Header("Ice")]
    [Tooltip("PhysicMaterial lodu (opcjonalnie)")]
    public PhysicsMaterial iceMaterial;

    [Tooltip("Tag lodu")]
    public string iceTag = "Ice";

    [Tooltip("Warstwy lodu")]
    public LayerMask iceLayers;

    [Header("Checkpoints")]
    [Tooltip("Referencja do managera checkpointów na scenie")]
    public CheckpointManager checkpointManager;

    [Tooltip("Normalizacja dystansu do checkpointa (metry)")]
    public float checkpointDistanceNormalization = 30f;

    [Tooltip("Ile checkpointów do przodu patrzymy w obserwacjach")]
    public int lookaheadCheckpointOffset = 2;

    [Header("Spawn")]
    [Tooltip("Minimalny dystans od innych aut przy respawnie")]
    public float spawnClearanceRadius = 8.0f;

    [Tooltip("Maksymalna liczba prób znalezienia wolnego miejsca")]
    public int spawnMaxAttempts = 40;

    [Header("═══ NAGRODY ═══")]

    [Tooltip("Nagroda za zaliczenie checkpointa")]
    public float checkpointReward = 3.0f;

    [Tooltip("Nagroda za ukończenie całej trasy")]
    public float trackCompleteReward = 15.0f;

    [Tooltip("Skala nagrody za zbliżanie się do checkpointa (za każdy metr)")]
    public float progressRewardScale = 0.05f;

    [Tooltip("EXISTENCE PENALTY — kara za każdy krok (wymusza szybkość!)")]
    public float existencePenalty = -0.002f;

    [Tooltip("Skala nagrody za prędkość w dobrym kierunku")]
    public float speedRewardScale = 0.01f;

    [Tooltip("Kara za cofanie")]
    public float reversePenalty = -0.01f;

    [Tooltip("Maksymalny czas epizodu w sekundach")]
    public float maxEpisodeTime = 90f;

    [Header("═══ KOLIZJE — ZERO TOLERANCJI ═══")]

    [Tooltip("Kara za KAŻDĄ kolizję (ściana/przeszkoda/auto) — natychmiast EndEpisode")]
    public float collisionPenalty = -1.0f;

    [Header("═══ ANTI-FREEZE — STANIE = NAJGORSZA OPCJA ═══")]

    [Tooltip("Kara za stuck (GORSZA niż kolizja! Agent woli spróbować jechać niż stać)")]
    public float stuckPenalty = -1.5f;

    [Tooltip("Prędkość poniżej której auto jest 'za wolne' [m/s]")]
    public float lowSpeedThreshold = 2.0f;

    [Tooltip("Kara za jazdę poniżej progu prędkości (za krok) — eskalująca")]
    public float lowSpeedPenalty = -0.01f;

    // Kąty raycastów na ściany
    private readonly float[] _rayAngles = { -70f, -45f, -25f, -10f, 0f, 10f, 25f, 45f, 70f };

    // Kąty raycastów na inne auta
    private readonly float[] _carRayAngles = { -60f, -25f, 0f, 25f, 60f };

    private CarController _car;
    private Rigidbody _rb;
    private int _nextCheckpointIndex;
    private float _episodeTimer;
    private float _stuckTimer;
    private float _previousDistToCheckpoint;
    private float _flipTimer;
    private float _iceContactRatio;
    private bool _episodeEnding; // guard against multiple collision callbacks in same frame

    private const float STUCK_SPEED = 0.5f;
    private const float STUCK_TIME = 1.5f;  // Skrócone z 3s — szybki reset!
    private const float FLIP_TIME = 1.5f;
    private float _lowSpeedTimer;

    // ─── Unity Lifecycle ───────────────────────────────────────────────────────

    public override void Initialize()
    {
        _car = GetComponent<CarController>();
        _rb = GetComponent<Rigidbody>();
    }

    // ─── ML-Agents Overrides ───────────────────────────────────────────────────

    public override void OnEpisodeBegin()
    {
        _episodeTimer = 0f;
        _stuckTimer = 0f;
        _flipTimer = 0f;
        _lowSpeedTimer = 0f;
        _nextCheckpointIndex = 0;
        _episodeEnding = false;

        if (checkpointManager != null)
        {
            Vector3 spawnPos = checkpointManager.GetSpawnPositionAvoidingOverlap(
                otherCarsLayers,
                otherCarsTag,
                spawnClearanceRadius,
                spawnMaxAttempts,
                transform.root);
            Quaternion spawnRot = checkpointManager.GetSpawnTransform().rotation;

            _car.SetStartTransform(spawnPos, spawnRot);
        }

        _car.ResetCar();

        // Inicjalizuj dystans do pierwszego checkpointa
        if (checkpointManager != null && checkpointManager.CheckpointCount > 0)
        {
            Vector3 cpPos = checkpointManager.GetCheckpointPosition(0);
            _previousDistToCheckpoint = Vector3.Distance(transform.position, cpPos);
        }
        else
        {
            _previousDistToCheckpoint = 0f;
        }
    }

    /// <summary>
    /// Zbieranie obserwacji — 25 wartości.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 origin = transform.position + Vector3.up * raycastHeight;

        // ── 9 obserwacji: raycasty na ściany/przeszkody ──
        foreach (float angle in _rayAngles)
        {
            Vector3 dir = Quaternion.AngleAxis(angle, Vector3.up) * transform.forward;
            float normalizedDist;

            if (Physics.Raycast(origin, dir, out RaycastHit hit, raycastDistance, raycastLayers, QueryTriggerInteraction.Ignore))
            {
                normalizedDist = hit.distance / raycastDistance;
                Debug.DrawRay(origin, dir * hit.distance, Color.red);
            }
            else
            {
                normalizedDist = 1f;
                Debug.DrawRay(origin, dir * raycastDistance, Color.green);
            }

            sensor.AddObservation(normalizedDist);
        }

        // ── 5 obserwacji: raycasty na inne auta ──
        foreach (float angle in _carRayAngles)
        {
            Vector3 dir = Quaternion.AngleAxis(angle, Vector3.up) * transform.forward;
            float normalizedDist = 1f;

            if (otherCarsLayers.value != 0)
            {
                if (Physics.Raycast(origin, dir, out RaycastHit carHit, carDetectionDistance, otherCarsLayers, QueryTriggerInteraction.Ignore))
                {
                    if (carHit.transform.root != transform.root)
                    {
                        normalizedDist = carHit.distance / carDetectionDistance;
                        Debug.DrawRay(origin, dir * carHit.distance, Color.magenta);
                    }
                }
            }

            if (normalizedDist >= 1f)
            {
                Debug.DrawRay(origin, dir * carDetectionDistance, new Color(0.5f, 0f, 0.5f, 0.3f));
            }

            sensor.AddObservation(normalizedDist);
        }

        // ── 1: prędkość do przodu ze znakiem ──
        sensor.AddObservation(Mathf.Clamp(_car.ForwardSpeed / _car.maxSpeed, -1f, 1f));

        // ── 1: prędkość bezwzględna ──
        sensor.AddObservation(_car.GetNormalizedSpeed());

        // ── 7: checkpoint + lookahead ──
        if (checkpointManager != null && checkpointManager.CheckpointCount > 0)
        {
            Vector3 cpPos = checkpointManager.GetCheckpointPosition(_nextCheckpointIndex);
            Vector3 toCPRaw = cpPos - transform.position;
            Vector3 toCP = toCPRaw.normalized;

            sensor.AddObservation(Vector3.SignedAngle(transform.forward, toCP, Vector3.up) / 180f);
            sensor.AddObservation(Vector3.Dot(transform.forward, toCP));

            Vector3 localCP = transform.InverseTransformPoint(cpPos);
            float cpNorm = Mathf.Max(checkpointDistanceNormalization, 0.001f);

            sensor.AddObservation(Mathf.Clamp(localCP.x / cpNorm, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(localCP.z / cpNorm, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp01(toCPRaw.magnitude / cpNorm));

            // Lookahead
            int cpCount = checkpointManager.CheckpointCount;
            if (lookaheadCheckpointOffset > 0)
            {
                int lookaheadIndex = Mathf.Clamp(
                    _nextCheckpointIndex + lookaheadCheckpointOffset,
                    0, cpCount - 1);
                Vector3 laPos = checkpointManager.GetCheckpointPosition(lookaheadIndex);
                Vector3 toLA = (laPos - transform.position).normalized;
                sensor.AddObservation(Vector3.SignedAngle(transform.forward, toLA, Vector3.up) / 180f);
                sensor.AddObservation(Vector3.Dot(transform.forward, toLA));
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }
        else
        {
            for (int i = 0; i < 7; i++)
                sensor.AddObservation(0f);
        }

        // ── 1: kontakt z lodem ──
        _iceContactRatio = _car.GetSurfaceContactRatio(iceMaterial, iceTag, iceLayers);
        sensor.AddObservation(_iceContactRatio);

        // ── 1: yaw rate ──
        sensor.AddObservation(Mathf.Clamp(_rb.angularVelocity.y / 5f, -1f, 1f));
    }

    /// <summary>
    /// System nagród v6 — "ZERO TOLERANCJI + ANTI-FREEZE":
    ///
    ///   ✦ EXISTENCE PENALTY  (-0.002/krok)  → Musi skończyć tor szybko.
    ///   ✦ PROGRESS            (+0.05/metr)   → Za każdy metr bliżej checkpointa.
    ///   ✦ SPEED × DIRECTION   (+0.01/krok)   → Jedź szybko w dobrym kierunku.
    ///   ✦ CHECKPOINT           +3.0           → Nagroda za bramkę.
    ///   ✦ FINISH              +15.0           → Mega bonus za ukończenie trasy.
    ///   ✦ REVERSE             -0.01/krok      → Nie cofaj.
    ///   ✦ KOLIZJA             -1.0 + END      → KAŻDA kolizja = natychmiastowy koniec.
    ///   ✦ FLIP                -1.0 + END      → Przewrócenie = koniec.
    ///   ✦ STUCK               -1.5 + END      → GORSZE niż kolizja! Stanie = najgorsze.
    ///   ✦ LOW SPEED           -0.01/krok      → Eskalująca kara za wolną jazdę.
    ///
    /// KLUCZOWY BALANS:
    ///   Stanie 1.5s = existence(-0.15) + lowSpeed(-0.75) + stuck(-1.5) = -2.4
    ///   Kolizja     = -1.0
    ///   → Agent ZAWSZE woli spróbować jechać (ryzyko -1.0) niż stać (-2.4)
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (_episodeEnding) return;

        float steer    = actionBuffers.ContinuousActions[0];
        float throttle = actionBuffers.ContinuousActions[1];
        _car.SetInputs(steer, throttle);

        // ── Timer epizodu ──
        _episodeTimer += Time.fixedDeltaTime;
        if (_episodeTimer >= maxEpisodeTime)
        {
            EndEpisode();
            return;
        }

        // ── Flip detection ──
        if (_car.IsFlipped)
        {
            _flipTimer += Time.fixedDeltaTime;
            if (_flipTimer > FLIP_TIME)
            {
                _episodeEnding = true;
                AddReward(-1f);
                EndEpisode();
                return;
            }
        }
        else
        {
            _flipTimer = 0f;
        }

        // ── Stuck detection — GORSZE niż kolizja! ──
        if (_car.SpeedMs < STUCK_SPEED)
        {
            _stuckTimer += Time.fixedDeltaTime;
            if (_stuckTimer > STUCK_TIME)
            {
                _episodeEnding = true;
                AddReward(stuckPenalty); // -1.5! GORSZE niż crash (-1.0)
                EndEpisode();
                return;
            }
        }
        else
        {
            _stuckTimer = 0f;
        }

        // ═══════════════════════════════════════════════════════════════
        // EXISTENCE PENALTY
        // ═══════════════════════════════════════════════════════════════
        AddReward(existencePenalty);

        // ═══════════════════════════════════════════════════════════════
        // LOW SPEED PENALTY — eskalująca kara za wolną jazdę
        // Im dłużej jedzie wolno, tym większa kara.
        // Stanie w miejscu jest GWARANTOWANĄ stratą.
        // ═══════════════════════════════════════════════════════════════
        if (_car.SpeedMs < lowSpeedThreshold && _episodeTimer > 0.5f)
        {
            _lowSpeedTimer += Time.fixedDeltaTime;
            // Eskalacja: im dłużej stoi, tym gorzej (1x po 0s, 2x po 1s, 3x po 2s...)
            float escalation = 1f + _lowSpeedTimer;
            AddReward(lowSpeedPenalty * escalation);
        }
        else
        {
            _lowSpeedTimer = Mathf.Max(0f, _lowSpeedTimer - Time.fixedDeltaTime * 2f); // powolny reset
        }

        // ═══════════════════════════════════════════════════════════════
        // PROGRESS — nagroda za zbliżanie się do checkpointa
        // ═══════════════════════════════════════════════════════════════
        if (checkpointManager != null && checkpointManager.CheckpointCount > 0)
        {
            Vector3 cpPos = checkpointManager.GetCheckpointPosition(_nextCheckpointIndex);
            float currentDist = Vector3.Distance(transform.position, cpPos);

            float distDelta = _previousDistToCheckpoint - currentDist;
            AddReward(distDelta * progressRewardScale);

            _previousDistToCheckpoint = currentDist;

            // ═══════════════════════════════════════════════════════════════
            // SPEED × DIRECTION — nagroda za szybką jazdę w kierunku CP
            // ═══════════════════════════════════════════════════════════════
            Vector3 toCP = (cpPos - transform.position).normalized;
            float dot = Vector3.Dot(transform.forward, toCP);
            float speedNorm = Mathf.Clamp01(_car.ForwardSpeed / _car.maxSpeed);

            if (dot > 0f && speedNorm > 0.05f)
            {
                AddReward(dot * speedNorm * speedRewardScale);
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // KARA: Cofanie
        // ═══════════════════════════════════════════════════════════════
        if (_car.ForwardSpeed < -0.3f)
        {
            AddReward(reversePenalty);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var cont = actionsOut.ContinuousActions;
        cont[0] = Input.GetAxis("Horizontal");
        cont[1] = Input.GetAxis("Vertical");
    }

    // ─── Kolizje — ZERO TOLERANCJI ───────────────────────────────────────────

    private void OnCollisionEnter(Collision col)
    {
        if (_episodeEnding) return;

        // Ignoruj kolizje z podłożem (droga)
        if (IsGroundCollision(col)) return;

        // KAŻDA inna kolizja = natychmiastowy koniec
        // Ściana, przeszkoda, inne auto — bez wyjątków
        if (IsOtherCarCollision(col) ||
            col.gameObject.CompareTag("Obstacle") ||
            col.gameObject.CompareTag("Wall"))
        {
            _episodeEnding = true;
            AddReward(collisionPenalty);
            EndEpisode();
        }
    }

    /// <summary>
    /// Sprawdza czy kolizja jest z podłożem/drogą (ignorujemy ją).
    /// Podłoże to obiekty z tagiem "Road", "Ground", "Untagged"
    /// lub na warstwie Default bez żadnego z naszych tagów.
    /// </summary>
    private bool IsGroundCollision(Collision col)
    {
        if (col == null) return false;
        GameObject go = col.gameObject;

        // Jeśli obiekt ma tag ściany/przeszkody/auta — to NIE jest podłoże
        if (go.CompareTag("Wall")) return false;
        if (go.CompareTag("Obstacle")) return false;
        if (!string.IsNullOrEmpty(otherCarsTag) && go.CompareTag(otherCarsTag)) return false;

        // Sprawdź czy to nie inne auto po warstwie
        if (otherCarsLayers.value != 0 &&
            (otherCarsLayers.value & (1 << go.layer)) != 0)
        {
            return false;
        }

        // Dodatkowe tagi podłoża — jeśli masz, sprawdź
        if (go.CompareTag("Ground") || (!string.IsNullOrEmpty(iceTag) && go.CompareTag(iceTag)))
            return true;

        // Sprawdź kontakt normalną — jeśli normalna wskazuje w górę, to podłoże
        if (col.contactCount > 0)
        {
            Vector3 avgNormal = Vector3.zero;
            for (int i = 0; i < col.contactCount; i++)
            {
                avgNormal += col.GetContact(i).normal;
            }
            avgNormal /= col.contactCount;

            // Normalna wskazuje w górę (>45°) = podłoże
            if (Vector3.Dot(avgNormal, Vector3.up) > 0.7f)
            {
                return true;
            }
        }

        return false;
    }

    private bool IsOtherCarCollision(Collision col)
    {
        if (col == null) return false;
        if (col.transform.root == transform.root) return false;

        bool layerMatch = otherCarsLayers.value != 0 &&
                          (otherCarsLayers.value & (1 << col.gameObject.layer)) != 0;
        bool tagMatch = !string.IsNullOrEmpty(otherCarsTag) && col.gameObject.CompareTag(otherCarsTag);

        return layerMatch || tagMatch;
    }

    // ─── Checkpointy ─────────────────────────────────────────────────────────

    public void OnCheckpointReached(int checkpointIndex)
    {
        if (_episodeEnding) return;

        int totalCheckpoints = checkpointManager != null ? checkpointManager.CheckpointCount : -1;
        Debug.Log($"[{name}] CP reached: got={checkpointIndex}, expecting={_nextCheckpointIndex}, total={totalCheckpoints}");

        if (checkpointIndex == _nextCheckpointIndex)
        {
            AddReward(checkpointReward);
            _nextCheckpointIndex++;
            _stuckTimer = 0f;
            _lowSpeedTimer = 0f;

            Debug.Log($"[{name}] ✓ CP {checkpointIndex} ZALICZONY! Reward +{checkpointReward}. Next={_nextCheckpointIndex}/{totalCheckpoints}");

            // Aktualizuj dystans do nowego checkpointa
            if (checkpointManager != null && _nextCheckpointIndex < checkpointManager.CheckpointCount)
            {
                Vector3 nextCpPos = checkpointManager.GetCheckpointPosition(_nextCheckpointIndex);
                _previousDistToCheckpoint = Vector3.Distance(transform.position, nextCpPos);
            }

            // Ukończył trasę!
            if (checkpointManager != null &&
                _nextCheckpointIndex >= checkpointManager.CheckpointCount)
            {
                Debug.Log($"[{name}] ★★★ TRASA UKOŃCZONA! Reward +{trackCompleteReward}. EndEpisode().");
                _episodeEnding = true;
                AddReward(trackCompleteReward);
                EndEpisode();
            }
        }
        else
        {
            Debug.LogWarning($"[{name}] ✗ CP {checkpointIndex} POMINIĘTY (oczekiwano {_nextCheckpointIndex})");
        }
    }
}
