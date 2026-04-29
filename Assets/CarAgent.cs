using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// ML-Agents Agent sterujący samochodem.
/// Algorytm: PPO (Proximal Policy Optimization) - domyślny i najlepszy dla ciągłych przestrzeni akcji.
/// 
/// OBSERWACJE (VectorSensor):
///   - 9 raycastów (kąty: -80, -60, -30, -15, 0, 15, 30, 60, 80 stopni) → odległość od przeszkód
///   - prędkość pojazdu (znormalizowana)
///   - kąt do następnego checkpointa (znormalizowany)
///   - dot product kierunku ruchu z kierunkiem do checkpointa
///   RAZEM: 12 wartości wejściowych
///
/// AKCJE (ciągłe):
///   - [0] Skręt:       -1 (lewo) do +1 (prawo)
///   - [1] Gaz/Hamulec: -1 (hamulec/wsteczny) do +1 (pełny gaz)
/// </summary>
[RequireComponent(typeof(CarController))]
public class CarAgent : Agent
{
    [Header("Sensors - Raycasts")]
    [Tooltip("Maksymalna długość każdego raycastu")]
    public float raycastDistance = 15f;

    [Tooltip("Wysokość, na której emitowane są raycaste (lokalnie od środka auta)")]
    public float raycastHeight = 0.3f;

    [Tooltip("Warstwy Unity wykrywane przez raycaste (np. Obstacles, Walls)")]
    public LayerMask raycastLayers = ~0;

    [Header("Checkpoints")]
    [Tooltip("Referencja do managera checkpointów na scenie")]
    public CheckpointManager checkpointManager;

    [Header("Rewards")]
    [Tooltip("Nagroda za zaliczenie checkpointa")]
    public float checkpointReward = 1.0f;

    [Tooltip("Kara za kolizję z przeszkodą")]
    public float collisionPenalty = -0.5f;

    [Tooltip("Kara za stanie w miejscu (anti-idle)")]
    public float idlePenalty = -0.005f;

    [Tooltip("Maksymalny czas epizodu w sekundach")]
    public float maxEpisodeTime = 60f;

    // Kąty raycastów (stopnie, 0 = prosto w przód)
    private readonly float[] _rayAngles = { -80f, -60f, -30f, -15f, 0f, 15f, 30f, 60f, 80f };

    private CarController _car;
    private int _nextCheckpointIndex;
    private float _episodeTimer;
    private float _stuckTimer;
    private Vector3 _lastPosition;
    private const float STUCK_THRESHOLD = 0.5f;
    private const float STUCK_TIME = 3f;

    // ─── Unity Lifecycle ───────────────────────────────────────────────────────

    public override void Initialize()
    {
        _car = GetComponent<CarController>();
    }

    // ─── ML-Agents Overrides ───────────────────────────────────────────────────

    /// <summary>
    /// Reset środowiska na początku każdego epizodu.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        _episodeTimer = 0f;
        _stuckTimer = 0f;
        _nextCheckpointIndex = 0;

        // Reset pozycji i fizyki samochodu
        _car.ResetCar();

        // Ustaw pozycję startową z managera
        if (checkpointManager != null)
        {
            Transform spawn = checkpointManager.GetSpawnTransform();
            transform.SetPositionAndRotation(spawn.position, spawn.rotation);
        }

        _lastPosition = transform.position;
    }

    /// <summary>
    /// Zbieranie obserwacji dla sieci neuronowej.
    /// Rozmiar wektora: 9 (raycasts) + 1 (prędkość) + 1 (kąt do CP) + 1 (dot) = 12
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // --- 9 obserwacji: raycasty ---
        Vector3 origin = transform.position + Vector3.up * raycastHeight;

        foreach (float angle in _rayAngles)
        {
            Vector3 dir = Quaternion.AngleAxis(angle, Vector3.up) * transform.forward;
            float normalizedDist;

            if (Physics.Raycast(origin, dir, out RaycastHit hit, raycastDistance, raycastLayers))
            {
                normalizedDist = hit.distance / raycastDistance;
                Debug.DrawRay(origin, dir * hit.distance, Color.red);
            }
            else
            {
                normalizedDist = 1f; // Brak przeszkody = pełna odległość
                Debug.DrawRay(origin, dir * raycastDistance, Color.green);
            }

            sensor.AddObservation(normalizedDist);
        }

        // --- 1 obserwacja: prędkość ---
        float speed = _car.GetNormalizedSpeed();
        sensor.AddObservation(speed);

        // --- 2 obserwacje: kierunek do checkpointa ---
        if (checkpointManager != null)
        {
            Vector3 cpPos = checkpointManager.GetCheckpointPosition(_nextCheckpointIndex);
            Vector3 toCP = (cpPos - transform.position).normalized;

            // Kąt (znormalizowany -1..1)
            float angle = Vector3.SignedAngle(transform.forward, toCP, Vector3.up) / 180f;
            sensor.AddObservation(angle);

            // Dot product (jak bardzo jedziemy w stronę CP)
            float dot = Vector3.Dot(transform.forward, toCP);
            sensor.AddObservation(dot);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    /// <summary>
    /// Wykonanie akcji zwróconych przez sieć neuronową.
    /// </summary>
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float steer = actionBuffers.ContinuousActions[0];      // -1..1
        float throttle = actionBuffers.ContinuousActions[1];   // -1..1

        _car.SetInputs(steer, throttle);

        // Timer epizodu
        _episodeTimer += Time.fixedDeltaTime;
        if (_episodeTimer >= maxEpisodeTime)
        {
            AddReward(-0.1f); // Mała kara za timeout
            EndEpisode();
            return;
        }

        // Kara za stanie w miejscu
        float distanceMoved = Vector3.Distance(transform.position, _lastPosition);
        if (distanceMoved < 0.1f * Time.fixedDeltaTime * 10f)
        {
            _stuckTimer += Time.fixedDeltaTime;
            if (_stuckTimer > STUCK_TIME)
            {
                AddReward(-0.5f);
                EndEpisode();
                return;
            }
        }
        else
        {
            _stuckTimer = 0f;
        }

        // Mała kara za bycie idle (zachęca do ruchu)
        AddReward(idlePenalty);

        // Nagroda za ruch w stronę checkpointa
        if (checkpointManager != null)
        {
            Vector3 cpPos = checkpointManager.GetCheckpointPosition(_nextCheckpointIndex);
            float dot = Vector3.Dot(transform.forward, (cpPos - transform.position).normalized);
            AddReward(dot * 0.001f * _car.GetNormalizedSpeed());
        }

        _lastPosition = transform.position;
    }

    /// <summary>
    /// Heurystyka dla trybu ręcznego (testowanie bez AI - klawisze WSAD/strzałki).
    /// </summary>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var cont = actionsOut.ContinuousActions;
        cont[0] = Input.GetAxis("Horizontal"); // A/D lub strzałki
        cont[1] = Input.GetAxis("Vertical");   // W/S lub strzałki
    }

    // ─── Kolizje ──────────────────────────────────────────────────────────────

    private void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag("Obstacle") || col.gameObject.CompareTag("Wall"))
        {
            AddReward(collisionPenalty);
            EndEpisode();
        }
    }

    // ─── Checkpointy (wywoływane przez Checkpoint.cs) ─────────────────────────

    public void OnCheckpointReached(int checkpointIndex)
    {
        if (checkpointIndex == _nextCheckpointIndex)
        {
            AddReward(checkpointReward);
            _nextCheckpointIndex++;
            _stuckTimer = 0f;

            // Zaliczono wszystkie checkpointy - koniec trasy!
            if (checkpointManager != null &&
                _nextCheckpointIndex >= checkpointManager.CheckpointCount)
            {
                AddReward(5f); // Duża nagroda za ukończenie trasy
                EndEpisode();
            }
        }
    }
}
