using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Manager checkpointów - trzyma listę punktów trasy i pozycję startu.
/// 
/// SETUP W UNITY:
/// 1. Stwórz pusty GameObject "CheckpointManager" i dodaj ten skrypt.
/// 2. W tablicy checkpoints[] przeciągnij Transformy kolejnych bramek w kolejności trasy.
/// 3. W spawnTransform przeciągnij Transform punktu startowego.
/// 4. Każda bramka powinna mieć komponent Checkpoint.cs (trigger collider).
/// 
/// SPAWN SYSTEM v2 — Grid Stagger:
///   Zamiast losowego spawnowania wokół jednego punktu, auta rozkładane są
///   wzdłuż kierunku trasy z regularnymi odstępami i lekkim bocznym offsetem.
///   Dzięki temu auta nie spawnują się na sobie nawet przy wielu agentach.
/// </summary>
public class CheckpointManager : MonoBehaviour
{
    [Header("Track Setup")]
    [Tooltip("Transformy checkpointów w kolejności trasy")]
    public Transform[] checkpoints;

    [Tooltip("Punkt startowy/respawn dla aut")]
    public Transform spawnTransform;

    [Header("Spawn Settings")]
    [Tooltip("Czy losowo przesuwać punkt startu o mały offset (różnicuje agentów)")]
    public bool randomizeSpawn = true;

    [Tooltip("Maksymalny losowy offset pozycji startu")]
    public float spawnRandomRange = 3.0f;

    [Tooltip("Odstęp między autami wzdłuż trasy przy grid spawn")]
    public float gridSpawnSpacing = 5.0f;

    [Tooltip("Maksymalny boczny offset w grid spawn")]
    public float gridSpawnLateralRange = 2.0f;

    [Tooltip("Dodatkowy losowy jitter pozycji (metrów)")]
    public float gridSpawnJitter = 0.5f;

    [Tooltip("Czy spawnować na checkpointach zamiast na jednym punkcie (rozrzuca auta po trasie)")]
    public bool spawnAlongTrack = false;

    [Tooltip("Ile pierwszych checkpointów może być użyte jako spawn (0 = wszystkie)")]
    public int spawnCheckpointRange = 5;

    public int CheckpointCount => checkpoints?.Length ?? 0;

    // Licznik spawniętych aut w bieżącej "fali" — resetowany co pewien czas
    private int _spawnCounter = 0;
    private float _lastSpawnResetTime = -999f;
    private const float SPAWN_RESET_INTERVAL = 2f; // Reset co 2 sekundy

    // ─── Public API ───────────────────────────────────────────────────────────

    public Vector3 GetCheckpointPosition(int index)
    {
        if (checkpoints == null || checkpoints.Length == 0)
            return Vector3.zero;

        if (index < 0 || index >= checkpoints.Length)
        {
            Debug.LogWarning($"CheckpointManager: index {index} out of range [0, {checkpoints.Length - 1}]");
            index = Mathf.Clamp(index, 0, checkpoints.Length - 1);
        }
        return checkpoints[index].position;
    }

    public Transform GetSpawnTransform()
    {
        if (spawnTransform == null)
            return transform;

        return spawnTransform;
    }

    public Vector3 GetRandomizedSpawnPosition()
    {
        Vector3 basePos = spawnTransform != null ? spawnTransform.position : Vector3.zero;

        if (randomizeSpawn)
        {
            basePos += new Vector3(
                Random.Range(-spawnRandomRange, spawnRandomRange),
                0f,
                Random.Range(-spawnRandomRange, spawnRandomRange)
            );
        }

        return basePos;
    }

    /// <summary>
    /// Główna metoda spawnowania — próbuje znaleźć wolne miejsce.
    /// 
    /// Algorytm v2 — Grid Stagger:
    ///   1. Jeśli spawnAlongTrack=true → wybiera losowy checkpoint jako bazę
    ///   2. W przeciwnym razie → rozkłada auta w siatce wzdłuż spawn forward
    ///   3. Sprawdza kolizje OverlapSphere
    ///   4. Fallback: deterministyczny grid daleko od bazy
    /// </summary>
    public Vector3 GetSpawnPositionAvoidingOverlap(
        LayerMask carLayers,
        string carTag,
        float clearanceRadius,
        int maxAttempts,
        Transform ignoreRoot)
    {
        if (clearanceRadius <= 0f)
        {
            return GetRandomizedSpawnPosition();
        }

        // Reset spawn counter jeśli minęło wystarczająco czasu
        // (nowa "fala" epizodów)
        if (Time.time - _lastSpawnResetTime > SPAWN_RESET_INTERVAL)
        {
            _spawnCounter = 0;
            _lastSpawnResetTime = Time.time;
        }

        Vector3 basePos = spawnTransform != null ? spawnTransform.position : transform.position;
        Vector3 spawnForward = spawnTransform != null ? spawnTransform.forward : Vector3.forward;
        Vector3 spawnRight = spawnTransform != null ? spawnTransform.right : Vector3.right;

        int myIndex = _spawnCounter;
        _spawnCounter++;

        // ── Strategia 1: Spawn wzdłuż checkpointów ──
        if (spawnAlongTrack && checkpoints != null && checkpoints.Length > 1)
        {
            Vector3 trackCandidate = GetTrackSpawnPosition(myIndex, carLayers, carTag, clearanceRadius, ignoreRoot);
            if (!IsSpawnOverlapping(trackCandidate, clearanceRadius, carLayers, carTag, ignoreRoot))
            {
                return trackCandidate;
            }
        }

        // ── Strategia 2: Grid Stagger wzdłuż spawnForward ──
        // Każde auto dostaje deterministyczne miejsce w siatce + mały jitter
        Vector3 gridCandidate = GetGridSpawnPosition(myIndex, basePos, spawnForward, spawnRight);
        if (!IsSpawnOverlapping(gridCandidate, clearanceRadius, carLayers, carTag, ignoreRoot))
        {
            return gridCandidate;
        }

        // ── Strategia 3: Losowe próby z rosnącym promieniem ──
        int attempts = Mathf.Max(1, maxAttempts);
        for (int i = 0; i < attempts; i++)
        {
            float expandedRange = spawnRandomRange + (i * clearanceRadius * 0.5f);

            Vector3 candidate = basePos
                + spawnForward * Random.Range(-expandedRange, expandedRange)
                + spawnRight   * Random.Range(-expandedRange * 0.5f, expandedRange * 0.5f);

            if (!IsSpawnOverlapping(candidate, clearanceRadius, carLayers, carTag, ignoreRoot))
            {
                return candidate;
            }
        }

        // ── Fallback: deterministyczny grid daleko od bazy ──
        float fallbackDistance = gridSpawnSpacing * (myIndex + 1) + clearanceRadius * 2f;
        float lateralOffset = ((myIndex % 3) - 1) * gridSpawnLateralRange;
        return basePos
            + spawnForward * fallbackDistance
            + spawnRight * lateralOffset;
    }

    // ─── Private Methods ──────────────────────────────────────────────────────

    /// <summary>
    /// Grid Stagger: rozkłada auta w siatce wzdłuż trasy
    /// Auto 0: pozycja 0, offset boczny 0
    /// Auto 1: pozycja +spacing, offset boczny +lateral
    /// Auto 2: pozycja +2*spacing, offset boczny -lateral
    /// itd.
    /// </summary>
    private Vector3 GetGridSpawnPosition(int index, Vector3 basePos, Vector3 forward, Vector3 right)
    {
        float forwardOffset = index * gridSpawnSpacing;

        // Szachownica boczna: -1, 0, +1, -1, 0, +1, ...
        int lateralPattern = (index % 3) - 1;
        float lateralOffset = lateralPattern * gridSpawnLateralRange;

        // Mały losowy jitter żeby nie było identycznie
        Vector3 jitter = Vector3.zero;
        if (gridSpawnJitter > 0f)
        {
            jitter = new Vector3(
                Random.Range(-gridSpawnJitter, gridSpawnJitter),
                0f,
                Random.Range(-gridSpawnJitter, gridSpawnJitter));
        }

        Vector3 rawPos = basePos + forward * forwardOffset + right * lateralOffset + jitter;
        return ValidateGroundPosition(rawPos, basePos);
    }

    /// <summary>
    /// Spawn wzdłuż trasy — rozkłada auta na różnych checkpointach
    /// </summary>
    private Vector3 GetTrackSpawnPosition(int index, LayerMask carLayers, string carTag, float clearanceRadius, Transform ignoreRoot)
    {
        if (checkpoints == null || checkpoints.Length == 0)
            return spawnTransform != null ? spawnTransform.position : Vector3.zero;

        int maxCP = spawnCheckpointRange > 0
            ? Mathf.Min(spawnCheckpointRange, checkpoints.Length)
            : checkpoints.Length;

        // Deterministyczny checkpoint + losowy offset
        int cpIndex = index % maxCP;
        if (checkpoints[cpIndex] == null)
            cpIndex = 0;

        Vector3 cpPos = checkpoints[cpIndex].position;

        // Kierunek trasy w tym punkcie
        Vector3 trackDir = Vector3.forward;
        if (cpIndex + 1 < checkpoints.Length && checkpoints[cpIndex + 1] != null)
        {
            trackDir = (checkpoints[cpIndex + 1].position - cpPos).normalized;
        }
        else if (cpIndex > 0 && checkpoints[cpIndex - 1] != null)
        {
            trackDir = (cpPos - checkpoints[cpIndex - 1].position).normalized;
        }

        Vector3 trackRight = Vector3.Cross(Vector3.up, trackDir).normalized;

        // Offset wzdłuż trasy i w bok
        float forwardJitter = Random.Range(-2f, 2f);
        float lateralJitter = Random.Range(-gridSpawnLateralRange, gridSpawnLateralRange);

        Vector3 rawPos = cpPos + trackDir * forwardJitter + trackRight * lateralJitter;
        return ValidateGroundPosition(rawPos, cpPos);
    }

    /// <summary>
    /// Rzuca Raycast w dół z dużej wysokości, żeby znaleźć faktyczną wysokość trasy
    /// Zapobiega spawnowaniu aut "pod mapą" (out of bounds)
    /// </summary>
    private Vector3 ValidateGroundPosition(Vector3 position, Vector3 safeFallback)
    {
        // Rzucamy promień z 10 metrów nad przewidywaną pozycją w dół
        Vector3 rayOrigin = position + Vector3.up * 10f;
        
        // Szukamy jakiegokolwiek collidera (warstwy Default, Road, Ground itp)
        // Ignorujemy auta, żeby nie spawnować jednych na drugich
        int layerMask = ~(1 << LayerMask.NameToLayer("Cars")); 
        
        if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hit, 20f, layerMask, QueryTriggerInteraction.Ignore))
        {
            // Zwracamy pozycję na ziemi
            return hit.point;
        }

        // Jeśli raycast nic nie trafił, oznacza to że punkt jest POZA MAPĄ
        // Wracamy do w pełni bezpiecznej pozycji bazowej (środek bramki)
        return safeFallback;
    }

    private bool IsSpawnOverlapping(
        Vector3 position,
        float radius,
        LayerMask carLayers,
        string carTag,
        Transform ignoreRoot)
    {
        if (carLayers.value == 0 && string.IsNullOrEmpty(carTag)) return false;

        int mask = carLayers.value != 0 ? carLayers.value : ~0;
        Collider[] hits = Physics.OverlapSphere(position, radius, mask, QueryTriggerInteraction.Ignore);

        foreach (Collider col in hits)
        {
            if (col == null) continue;
            if (ignoreRoot != null && col.transform.root == ignoreRoot) continue;
            if (IsCarCollider(col, carLayers, carTag)) return true;
        }

        return false;
    }

    private bool IsCarCollider(Collider col, LayerMask carLayers, string carTag)
    {
        if (col == null) return false;

        if (carLayers.value != 0 && (carLayers.value & (1 << col.gameObject.layer)) != 0)
        {
            return true;
        }

        if (!string.IsNullOrEmpty(carTag))
        {
            if (col.CompareTag(carTag)) return true;
            Transform root = col.transform.root;
            if (root != null && root.CompareTag(carTag)) return true;
        }

        return false;
    }

    // ─── Debug Gizmos ─────────────────────────────────────────────────────────

    private void OnDrawGizmos()
    {
        if (checkpoints == null) return;

        for (int i = 0; i < checkpoints.Length; i++)
        {
            if (checkpoints[i] == null) continue;

            // Rysuj bramki
            Gizmos.color = (i == 0) ? Color.green : Color.yellow;
            Gizmos.DrawWireSphere(checkpoints[i].position, 0.5f);
            Gizmos.DrawWireCube(checkpoints[i].position, new Vector3(5f, 1f, 0.2f));

            // Numery
#if UNITY_EDITOR
            UnityEditor.Handles.Label(
                checkpoints[i].position + Vector3.up * 1.5f,
                $"CP {i}"
            );
#endif

            // Linia do następnego
            if (i < checkpoints.Length - 1 && checkpoints[i + 1] != null)
            {
                Gizmos.color = Color.cyan;
                Gizmos.DrawLine(checkpoints[i].position, checkpoints[i + 1].position);
            }
        }

        // Spawn point
        if (spawnTransform != null)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(spawnTransform.position, 0.8f);

            // Wizualizacja grid spawn
            Vector3 fwd = spawnTransform.forward;
            Vector3 right = spawnTransform.right;
            Gizmos.color = new Color(0f, 1f, 0f, 0.3f);
            for (int i = 0; i < 6; i++)
            {
                Vector3 gp = GetGridSpawnPosition(i, spawnTransform.position, fwd, right);
                Gizmos.DrawWireSphere(gp, 1.5f);
            }
        }
    }
}
