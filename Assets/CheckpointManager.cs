using UnityEngine;

/// <summary>
/// Manager checkpointów - trzyma listę punktów trasy i pozycję startu.
/// 
/// SETUP W UNITY:
/// 1. Stwórz pusty GameObject "CheckpointManager" i dodaj ten skrypt.
/// 2. W tablicy checkpoints[] przeciągnij Transformy kolejnych bramek w kolejności trasy.
/// 3. W spawnTransform przeciągnij Transform punktu startowego.
/// 4. Każda bramka powinna mieć komponent Checkpoint.cs (trigger collider).
/// </summary>
public class CheckpointManager : MonoBehaviour
{
    [Header("Track Setup")]
    [Tooltip("Transformy checkpointów w kolejności trasy")]
    public Transform[] checkpoints;

    [Tooltip("Punkt startowy/respawn dla aut")]
    public Transform spawnTransform;

    [Tooltip("Czy losowo przesuwać punkt startu o mały offset (różnicuje agentów)")]
    public bool randomizeSpawn = true;

    [Tooltip("Maksymalny losowy offset pozycji startu")]
    public float spawnRandomRange = 0.5f;

    public int CheckpointCount => checkpoints?.Length ?? 0;

    // ─── Public API ───────────────────────────────────────────────────────────

    public Vector3 GetCheckpointPosition(int index)
    {
        if (checkpoints == null || checkpoints.Length == 0)
            return Vector3.zero;

        index = Mathf.Clamp(index, 0, checkpoints.Length - 1);
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
        }
    }
}
