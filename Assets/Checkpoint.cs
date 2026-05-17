using UnityEngine;

/// <summary>
/// Skrypt bramki/checkpointa. Wykrywa przejazd samochodu i informuje agenta.
///
/// SETUP W UNITY:
/// 1. Dodaj ten skrypt do GameObject bramki.
/// 2. Dodaj BoxCollider z Is Trigger = true (szerokość trasy, niewidoczny).
/// 3. Ustaw index (numer kolejny tej bramki na trasie).
/// 4. Tag samochodu musi być "Agent" — na ROOT obiekcie auta.
/// </summary>
public class Checkpoint : MonoBehaviour
{
    [Tooltip("Numer tego checkpointa na trasie (0 = pierwszy)")]
    public int checkpointIndex = 0;

    [Tooltip("Tag obiektu samochodu (sprawdzany na ROOT)")]
    public string carTag = "Agent";

    private void OnTriggerEnter(Collider other)
    {
        // Sprawdzamy tag na ROOT obiekcie (nie na colliderze dziecka/koła!)
        // Bo tag "Agent" jest na głównym GameObject auta, a trigger wyzwala
        // kolizję z WheelColliderem lub child BoxColliderem
        Transform root = other.transform.root;

        bool tagOnSelf = other.CompareTag(carTag);
        bool tagOnRoot = root != null && root.CompareTag(carTag);

        if (!tagOnSelf && !tagOnRoot) return;

        // Pobierz agenta — szukaj na root, potem na parent, potem na self
        CarAgent agent = root.GetComponent<CarAgent>();
        if (agent == null)
            agent = other.GetComponentInParent<CarAgent>();
        if (agent == null)
            agent = other.GetComponent<CarAgent>();

        if (agent != null)
        {
            agent.OnCheckpointReached(checkpointIndex);
        }
        else
        {
            Debug.LogWarning($"[Checkpoint {checkpointIndex}] Obiekt '{other.name}' (root: '{root.name}') " +
                             $"ma tag '{carTag}', ale nie ma CarAgent! Sprawdź hierarchię prefabu.");
        }
    }

    // ─── Debug Gizmos ─────────────────────────────────────────────────────────

    private void OnDrawGizmos()
    {
        Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
        BoxCollider bc = GetComponent<BoxCollider>();
        if (bc != null)
        {
            Gizmos.matrix = transform.localToWorldMatrix;
            Gizmos.DrawCube(bc.center, bc.size);
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(bc.center, bc.size);
        }

#if UNITY_EDITOR
        UnityEditor.Handles.Label(
            transform.position + Vector3.up * 2f,
            $"[CP {checkpointIndex}]",
            new GUIStyle { normal = { textColor = Color.yellow }, fontSize = 14 }
        );
#endif
    }
}
