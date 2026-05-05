using UnityEngine;

public class CameraController : MonoBehaviour
{
    [Header("Kamera i gracz")]
    public GameObject Camera;
    public GameObject Player;

    [Header("Pozycja względem gracza (x, y, z)")]
    public Vector3 positionOffset = new Vector3(0f, 2f, -8f);

    [Header("Rotacja względem gracza (x, y, z)")]
    public Vector3 rotationOffsetEuler = new Vector3(12f, 0f, 0f);

    [Header("Wygładzanie")]
    [Min(0.01f)]
    public float positionSmoothTime = 0.2f;
    [Range(0f, 20f)]
    public float rotationSmoothSpeed = 8f;

    [Header("Kolizja kamery")]
    [Tooltip("Warstwy z którymi kamera koliduje (ściany, podłogi itp.)")]
    public LayerMask collisionLayers = ~0;

    [Tooltip("Promień sfery sprawdzającej kolizję")]
    public float collisionRadius = 0.3f;

    [Tooltip("Minimalna odległość kamery od gracza")]
    public float minDistance = 1.0f;

    [Tooltip("Jak szybko kamera wraca na docelową pozycję po minięciu przeszkody")]
    public float returnSpeed = 4f;

    private Vector3 _currentVelocity;
    private float _currentDistance;      // aktualna odległość od gracza
    private float _targetDistance;       // docelowa odległość (bez przeszkód)

    private void Reset()
    {
        if (Camera == null) Camera = gameObject;
        if (Player == null) Player = GameObject.FindGameObjectWithTag("Player");
    }

    private void Awake()
    {
        // Inicjalizuj odległość na podstawie offsetu
        _currentDistance = positionOffset.magnitude;
        _targetDistance  = _currentDistance;
    }

    private void LateUpdate()
    {
        if (Camera == null || Player == null) return;

        Transform playerTransform = Player.transform;
        Transform cameraTransform = Camera.transform;

        // ─── Punkt startowy SphereCast ────────────────────────────────────────
        // Castujemy z wysokości oczu gracza (y + offset), nie z nóg
        Vector3 castOrigin = playerTransform.position + Vector3.up * positionOffset.y;

        // Docelowa pozycja kamery (bez kolizji)
        Vector3 desiredPosition = playerTransform.position
                                + playerTransform.TransformDirection(positionOffset);

        // Kierunek i maksymalna odległość SphereCasta
        Vector3 castDirection  = (desiredPosition - castOrigin).normalized;
        float   maxCastDist    = Vector3.Distance(castOrigin, desiredPosition);
        _targetDistance        = maxCastDist;

        // ─── SphereCast ───────────────────────────────────────────────────────
        if (Physics.SphereCast(
                castOrigin,
                collisionRadius,
                castDirection,
                out RaycastHit hit,
                maxCastDist,
                collisionLayers,
                QueryTriggerInteraction.Ignore))
        {
            // Przeszkoda znaleziona – przyciągnij kamerę bliżej gracza
            float safeDistance = Mathf.Max(hit.distance - collisionRadius, minDistance);
            _currentDistance   = Mathf.Min(_currentDistance, safeDistance);
        }
        else
        {
            // Brak przeszkody – płynny powrót na docelową odległość
            _currentDistance = Mathf.Lerp(
                _currentDistance,
                _targetDistance,
                returnSpeed * Time.deltaTime);
        }

        // ─── Finalna pozycja kamery ───────────────────────────────────────────
        Vector3 clampedPosition = castOrigin + castDirection * _currentDistance;

        cameraTransform.position = Vector3.SmoothDamp(
            cameraTransform.position,
            clampedPosition,
            ref _currentVelocity,
            positionSmoothTime);

        // ─── Rotacja (bez zmian) ──────────────────────────────────────────────
        Quaternion desiredRotation = playerTransform.rotation
                                   * Quaternion.Euler(rotationOffsetEuler);
        cameraTransform.rotation = Quaternion.Slerp(
            cameraTransform.rotation,
            desiredRotation,
            rotationSmoothSpeed * Time.deltaTime);
    }

    private void OnValidate()
    {
        positionSmoothTime   = Mathf.Max(0.01f, positionSmoothTime);
        rotationSmoothSpeed  = Mathf.Max(0f, rotationSmoothSpeed);
        collisionRadius      = Mathf.Max(0.05f, collisionRadius);
        minDistance          = Mathf.Max(0.5f, minDistance);
    }
}