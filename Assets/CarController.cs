using UnityEngine;

/// <summary>
/// Prosty kontroler ruchu auta bez WheelCollider.
/// Steruje calym obiektem (Rigidbody) na podstawie wejsc: predkosc i skret.
///
/// SETUP W UNITY:
/// 1. Dodaj ten skrypt do GameObject z Rigidbody
/// 2. Dodaj collider do karoserii (np. BoxCollider)
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class CarController : MonoBehaviour
{
    [Header("Car Physics")]
    [Tooltip("Maksymalna predkosc ruchu (m/s)")]
    public float maxSpeed = 12f;

    [Tooltip("Predkosc skretu (stopnie na sekunde)")]
    public float steeringSpeed = 120f;

    [Tooltip("Obniżenie środka ciężkości (zapobiega przewracaniu)")]
    public float centerOfMassOffset = -0.5f;

    [Header("Grounding")]
    [Tooltip("Warstwy uznawane za podloze")]
    public LayerMask groundLayers = ~0;

    [Tooltip("Start raycastu nad ziemia (m)")]
    public float groundCheckOffset = 0.5f;

    [Tooltip("Maksymalny zasieg sprawdzania podloza (m)")]
    public float groundCheckDistance = 1.2f;

    [Tooltip("Jak szybko dopasowac obrot do normalnej podloza")]
    public float groundAlignSpeed = 12f;

    [Tooltip("Dodatkowy docisk do podloza (m/s^2)")]
    public float downforce = 30f;

    private Rigidbody _rb;
    private float _currentSteer;
    private float _currentThrottle;
    private Vector3 _startPosition;
    private Quaternion _startRotation;
    private bool _isGrounded;
    private Vector3 _groundNormal = Vector3.up;

    // ─── Unity Lifecycle ───────────────────────────────────────────────────────

    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();

        // Obniżenie środka masy - kluczowe dla stabilności!
        // _rb.centerOfMass = new Vector3(0f, centerOfMassOffset, 0f);

        _startPosition = transform.position;
        _startRotation = transform.rotation;
    }

    private void FixedUpdate()
    {
        UpdateGrounding();
        ApplyInputs();
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Ustawia wejścia sterowania (wywoływane przez CarAgent).
    /// </summary>
    /// <param name="steer">-1 = pełny skręt w lewo, +1 = pełny skręt w prawo</param>
    /// <param name="throttle">-1 = hamowanie/wstecz, +1 = pełny gaz</param>
    public void SetInputs(float steer, float throttle)
    {
        _currentSteer = Mathf.Clamp(steer, -1f, 1f);
        _currentThrottle = Mathf.Clamp(throttle, -1f, 1f);
    }

    /// <summary>
    /// Zwraca znormalizowaną prędkość (0..1) do obserwacji.
    /// </summary>
    public float GetNormalizedSpeed()
    {
        return Mathf.Clamp01(_rb.linearVelocity.magnitude / maxSpeed);
    }

    /// <summary>
    /// Reset samochodu do pozycji startowej (wywoływane przez CarAgent.OnEpisodeBegin).
    /// </summary>
    public void ResetCar()
    {
        // Stop fizyki
        _rb.linearVelocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;

        // Reset transform
        transform.SetPositionAndRotation(_startPosition, _startRotation);

        // Reset wejść
        _currentSteer = 0f;
        _currentThrottle = 0f;

    }

    /// <summary>
    /// Aktualizuje pozycję startową (np. gdy Manager wylosuje spawn point).
    /// </summary>
    public void SetStartTransform(Vector3 pos, Quaternion rot)
    {
        _startPosition = pos;
        _startRotation = rot;
    }

    // ─── Private Methods ──────────────────────────────────────────────────────

    private void ApplyInputs()
    {
        // Ruch do przodu/tylu
        Vector3 forward = _isGrounded
            ? Vector3.ProjectOnPlane(transform.forward, _groundNormal).normalized
            : transform.forward;
        if (forward.sqrMagnitude < 0.001f)
        {
            forward = transform.forward;
        }

        Vector3 move = forward * (_currentThrottle * maxSpeed * Time.fixedDeltaTime);
        _rb.MovePosition(_rb.position + move);

        // Skret wokol osi Y
        float turn = _currentSteer * steeringSpeed * Time.fixedDeltaTime;
        Quaternion turnRot = Quaternion.AngleAxis(turn, _isGrounded ? _groundNormal : Vector3.up);
        Quaternion targetRot = turnRot * _rb.rotation;

        if (_isGrounded)
        {
            Quaternion alignRot = Quaternion.FromToRotation(transform.up, _groundNormal) * targetRot;
            targetRot = Quaternion.Slerp(_rb.rotation, alignRot, groundAlignSpeed * Time.fixedDeltaTime);
        }

        _rb.MoveRotation(targetRot);
    }

    private void UpdateGrounding()
    {
        Vector3 origin = transform.position + Vector3.up * groundCheckOffset;
        float rayLength = groundCheckDistance + groundCheckOffset;

        if (Physics.Raycast(origin, Vector3.down, out RaycastHit hit, rayLength, groundLayers,
            QueryTriggerInteraction.Ignore))
        {
            _isGrounded = true;
            _groundNormal = hit.normal;

            if (downforce > 0f)
            {
                _rb.AddForce(-_groundNormal * downforce, ForceMode.Acceleration);
            }
        }
        else
        {
            _isGrounded = false;
            _groundNormal = Vector3.up;
        }
    }
}
