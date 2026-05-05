using UnityEngine;

/// <summary>
/// Kontroler auta oparty na WheelColliderach.
///
/// SETUP W UNITY:
/// 1. Główny GameObject: Rigidbody + ten skrypt
/// 2. Utwórz 4 puste child GameObject'y jako "koła fizyczne" (WheelColliders)
///    np: WheelFL, WheelFR, WheelRL, WheelRR
/// 3. Każdy z nich dostaje komponent WheelCollider
/// 4. Utwórz 4 child GameObject'y jako "koła wizualne" (meshe kół)
/// 5. Przypisz wszystko w Inspectorze
///
/// USTAWIENIA WHEELCOLLIDER (każde koło):
///  Mass:            20
///  Radius:          dopasuj do rozmiaru koła (np. 0.35)
///  Suspension Distance: 0.15
///  Spring: Spring: 25000, Damper: 2500, Target Position: 0.3
///  Forward Friction:  Extremum Slip 0.4 / Value 1.0 | Asymptote Slip 0.8 / Value 0.5 | Stiffness 1.5
///  Sideways Friction: Extremum Slip 0.2 / Value 1.0 | Asymptote Slip 0.5 / Value 0.7 | Stiffness 2.0
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class CarController : MonoBehaviour
{
    // ─── Wheel References ─────────────────────────────────────────────────────

    [Header("Wheel Colliders")]
    public WheelCollider wheelFL;
    public WheelCollider wheelFR;
    public WheelCollider wheelRL;
    public WheelCollider wheelRR;

    [Header("Wheel Meshes (Visual)")]
    public Transform meshFL;
    public Transform meshFR;
    public Transform meshRL;
    public Transform meshRR;

    [Header("Wheel Mesh Offsets")]
public Vector3 leftWheelOffset  = new Vector3(0f, 90f, 0f);
public Vector3 rightWheelOffset = new Vector3(0f, -90f, 0f);

    // ─── Engine ───────────────────────────────────────────────────────────────

    [Header("Engine")]
    [Tooltip("Maksymalny moment obrotowy silnika [Nm]")]
    public float motorTorque = 1500f;

    [Tooltip("Moment hamowania [Nm]")]
    public float brakeTorque = 3000f;

    [Tooltip("Hamowanie silnikiem gdy brak gazu [Nm]")]
    public float engineBrakeTorque = 300f;

    [Tooltip("Maksymalna prędkość [m/s]")]
    public float maxSpeed = 15f;

    // ─── Steering ─────────────────────────────────────────────────────────────

    [Header("Steering")]
    [Tooltip("Maksymalny kąt skrętu kół przednich [°]")]
    public float maxSteeringAngle = 30f;

    [Tooltip("Prędkość interpolacji skrętu (płynność)")]
    public float steeringSpeed = 5f;

    // ─── Physics ──────────────────────────────────────────────────────────────

    [Header("Physics")]

    [Tooltip("Docisk do podłoża rośnie z prędkością [N/(m/s)]")]
    public float downforcePerSpeed = 10f;

    // ─── Internals ────────────────────────────────────────────────────────────

    private Rigidbody _rb;
    private float _currentSteer;
    private float _currentThrottle;
    private float _steerTarget;
    private Vector3 _startPosition;
    private Quaternion _startRotation;

    // Publiczne właściwości do odczytu (ML-Agents / UI)
    public float SpeedMs       => _rb.linearVelocity.magnitude;
    public float SpeedKmH      => SpeedMs * 3.6f;
    public bool  IsGrounded    => wheelRL.isGrounded || wheelRR.isGrounded;

    // ─── Unity Lifecycle ──────────────────────────────────────────────────────

    public float LateralSlip
{
    get
    {
        // Pobieramy poślizg z obu tylnych kół i uśredniamy
        // sideways to boczny poślizg opony – to właśnie chcemy mierzyć
        WheelHit hitRL, hitRR;
        float slip = 0f;
        int count  = 0;

        if (wheelRL.GetGroundHit(out hitRL)) { slip += Mathf.Abs(hitRL.sidewaysSlip); count++; }
        if (wheelRR.GetGroundHit(out hitRR)) { slip += Mathf.Abs(hitRR.sidewaysSlip); count++; }

        return count > 0 ? slip / count : 0f;
    }
}

    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        _rb.interpolation  = RigidbodyInterpolation.Interpolate;

        _startPosition = transform.position;
        _startRotation = transform.rotation;
    }

    private void FixedUpdate()
    {
        ApplySteering();
        ApplyMotor();
        ApplyDownforce();
        UpdateWheelMeshes();
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Ustawia wejścia sterowania (wywoływane przez CarAgent).
    /// </summary>
    /// <param name="steer">-1 = lewo, +1 = prawo</param>
    /// <param name="throttle">-1 = hamulec/wsteczny, +1 = gaz</param>
    public void SetInputs(float steer, float throttle)
    {
        _currentSteer    = Mathf.Clamp(steer,    -1f, 1f);
        _currentThrottle = Mathf.Clamp(throttle, -1f, 1f);
    }

    public float GetNormalizedSpeed() => Mathf.Clamp01(SpeedMs / maxSpeed);

    public void ResetCar()
    {
        // Wyzeruj siły
        _rb.linearVelocity  = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;
        _rb.position        = _startPosition;
        _rb.rotation        = _startRotation;
        transform.SetPositionAndRotation(_startPosition, _startRotation);

        // Wyzeruj WheelCollidery
        foreach (var wheel in new[] { wheelFL, wheelFR, wheelRL, wheelRR })
        {
            wheel.motorTorque  = 0f;
            wheel.brakeTorque  = 0f;
            wheel.steerAngle   = 0f;
        }

        _currentSteer    = 0f;
        _currentThrottle = 0f;
        _steerTarget     = 0f;
    }

    public void SetStartTransform(Vector3 pos, Quaternion rot)
    {
        _startPosition = pos;
        _startRotation = rot;
    }

    // ─── Private Methods ──────────────────────────────────────────────────────

    private void ApplyMotor()
    {
        float forwardSpeed = Vector3.Dot(_rb.linearVelocity, transform.forward);

        if (_currentThrottle > 0f)
        {
            // ── Przyspieszanie ──
            if (SpeedMs < maxSpeed)
            {
                // Krzywa momentu: pełna siła do 50% prędkości, potem spada
                float speedRatio  = Mathf.Clamp01(SpeedMs / maxSpeed);
                float torqueCurve = Mathf.Pow(1f - speedRatio, 1.5f);
                float torque      = _currentThrottle * motorTorque * torqueCurve;

                // Napęd na wszystkie koła (4WD)
                // Zmień na wheelRL/RR jeśli chcesz tylnonapędowe (RWD → więcej driftu!)
                wheelFL.motorTorque = torque;
                wheelFR.motorTorque = torque;
                wheelRL.motorTorque = torque;
                wheelRR.motorTorque = torque;
            }
            else
            {
                // Osiągnięto maxSpeed – brak momentu
                SetMotorTorque(0f);
            }

            // Brak hamowania
            SetBrakeTorque(0f);
        }
        else if (_currentThrottle < 0f)
        {
            SetMotorTorque(0f);

            if (forwardSpeed > 0.5f)
            {
                // ── Hamowanie ──
                SetBrakeTorque(Mathf.Abs(_currentThrottle) * brakeTorque);
            }
            else
            {
                // ── Wsteczny (tylko tylne koła) ──
                SetBrakeTorque(0f);
                wheelRL.motorTorque = _currentThrottle * motorTorque * 0.5f;
                wheelRR.motorTorque = _currentThrottle * motorTorque * 0.5f;
                wheelFL.motorTorque = 0f;
                wheelFR.motorTorque = 0f;
            }
        }
        else
        {
            // ── Hamowanie silnikiem ──
            SetMotorTorque(0f);
            SetBrakeTorque(engineBrakeTorque);
        }
    }

    private void ApplySteering()
    {
        // Płynna interpolacja skrętu (bez szarpania)
        _steerTarget = _currentSteer * maxSteeringAngle;
        float smoothed = Mathf.Lerp(wheelFL.steerAngle, _steerTarget,
                                    steeringSpeed * Time.fixedDeltaTime);

        wheelFL.steerAngle = smoothed;
        wheelFR.steerAngle = smoothed;
    }

    private void ApplyDownforce()
    {
        // Docisk rośnie kwadratowo z prędkością (jak w prawdziwym aucie)
        float force = downforcePerSpeed * SpeedMs * SpeedMs;
        _rb.AddForce(-transform.up * force, ForceMode.Force);
    }

    /// <summary>
    /// Synchronizuje pozycję i rotację mesha wizualnego z WheelColliderem.
    /// </summary>
    private void UpdateWheelMeshes()
    {
        UpdateSingleMesh(wheelFL, meshFL, leftWheelOffset);
        UpdateSingleMesh(wheelFR, meshFR, rightWheelOffset);
        UpdateSingleMesh(wheelRL, meshRL, leftWheelOffset);
        UpdateSingleMesh(wheelRR, meshRR, rightWheelOffset);
    }

    private void UpdateSingleMesh(WheelCollider col, Transform mesh, Vector3 rotationOffset)
{
    if (mesh == null) return;
    col.GetWorldPose(out Vector3 pos, out Quaternion rot);
    mesh.SetPositionAndRotation(pos, rot * Quaternion.Euler(rotationOffset));
}

    private void SetMotorTorque(float torque)
    {
        wheelFL.motorTorque = torque;
        wheelFR.motorTorque = torque;
        wheelRL.motorTorque = torque;
        wheelRR.motorTorque = torque;
    }

    private void SetBrakeTorque(float torque)
    {
        wheelFL.brakeTorque = torque;
        wheelFR.brakeTorque = torque;
        wheelRL.brakeTorque = torque;
        wheelRR.brakeTorque = torque;
    }
}