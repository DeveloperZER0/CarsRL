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
    public float motorTorque = 3500f;

    [Tooltip("Moment hamowania [Nm]")]
    public float brakeTorque = 5000f;

    [Tooltip("Hamowanie silnikiem gdy brak gazu [Nm] — ZERO = auto toczy się bez oporu")]
    public float engineBrakeTorque = 5f;

    [Tooltip("Maksymalna prędkość [m/s] (25 ≈ 90 km/h)")]
    public float maxSpeed = 25f;

    // ─── Steering ─────────────────────────────────────────────────────────────

    [Header("Steering")]
    [Tooltip("Maksymalny kąt skrętu kół przednich [°]")]
    public float maxSteeringAngle = 35f;

    [Tooltip("Redukcja kąta skrętu przy max prędkości (0 = brak redukcji, 1 = pełna)")]
    [Range(0f, 0.8f)]
    public float speedSteerReduction = 0.4f;

    [Tooltip("Szybkość interpolacji skrętu — wysoka = responsywny")]
    public float steeringSpeed = 25f;

    // ─── Physics ──────────────────────────────────────────────────────────────

    [Header("Physics")]

    [Tooltip("Współczynnik docisku aerodynamicznego [N/(m/s)²]")]
    public float downforceCoefficient = 8f;

    [Tooltip("Maksymalna prędkość obrotowa [rad/s] (ogranicza przewracanie)")]
    public float maxAngularVelocity = 4f;

    [Tooltip("Wysokość spawnu nad podłożem (zapobiega klipowaniu)")]
    public float spawnHeightOffset = 0.3f;

    [Tooltip("Offset środka masy w dół — stabilizuje auto")]
    public float centerOfMassYOffset = -0.5f;

    // ─── Internals ────────────────────────────────────────────────────────────

    private Rigidbody _rb;
    private float _currentSteer;
    private float _currentThrottle; // UWAGA: to jest surowy throttle z agenta, remapowany w ApplyMotor

    private Vector3 _startPosition;
    private Quaternion _startRotation;

    // Publiczne właściwości do odczytu (ML-Agents / UI)
    public float SpeedMs       => _rb.linearVelocity.magnitude;
    public float SpeedKmH      => SpeedMs * 3.6f;
    public bool  IsGrounded    => wheelRL.isGrounded || wheelRR.isGrounded || wheelFL.isGrounded || wheelFR.isGrounded;
    /// <summary>Prędkość w kierunku przodu (ujemna = cofanie)</summary>
    public float ForwardSpeed  => Vector3.Dot(_rb.linearVelocity, transform.forward);
    /// <summary>Czy auto jest do góry nogami (dot up z world up < 0.1)</summary>
    public bool  IsFlipped     => Vector3.Dot(transform.up, Vector3.up) < 0.1f;

    // ─── Unity Lifecycle ──────────────────────────────────────────────────────

    public float LateralSlip
    {
        get
        {
            // Pobieramy poślizg z obu tylnych kół i uśredniamy
            WheelHit hitRL, hitRR;
            float slip = 0f;
            int count  = 0;

            if (wheelRL.GetGroundHit(out hitRL)) { slip += Mathf.Abs(hitRL.sidewaysSlip); count++; }
            if (wheelRR.GetGroundHit(out hitRR)) { slip += Mathf.Abs(hitRR.sidewaysSlip); count++; }

            return count > 0 ? slip / count : 0f;
        }
    }

    public float GetSurfaceContactRatio(PhysicsMaterial material, string tag, LayerMask layers)
    {
        if (material == null && string.IsNullOrEmpty(tag) && layers.value == 0)
        {
            return 0f;
        }

        int hits = 0;
        int matches = 0;

        CountWheelSurface(wheelFL, material, tag, layers, ref hits, ref matches);
        CountWheelSurface(wheelFR, material, tag, layers, ref hits, ref matches);
        CountWheelSurface(wheelRL, material, tag, layers, ref hits, ref matches);
        CountWheelSurface(wheelRR, material, tag, layers, ref hits, ref matches);

        return hits > 0 ? (float)matches / hits : 0f;
    }

    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        _rb.interpolation  = RigidbodyInterpolation.Interpolate;
        _rb.maxAngularVelocity = maxAngularVelocity;

        // Obniżony środek masy — auto jest stabilniejsze, trudniej się przewraca
        Vector3 com = _rb.centerOfMass;
        com.y += centerOfMassYOffset;
        _rb.centerOfMass = com;

        _startPosition = transform.position;
        _startRotation = transform.rotation;
    }

    private void FixedUpdate()
    {
        ApplySteering();
        ApplyMotor();
        ApplyDownforce();
        ClampSpeed();
        UpdateWheelMeshes();
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Ustawia wejścia sterowania (wywoływane przez CarAgent).
    /// WAŻNE: throttle jest remapowany w ApplyMotor!
    ///   Agent wysyła [-1, +1], ale jest to traktowane jako:
    ///     [-1, 0) = hamulec (proporcjonalny)
    ///     [0, +1] = gaz (proporcjonalny)
    ///   Agent ZAWSZE jedzie do przodu, nie ma wstecznego.
    /// </summary>
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

        // Spawn lekko nad podłożem — zapobiega klipowaniu z gruntem
        Vector3 safePos = _startPosition + Vector3.up * spawnHeightOffset;
        _rb.position = safePos;
        _rb.rotation = _startRotation;
        transform.SetPositionAndRotation(safePos, _startRotation);

        // Wyzeruj WheelCollidery
        foreach (var wheel in new[] { wheelFL, wheelFR, wheelRL, wheelRR })
        {
            wheel.motorTorque  = 0f;
            wheel.brakeTorque  = 0f;
            wheel.steerAngle   = 0f;
        }

        _currentSteer    = 0f;
        _currentThrottle = 0f;
    }

    public void SetStartTransform(Vector3 pos, Quaternion rot)
    {
        _startPosition = pos;
        _startRotation = rot;
    }

    // ─── Private Methods ──────────────────────────────────────────────────────

    private void ApplyMotor()
    {
        // ═══════════════════════════════════════════════════════════════════
        // REMAPOWANIE THROTTLE:
        //   Agent wysyła _currentThrottle w zakresie [-1, +1]
        //   Mapujemy na:
        //     gas    = max(0, _currentThrottle)     → [0, 1]
        //     brake  = max(0, -_currentThrottle)    → [0, 1]
        //   Dzięki temu neutralna pozycja (0) = auto toczy się bez oporu
        //   a agent musi AKTYWNIE hamować (wartości ujemne)
        // ═══════════════════════════════════════════════════════════════════

        float gas   = Mathf.Max(0f, _currentThrottle);
        float brake = Mathf.Max(0f, -_currentThrottle);

        if (gas > 0.01f)
        {
            // ── Przyspieszanie ──
            float speedRatio  = Mathf.Clamp01(Mathf.Max(ForwardSpeed, 0f) / maxSpeed);
            float torqueCurve = Mathf.Lerp(1f, 0.1f, speedRatio * speedRatio);
            float torque      = gas * motorTorque * torqueCurve;

            // Napęd na wszystkie koła (4WD)
            wheelFL.motorTorque = torque;
            wheelFR.motorTorque = torque;
            wheelRL.motorTorque = torque;
            wheelRR.motorTorque = torque;

            SetBrakeTorque(0f);
        }
        else if (brake > 0.01f)
        {
            // ── Aktywne hamowanie ──
            SetMotorTorque(0f);
            SetBrakeTorque(brake * brakeTorque);
        }
        else
        {
            // ── Neutralna — minimalne hamowanie, auto się toczy ──
            SetMotorTorque(0f);
            SetBrakeTorque(engineBrakeTorque);
        }
    }

    private void ApplySteering()
    {
        // Redukcja kąta skrętu przy wyższej prędkości — stabilność
        float speedFactor = Mathf.Clamp01(SpeedMs / maxSpeed);
        float effectiveMaxAngle = maxSteeringAngle * (1f - speedFactor * speedSteerReduction);

        float steerTarget = _currentSteer * effectiveMaxAngle;

        // Szybka, responsywna interpolacja
        float smoothed = Mathf.Lerp(wheelFL.steerAngle, steerTarget,
                                    steeringSpeed * Time.fixedDeltaTime);

        wheelFL.steerAngle = smoothed;
        wheelFR.steerAngle = smoothed;
    }

    private void ApplyDownforce()
    {
        // Docisk rośnie kwadratowo z prędkością (jak w prawdziwym aucie)
        float force = downforceCoefficient * SpeedMs * SpeedMs;
        _rb.AddForce(-transform.up * force, ForceMode.Force);
    }

    /// <summary>
    /// Twardy limit prędkości — zapobiega nieskończonemu przyspieszaniu
    /// </summary>
    private void ClampSpeed()
    {
        if (SpeedMs > maxSpeed * 1.05f)
        {
            _rb.linearVelocity = _rb.linearVelocity.normalized * maxSpeed;
        }
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

    private void CountWheelSurface(
        WheelCollider wheel,
        PhysicsMaterial material,
        string tag,
        LayerMask layers,
        ref int hits,
        ref int matches)
    {
        if (wheel == null) return;

        if (wheel.GetGroundHit(out WheelHit hit))
        {
            hits++;
            if (IsSurfaceMatch(hit, material, tag, layers))
            {
                matches++;
            }
        }
    }

    private bool IsSurfaceMatch(WheelHit hit, PhysicsMaterial material, string tag, LayerMask layers)
    {
        Collider col = hit.collider;
        if (col == null) return false;

        if (material != null && col.sharedMaterial == material) return true;
        if (!string.IsNullOrEmpty(tag) && col.CompareTag(tag)) return true;
        if (layers.value != 0 && (layers.value & (1 << col.gameObject.layer)) != 0) return true;

        return false;
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