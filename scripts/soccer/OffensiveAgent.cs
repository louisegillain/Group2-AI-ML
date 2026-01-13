using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class OffensiveAgent: AgentSoccer
{
    public Transform ball;
    public Rigidbody ballRb;
    public Transform teammate;
    public Transform opponentGoal; 
    public Transform opponentAgent;
    public float fieldLength = 40f;

    private int stepsSinceLastTouch = 0;
    private float lastAction = 0;  

    private Rigidbody agentRb;
    private Rigidbody opponentRb;

    private const float MOVEMENT_MULTIPLIER = 10f; // scales the resultant force for singnificant output

    public override void Initialize()
    {
        SoccerEnvController envController = GetComponentInParent<SoccerEnvController>();
        if (envController != null)
        {
            m_Existential = 1f / envController.MaxEnvironmentSteps;
        }
        else
        {
            m_Existential = 1f / MaxStep;
        }

        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();
        if (m_BehaviorParameters.TeamId == (int)Team.Blue)
        {
            team = Team.Blue;
            initialPos = new Vector3(transform.position.x - 5f, .5f, transform.position.z);
            rotSign = 1f;
        }
        else
        {
            team = Team.Purple;
            initialPos = new Vector3(transform.position.x + 5f, .5f, transform.position.z);
            rotSign = -1f;
        }
        if (position == Position.Goalie)
        {
            m_LateralSpeed = 1.0f;
            m_ForwardSpeed = 1.0f;
        }
        else if (position == Position.Striker)
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.3f;
        }
        else
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.0f;
        }
        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        agentRb = GetComponent<Rigidbody>();
        agentRb.maxAngularVelocity = 500;

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    public override void OnEpisodeBegin()
    {
        m_BallTouch = m_ResetParams.GetWithDefault("ball_touch", 0);

        stepsSinceLastTouch = 0;
        lastAction = 0;

    }

    //
    public override void CollectObservations(VectorSensor sensor)
    {
        
        float fieldHalf = fieldLength / 2f;

        //ball distance
        float distanceToBall = Vector3.Distance(transform.position, ball.position);
        sensor.AddObservation(distanceToBall); 

        // teammate relative position (calculated with vector difference) 
        sensor.AddObservation(transform.InverseTransformPoint(teammate.position)); 

        // goal angle (opponent)
        Vector3 oppGoalDirection = (opponentGoal.position - transform.position).normalized;
        sensor.AddObservation(transform.InverseTransformDirection(oppGoalDirection));

        // 4. Normalized X-Position
        // Normalized position from -1 (own side) to +1 (opponent side).
        float normalizedZ = transform.localPosition.z / fieldHalf;
        sensor.AddObservation(normalizedZ); 

        //opponent position 
        if (opponentAgent != null && opponentRb != null)
        {
            // nearest opponent distance (records only one so adequate for both games)
            sensor.AddObservation(Vector3.Distance(transform.position, opponentAgent.position));

            // opponent relative velocity
            sensor.AddObservation(transform.InverseTransformDirection(opponentRb.velocity));
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(Vector3.zero); 
        }

        //ball velocity 
        sensor.AddObservation(ballRb != null ? ballRb.velocity : Vector3.zero);

        //time since last touch 
        sensor.AddObservation(stepsSinceLastTouch);

    }

    public override void OnActionReceived(ActionBuffers actions){
        stepsSinceLastTouch++;

        if (position == Position.Goalie)
        {
            // Existential bonus for Goalies.
            AddReward(m_Existential);
        }
        else if (position == Position.Striker)
        {
            // Existential penalty for Strikers
            AddReward(-m_Existential);
        }
        MoveAgent(actionBuffers.DiscreteActions);

        //example movement
        float forward = actions.ContinuousActions[1];
        float rotate = actions.ContinuousActions[2];
        
        Vector3 move = transform.forward * forward * MOVEMENT_MULTIPLIER;
        agentRb.AddForce(move, ForceMode.VelocityChange);
        transform.Rotate(transform.up, rotate * 5f); 
        
        lastAction = actions.ContinuousActions[0];} 

        private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.transform == ball)
        {
            stepsSinceLastTouch = 0;
        }
    }
        
    



}