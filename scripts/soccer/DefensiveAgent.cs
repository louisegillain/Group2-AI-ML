using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class DefensiveAgent : AgentSoccer
{
    [Header("Defensive References")]
    public Transform ball;
    public Rigidbody ballRb;
    public Transform myGoal;
    public Transform opponentAgent; 
    public float fieldLength = 40f; 
    public LayerMask opponentLayer;

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
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
    
        float distanceToBall = Vector3.Distance(transform.position, ball.position);
        float fieldHalf = fieldLength / 2f;

        // distance to ball
        sensor.AddObservation(distanceToBall); 

        // opponent data 
        if (opponentAgent != null && opponentRb != null)
        {
            // nearest opponent position 
            sensor.AddObservation(transform.InverseTransformPoint(opponentAgent.position)); 

            // Opponent-Ball-Goal Alignment (Dot Product)
            Vector3 ballToMyGoal = (myGoal.position - ball.position).normalized;
            Vector3 oppToBall = (ball.position - opponentAgent.position).normalized;
            float alignment = Vector3.Dot(oppToBall, ballToMyGoal);
            sensor.AddObservation(alignment); 
        }
        else
        {
            sensor.AddObservation(Vector3.zero); 
            sensor.AddObservation(0f); 
        }

        //goal angle (own goal)
        Vector3 myGoalDirection = (myGoal.position - transform.position).normalized;
        sensor.AddObservation(transform.InverseTransformDirection(myGoalDirection));

        //normalized z-position 
        float normalizedZ = transform.localPosition.z / fieldHalf;
        sensor.AddObservation(normalizedZ);

        // ball-to-goal direction
        Vector3 ballToMyGoal_Dir = (myGoal.position - ball.position).normalized;
        sensor.AddObservation(ballToMyGoal_Dir); 

        // goal blocked status 
        bool isBlocked = Physics.Linecast(ball.position, myGoal.position, opponentLayer);
        sensor.AddObservation(isBlocked ? 1f : 0f);

        // agent's angular velocity 
        sensor.AddObservation(agentRb.angularVelocity);
        
        
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
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
        
        // example movement
        float forward = actions.ContinuousActions[1];
        float rotate = actions.ContinuousActions[2];
        
        Vector3 move = transform.forward * forward * MOVEMENT_MULTIPLIER;
        agentRb.AddForce(move, ForceMode.VelocityChange);
        transform.Rotate(transform.up, rotate * 5f); 
    }
}