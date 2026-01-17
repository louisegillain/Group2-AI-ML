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

    float m_KickPower;

    public float m_Existential;

    const float k_Power = 2000f;

    public Transform teammate;
    private int stepsSinceLastTouch = 0;
    private Rigidbody d_agentRb;
    private Rigidbody opponentRb; 
    
    private const float MOVEMENT_MULTIPLIER = 10f; // scales the resultant force for singnificant output

    public override void Initialize()
    {
        base.Initialize();
        if (ball != null && ballRb) ballRb = ball.GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        if (ball == null || teammate == null) return;
    
        float fieldHalf = fieldLength / 2f;

        // distance to ball
        float distanceToBall = Vector3.Distance(transform.position, ball.position);
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

        sensor.AddObservation(stepsSinceLastTouch); 
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
        MoveAgent(actions.DiscreteActions);
        
    }
    private void OnCollisionEnter(Collision c){
        var force = k_Power * m_KickPower;
            if (position == Position.Goalie)
                {
                    force = k_Power;
                }
            if (c.gameObject.CompareTag("ball"))
                {
                    //AddReward(.2f * m_BallTouch);
                    var dir = (c.contacts[0].point - transform.position).normalized;
                    c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force);
                }

        if(c.gameObject.CompareTag("ball"))
        {
            stepsSinceLastTouch = 0;
        }
    }
}